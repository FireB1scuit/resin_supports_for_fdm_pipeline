"""Stage 3, resin style — the SLA scaffold, dimensioned for FDM.

This is the structure resin slicers build, and the one this project exists to
make printable on filament. It is *not* an FDM organic tree: nothing fuses into
a thickening trunk that wanders down to the plate. An SLA support is a set of
discrete components, and it keeps its shape all the way down::

    contact ── tip ── arm ── shaft ── base
                (angled)  (vertical, thin)  (on the plate)

* **tip** — a small cone that meets the surface at roughly a right angle and
  penetrates it slightly, so it snaps off leaving a dot rather than a scar.
* **arm** — the short angled link from the shaft up to the tip. Several arms
  can fan off one shaft; that is how resin slicers keep the shaft count down
  (Lychee calls it parenting, or "optimize supports").
* **shaft** — ``shaft_upper_diameter`` at the top and ``shaft_lower_diameter``
  at the bottom. A slight taper, not a trunk. Straight and vertical whenever it
  can be, which is nearly always; where the model is in the way it leans around
  it instead of stopping on it (:func:`_route_to_plate`).
* **base** — the footprint on the plate: a disc of ``foot_diameter``, then a
  flare in to the shaft. On a lifted model the discs of neighbouring supports
  overlap into what amounts to a raft.
* **base tip** — when a shaft has to land on the model instead of the plate, it
  ends in a tip there too, so it snaps off at both ends rather than fusing.

Shafts are then **cross-linked into a scaffold**. That is what carries a real
model: a lattice of struts bracing each other, rather than a field of
independent sticks each balancing alone.

The one thing that cannot carry over
------------------------------------
Resin cross-links are usually horizontal. A vat printer does not care — every
layer is supported by the resin around it. An FDM printer cannot bridge a
horizontal strut hanging in mid-air at all, so ours are placed at a chosen
diagonal inside the printable band. Same job, same lattice, an angle the nozzle
can actually lay down. Adapting exactly this kind of thing is the whole point
of the project.

Getting down there
------------------
Collision and reachability come from :class:`rsupport.avoidance.AvoidanceField`,
which is independent of what gets built on top of it. Three things follow from
it, and they are the whole of how a support decides where to go:

1. **A shaft routes around the model rather than into or onto it.**
   :func:`_route_to_plate` walks the field's reachability maps down layer by
   layer, staying put where the layer below still allows it and sliding to the
   nearest place that does where it does not. Nothing directly below a contact
   is a reason to give up, only a reason to lean.
2. **Arms and cross-links are checked too.** They are placed by geometry rather
   than routed through the field, so :func:`_strut_clear` asks about each one.
   A shaft has never been the only thing that could cross a sculpt: an arm that
   leaves a shaft standing on the wrong side of the model is a spear straight
   through it.
3. **The plate is the only landing, by default.** ``params.plate_only`` is on,
   so a contact with no route down is refused and reported rather than propped
   off the sculpt. Turning it off allows the last-resort landing in
   :func:`_drop_shaft` — a shaft that stops where it is, tip first. Supports
   that *start* on the model are not implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
import shapely
import trimesh
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely.ops import nearest_points

from .avoidance import AvoidanceField, strut_lean
from .mesh_io import concat
from .raycast import DownRay
from .supports import (
    LineXY,
    PathXY,
    foot_profile,
    make_strut,
    mesh_rings,
    rings_on_axis,
    tip_profile,
)
from .types import SupportBuild, SupportParams, SupportPoint

__all__ = ["build_resin", "Shaft", "Arm"]

_EPS = 1e-9

#: How many neighbours one shaft will brace against. More than this and the
#: scaffold becomes a solid wall that is miserable to cut off. It counts
#: neighbours, not struts: a ladder of rungs up one pair is still one neighbour.
_LINKS_PER_SHAFT = 3

#: Most rungs up any one pair of shafts, however tall they are.
_MAX_RUNGS = 6


@dataclass
class Arm:
    """The angled link from a shaft up to one contact point."""

    contact: np.ndarray
    normal: np.ndarray
    attach: np.ndarray  # where it leaves the shaft
    elbow: np.ndarray  # where the arm ends and the tip begins


@dataclass
class Shaft:
    """One strut, its arms, and where it stands.

    ``path`` is the axis, bottom to top, as ``(x, y, z)`` rows. A shaft with a
    clear column below it is two rows and runs dead straight, which is what an
    SLA shaft normally does. Where the model is in the way the path picks up a
    row per collision layer and steps around it instead — see
    :func:`_route_to_plate`.
    """

    path: np.ndarray
    on_model: bool
    arms: list[Arm] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        pts = np.atleast_2d(np.asarray(self.path, dtype=np.float64))
        self.path = pts[np.argsort(pts[:, 2], kind="stable")]

    @property
    def xy(self) -> np.ndarray:
        """Where the shaft *tops out*. Arms leave from here."""
        return self.path[-1, :2]

    @property
    def base_xy(self) -> np.ndarray:
        """Where it stands. Not the same place once it has routed around."""
        return self.path[0, :2]

    @property
    def top_z(self) -> float:
        return float(self.path[-1, 2])

    @property
    def land_z(self) -> float:
        return float(self.path[0, 2])

    @property
    def height(self) -> float:
        return max(0.0, self.top_z - self.land_z)

    @property
    def detours(self) -> int:
        """How many times the axis actually stepped sideways."""
        if len(self.path) < 2:
            return 0
        step = np.linalg.norm(np.diff(self.path[:, :2], axis=0), axis=1)
        return int((step > 1e-9).sum())

    def xy_at(self, z: float) -> np.ndarray:
        return PathXY(self.path)(z)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def build_resin(
    mesh,
    points,
    params: SupportParams | None = None,
    ray: DownRay | None = None,
    field: AvoidanceField | None = None,
) -> SupportBuild:
    """Build an SLA-style scaffold under `points`."""
    params = params or SupportParams()
    points = [p for p in points]
    if not points:
        return SupportBuild(mesh=trimesh.Trimesh(), n_points=0)

    ray = ray or DownRay(mesh)
    top_z = max(float(p.position[2]) for p in points)
    field = field or AvoidanceField(mesh, params, top_z=top_z)

    shafts, dropped, warnings = _plan_shafts(field, ray, points, params)

    parts: list[trimesh.Trimesh] = []
    for shaft in shafts:
        parts.extend(_shaft_meshes(shaft, field, params))

    links, n_links = _link_shafts(field, ray, shafts, params)
    parts.extend(links)

    if any(s.on_model for s in shafts):
        n = sum(1 for s in shafts if s.on_model)
        warnings.append(
            f"{n} shaft(s) stand on the model; the plate was not reachable from there. "
            "They end in a tip, so they still snap off"
        )

    parts = [m for m in parts if m is not None and len(m.faces)]
    return SupportBuild(
        mesh=concat(*parts),
        n_points=sum(len(s.arms) for s in shafts),
        n_braces=n_links,
        dropped=dropped,
        warnings=warnings,
    )


def _strut_clear(
    field: AvoidanceField,
    ray: DownRay,
    a,
    b,
    radius: float,
    params: SupportParams,
) -> bool:
    """Does a strut from `a` to `b` stay out of the model?

    Shafts get this for free — they are routed through the avoidance field, so
    they cannot be anywhere else. Arms and cross-links are placed by geometry
    rather than by the field, so they have to be asked, and until they were the
    scaffold's one hard guarantee had a hole in it big enough to run a 26 mm
    strut through the middle of a sculpt.

    Two tests, because they catch different things. The parity test is the
    guarantee: nowhere on the strut may be *inside* the model, full stop. The
    field test is the manners: keep ``xy_clearance`` off the surface as well, so
    a strut does not graze what it passes. The top ``tip_length`` of the run is
    exempt from the second one only — that end is on its way to touch the
    model, and being close to it there is the entire point.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    span = float(np.linalg.norm(b - a))
    if span <= _EPS:
        return True

    n = int(np.clip(math.ceil(span / max(field.pitch * 0.5, 1e-3)), 4, 64))
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]  # endpoints touch by design
    pts = a + (b - a) * t[:, None]

    if bool(ray.inside(pts).any()):
        return False

    graze_from = max(a[2], b[2]) - params.tip_length
    probe = pts[pts[:, 2] <= graze_from]
    if not len(probe):
        return True

    bucket = field.bucket(radius)
    layers = np.array([field.layer_of(z) for z in probe[:, 2]])
    for layer in np.unique(layers):
        sel = probe[layers == layer]
        region = field.free(bucket, int(layer))
        if region is None or region.is_empty:
            return False
        if not bool(shapely.contains_xy(region, sel[:, 0], sel[:, 1]).all()):
            return False
    return True


def _elbow(pt: SupportPoint, params: SupportParams, ray: DownRay | None = None):
    """The joint between tip and arm, where the tip's own run ends.

    A resin tip leaves the surface roughly perpendicular, so on a downward
    overhang the elbow sits directly below the contact and on a wall it stands
    off to the side — which is exactly the SLA look.

    Everything below is measured from here, never from the contact. The tip has
    already covered that horizontal step, and charging the arm a vertical rise
    for the same distance is what made contacts near the plate impossible to
    support: their shaft top came out below the build plate.

    Given a `ray`, the tip is also *aimed*: the natural direction is tried
    first, and if the model is in the way of it the tip is swung around the
    contact — steeper, then off to either side — until it finds a line that is
    clear. That is the cheap half of relocating an awkward contact, and it is
    the half that keeps the contact exactly where sampling put it. Returns None
    when every direction is blocked, which makes the point genuinely
    unsupportable from below and it is dropped.
    """
    contact = np.asarray(pt.position, dtype=np.float64)
    normal = np.asarray(pt.normal, dtype=np.float64)
    if ray is None:
        return contact - _tip_axis(normal, params)

    for axis in _tip_candidates(normal, params):
        elbow = contact - axis
        if elbow[2] <= 0.0:
            continue
        if not ray.segment_blocked([elbow], [contact])[0]:
            return elbow
    return None


def _tip_candidates(normal: np.ndarray, params: SupportParams):
    """Tip directions to try, best first.

    The surface normal is what a resin slicer would use and is always first.
    The rest lean the tip progressively further away from it — up towards
    vertical, then rotated around the contact — which is what gets a tip in
    under a lip that its own normal points straight into.
    """
    yield _tip_axis(normal, params)

    n = np.asarray(normal, dtype=np.float64)
    flat = np.array([-n[0], -n[1], 0.0])
    norm = float(np.linalg.norm(flat))
    if norm > 1e-9:
        flat /= norm
        for deg in (30.0, 60.0, -30.0, -60.0):
            rad = math.radians(deg)
            c, s = math.cos(rad), math.sin(rad)
            turned = np.array([flat[0] * c - flat[1] * s, flat[0] * s + flat[1] * c, 0.0])
            yield _tip_axis(-np.array([turned[0], turned[1], -0.35]), params)

    # Straight up: nothing else to try, and on a downward overhang it is the
    # direction the arm would rather have anyway.
    yield np.array([0.0, 0.0, params.tip_length])


def _plan_shafts(field: AvoidanceField, ray: DownRay, points, params: SupportParams):
    """Group contacts onto shared shafts and work out where each shaft stands."""
    warnings: list[str] = []
    dropped: list[SupportPoint] = []

    kept, elbow_list = [], []
    for pt in points:
        elbow = _elbow(pt, params, ray)
        if elbow is None:
            dropped.append(pt)
            continue
        kept.append(pt)
        elbow_list.append(elbow)
    n_unreachable = len(dropped)

    if not kept:
        warnings.append(f"{len(dropped)} contact point(s) could not be reached by a tip")
        return [], dropped, warnings

    points = kept
    elbows = np.asarray(elbow_list, dtype=np.float64)
    groups = _group_contacts(elbows, points, params)

    shafts: list[Shaft] = []
    for members in groups:
        shaft = _make_shaft(field, ray, [points[i] for i in members], elbows[members], params)
        if shaft is None:
            # A shared shaft did not fit. Try each contact on its own before
            # giving up on it — one awkward neighbour should not cost the rest.
            for i in members:
                solo = _make_shaft(field, ray, [points[i]], elbows[[i]], params)
                if solo is None:
                    dropped.append(points[i])
                else:
                    shafts.append(solo)
        else:
            shafts.append(shaft)

    shafts = _absorb_neighbours(field, ray, shafts, params)

    n_stranded = len(dropped) - n_unreachable
    if n_unreachable:
        warnings.append(
            f"{n_unreachable} contact point(s) could not be reached by a tip from any direction"
        )
    if n_stranded:
        detail = (
            "no collision-free route to the plate"
            if params.plate_only
            else "nowhere to stand a shaft"
        )
        warnings.append(f"{n_stranded} contact point(s) left unsupported: {detail}")
    if params.plate_only and n_stranded:
        warnings.append(
            "supports are restricted to the build plate; starting them on the model "
            "is not implemented yet"
        )
    return shafts, dropped, warnings


def _absorb_neighbours(
    field: AvoidanceField, ray: DownRay, shafts: list[Shaft], params: SupportParams
) -> list[Shaft]:
    """Fold shafts that are practically touching into one.

    Grouping happens per contact, so two contacts that could not share a shaft
    on the way down can still end up standing a shaft each half a millimetre
    apart. Two shafts closer together than their own diameter are one shaft with
    extra steps: they waste resin, they double the number of feet to snap off,
    and — because a cross-link needs real distance to span — they leave nothing
    for the scaffold to brace against.

    Neighbours are neighbours *in three dimensions*. Looking only at where two
    shafts stand is what let a 2 mm stub holding up a lifted model's underside
    adopt the arms of a contact 25 mm above it: the two tops are nowhere near
    each other, the arm ends up leaving the host at the host's top rather than
    at the height it asked for, and what gets built is a 25 mm near-vertical
    spear straight through the sculpt. So an arm only moves if it still leaves
    the host at close to its intended height, and only if the strut it would
    become is clear of the model.
    """
    if len(shafts) < 2:
        return shafts

    reach = params.shaft_lower_diameter * 1.5
    # Match on where the shafts *top out*, not on where they stand: that is
    # where the arms leave from, and once shafts route around the model the two
    # can be centimetres apart.
    pos = np.array([s.xy for s in shafts])
    kdt = cKDTree(pos)
    r_arm = params.arm_diameter * 0.5

    # Tallest and most-loaded first: those are the ones worth keeping.
    order = sorted(range(len(shafts)), key=lambda i: (-len(shafts[i].arms), -shafts[i].height))
    gone = [False] * len(shafts)
    kept: list[Shaft] = []

    for i in order:
        if gone[i]:
            continue
        host = shafts[i]
        gone[i] = True
        for j in kdt.query_ball_point(pos[i], reach):
            if gone[j]:
                continue
            other = shafts[j]
            # Never trade a plate landing for a model one.
            if other.on_model != host.on_model and not other.on_model:
                continue
            # Tops too far apart in Z: these are not the same shaft seen twice,
            # and pretending otherwise is what builds the spear.
            if abs(other.top_z - host.top_z) > params.tip_length:
                continue
            moved = []
            for arm in other.arms:
                d = float(np.linalg.norm(arm.elbow[:2] - host.xy))
                attach_z = arm.elbow[2] - _arm_rise(d, params)
                if attach_z < host.land_z + params.layer_height:
                    break  # this arm cannot reach from over here
                # It leaves at the shaft top when the shaft tops out below the
                # height it wanted. Bounded, or the arm stops being an arm.
                if attach_z - host.top_z > params.tip_length:
                    break
                attach = np.array([host.xy[0], host.xy[1], min(attach_z, host.top_z)])
                if not _strut_clear(field, ray, attach, arm.elbow, r_arm, params):
                    break
                moved.append(
                    Arm(contact=arm.contact, normal=arm.normal, attach=attach, elbow=arm.elbow)
                )
            else:
                host.arms.extend(moved)
                gone[j] = True
        kept.append(host)
    return kept


def _group_contacts(elbows: np.ndarray, points, params: SupportParams) -> list[np.ndarray]:
    """Cluster contacts that can share one shaft.

    Parenting is bounded by geometry, not taste: an arm may only lean
    ``arm_angle_deg`` off vertical, so a contact far from the shaft needs a long
    vertical run to reach it. Past a point the shaft would have to start below
    the plate. `parenting` scales how far within that limit we are willing to go.
    """
    strength = float(np.clip(params.parenting, 0.0, 1.0))
    if strength <= 0.0 or len(points) < 2:
        return [np.array([i]) for i in range(len(points))]

    # Keep the reach short. An arm that stretches far has to start far down,
    # which turns the shaft into a stub and leaves nothing tall enough to brace
    # against. Resin supports have long shafts and short arms, not the reverse.
    radius = params.tip_length + strength * params.support_spacing
    tree = cKDTree(elbows[:, :2])
    heights = elbows[:, 2]

    taken = np.zeros(len(points), dtype=bool)
    groups: list[np.ndarray] = []

    # Highest contacts first: they have the most room beneath them for arms.
    for i in np.argsort(-heights):
        if taken[i]:
            continue
        members = [i]
        taken[i] = True
        for j in tree.query_ball_point(elbows[i, :2], radius):
            if taken[j]:
                continue
            # Sharing costs shaft height: the new arm reaches further, so it
            # has to leave the shaft lower down. Refuse once the shaft would
            # not even clear its own base — at that point it is a foot with an
            # arm out of it, there is nothing left for a cross-link to grip,
            # and each contact is better off on a shaft of its own.
            if _shaft_top_for(elbows[i, :2], elbows[[*members, j]], params) <= _min_top(params):
                continue
            members.append(j)
            taken[j] = True
        groups.append(np.array(members))
    return groups


def _min_top(params: SupportParams) -> float:
    """Shortest a shaft may be and still be worth calling a shaft.

    Its own base height: below that the whole shaft is buried inside the base
    disc. Cross-links start half a base height up (see :func:`_link_shafts`), so
    a shaft shorter than this cannot be braced at all.
    """
    return max(0.0, float(params.foot_height))


def _arm_rise(dist: float, params: SupportParams) -> float:
    """Vertical run an arm needs to cover `dist` horizontally and stay printable."""
    angle = min(float(params.arm_angle_deg), strut_lean(params))
    return dist / max(math.tan(math.radians(max(angle, 1.0))), 1e-6)


def _shaft_top_for(xy: np.ndarray, elbows: np.ndarray, params: SupportParams) -> float:
    """Highest the shaft top may sit and still feed every arm."""
    elbows = np.atleast_2d(np.asarray(elbows, dtype=np.float64))
    xy = np.asarray(xy, dtype=np.float64)
    d = np.linalg.norm(elbows[:, :2] - xy, axis=1)
    rises = np.array([_arm_rise(float(v), params) for v in d])
    return float((elbows[:, 2] - rises).min())


def _make_shaft(field: AvoidanceField, ray: DownRay, members, elbows, params: SupportParams):
    """Place one shaft under a group of contacts, or return None.

    The plate is tried first and properly: :func:`_route_to_plate` will lean the
    shaft around anything in the way rather than give up at the first
    obstruction. Only when there is genuinely no route down — and only when
    ``params.plate_only`` is off — does the shaft fall back to standing on the
    model.
    """
    elbows = np.atleast_2d(np.asarray(elbows, dtype=np.float64))
    start = elbows[:, :2].mean(axis=0)
    if _shaft_top_for(start, elbows, params) <= 0.0:
        # A contact barely clear of the plate still gets a support: the base
        # cone simply runs straight into the tip with no shaft in between. Only
        # refuse when the arm would have to start below the plate.
        return None

    bucket = field.bucket(params.shaft_lower_diameter * 0.5)
    modes = (True,) if params.plate_only else (True, False)

    for to_plate in modes:
        xy = _settle(field, start, _shaft_top_for(start, elbows, params), bucket, to_plate)
        if xy is None:
            continue
        top_z = _shaft_top_for(xy, elbows, params)
        if top_z <= 0.0:
            continue

        if to_plate:
            path = _route_to_plate(field, xy, top_z, bucket)
            if path is None:
                continue
            on_model = False
        else:
            land_z, on_model, top_z = _drop_shaft(field, ray, xy, top_z, bucket, params)
            if top_z - land_z <= 0.0:
                continue
            path = np.array([[xy[0], xy[1], land_z], [xy[0], xy[1], top_z]])

        shaft = Shaft(path=path, on_model=on_model)
        arms = _fit_arms(field, ray, shaft, members, elbows, params)
        if arms is None:
            continue
        shaft.arms = arms
        return shaft
    return None


def _fit_arms(
    field: AvoidanceField,
    ray: DownRay,
    shaft: Shaft,
    members,
    elbows: np.ndarray,
    params: SupportParams,
):
    """Hang every contact in the group off this shaft, or give up on the group.

    All-or-nothing on purpose: the caller retries the members one at a time, and
    a contact on a shaft of its own is a better support than a contact on an arm
    that has to cross the sculpt to reach it.
    """
    xy, top_z, land_z = shaft.xy, shaft.top_z, shaft.land_z
    r_arm = params.arm_diameter * 0.5
    arms: list[Arm] = []
    for pt, elbow in zip(members, elbows):
        elbow = np.asarray(elbow, dtype=np.float64)
        d = float(np.linalg.norm(elbow[:2] - xy))
        attach_z = min(top_z, elbow[2] - _arm_rise(d, params))
        attach = np.array([xy[0], xy[1], max(attach_z, land_z)])
        if not _strut_clear(field, ray, attach, elbow, r_arm, params):
            return None
        arms.append(
            Arm(
                contact=np.asarray(pt.position, dtype=np.float64),
                normal=np.asarray(pt.normal, dtype=np.float64),
                attach=attach,
                elbow=elbow,
            )
        )
    return arms


def _tip_axis(normal: np.ndarray, params: SupportParams) -> np.ndarray:
    """Vector from the base of the tip up to the contact.

    A resin tip leaves the surface along its normal, meeting it at roughly a
    right angle — that is what makes the contact a dot rather than a smear. But
    a tip is a small strut like any other, so it may not lean further off
    vertical than the printable limit; on a near-vertical wall it gets pitched
    up until it can print.
    """
    n = np.asarray(normal, dtype=np.float64)
    if np.linalg.norm(n) < 1e-9:
        n = np.array([0.0, 0.0, -1.0])
    n = n / np.linalg.norm(n)

    axis = -n * params.tip_length  # points from the tip's base up to the contact
    if axis[2] <= 0.0:
        # An upward-facing contact cannot be reached from below along its
        # normal; come straight up instead.
        return np.array([0.0, 0.0, params.tip_length])

    max_lean = math.tan(math.radians(strut_lean(params)))
    flat = float(np.linalg.norm(axis[:2]))
    if flat > axis[2] * max_lean:
        # Too shallow to print: keep the direction, steepen the climb.
        scale = (axis[2] * max_lean) / flat
        axis = np.array([axis[0] * scale, axis[1] * scale, axis[2]])
    return axis


def _settle(field: AvoidanceField, xy, z: float, bucket: int, to_plate: bool = True):
    """Nudge a shaft top onto ground it may stand on at this height.

    With `to_plate` the ground is the *reachable* set, so settling here is what
    establishes :func:`_route_to_plate`'s precondition; without it, merely the
    collision-free set, which is all a shaft that will end up on the model
    needs.
    """
    layer = field.layer_of(z)
    region = field.standable(bucket, layer, to_plate)
    if region is None or region.is_empty:
        return None
    if field.contains(region, xy):
        return np.asarray(xy, dtype=np.float64)
    q = nearest_points(region, shapely.Point(float(xy[0]), float(xy[1])))[0]
    moved = np.array([q.x, q.y])
    if float(np.linalg.norm(moved - xy)) <= field.max_move * 4.0:
        return moved
    return None


def _route_to_plate(field: AvoidanceField, xy, top_z: float, bucket: int):
    """Walk a shaft down to the plate, stepping around the model on the way.

    A shaft that finds clear air below it drops dead straight, which is what an
    SLA shaft does and what nearly all of them do here. This is the other case:
    something is in the way, and rather than stopping on it, the shaft leans
    around it.

    The route is read straight out of the reachability sweep. ``reach[i]`` is
    every position on layer ``i`` from which the plate is still gettable without
    entering the model, and it is built as
    ``free[i] ∩ reach[i-1].buffer(max_move)`` — so a position inside ``reach[i]``
    is guaranteed to have a position inside ``reach[i-1]`` within one layer's
    sideways travel of it. Descending is then just: stay put if the layer below
    still allows it, otherwise slide to the nearest place that does.

    That guarantee is why this cannot paint itself into a corner and why it
    needs no search, no backtracking and no heuristic. The precondition is the
    whole of it: the starting position must already be inside ``reach`` at the
    top, which is what :func:`_settle` establishes.

    Returns the axis bottom-to-top as an (N, 3) array, or None if the start was
    not reachable after all.
    """
    top_layer = field.layer_of(top_z)
    # `layer_of` rounds, so it can land a hair above the top. Never start the
    # descent above the shaft's own top: the first row is the top itself.
    while top_layer > 0 and field.heights[top_layer] > top_z:
        top_layer -= 1

    cur = np.asarray(xy, dtype=np.float64)
    if not field.contains(field.reach(bucket, top_layer), cur):
        return None

    # Simplifying the buffered region can nudge its boundary out by up to the
    # simplify tolerance, so the nearest reachable point can sit a whisker
    # beyond one layer's travel. Allow for that and no more.
    slack = field.max_move + max(field.params.nozzle_diameter * 0.25, 1e-3)

    rows = [(cur[0], cur[1], float(top_z))]
    for layer in range(top_layer - 1, -1, -1):
        region = field.reach(bucket, layer)
        if region is None or region.is_empty:
            return None
        if not field.contains(region, cur):
            q = nearest_points(region, shapely.Point(float(cur[0]), float(cur[1])))[0]
            moved = np.array([q.x, q.y])
            if float(np.linalg.norm(moved - cur)) > slack:
                return None
            cur = moved
        rows.append((cur[0], cur[1], float(field.heights[layer])))

    rows.reverse()
    return _simplify_path(np.asarray(rows, dtype=np.float64))


def _simplify_path(path: np.ndarray) -> np.ndarray:
    """Drop rows that only repeat the previous XY.

    The sweep produces one row per collision layer, and on a shaft that never
    has to dodge anything that is dozens of identical rows — dozens of rings to
    loft for a strut that is a straight cylinder. Collinear runs collapse to
    their endpoints; a genuine step keeps both of its.
    """
    if len(path) < 3:
        return path
    keep = [0]
    for i in range(1, len(path) - 1):
        a, b, c = path[keep[-1]], path[i], path[i + 1]
        dz_ab, dz_bc = b[2] - a[2], c[2] - b[2]
        if dz_ab <= _EPS or dz_bc <= _EPS:
            keep.append(i)
            continue
        # Where would the straight line a->c be at b's height?
        t = (b[2] - a[2]) / (c[2] - a[2])
        want = a[:2] + (c[:2] - a[:2]) * t
        if float(np.linalg.norm(b[:2] - want)) > 1e-9:
            keep.append(i)
    keep.append(len(path) - 1)
    return path[keep]


def _drop_shaft(field: AvoidanceField, ray: DownRay, xy, top_z, bucket, params: SupportParams):
    """Take a vertical shaft straight down until something stops it.

    The fallback for when the plate is genuinely unreachable and
    ``params.plate_only`` is off: no routing, no leaning, just stand on whatever
    is directly underneath. Reaching the plate is :func:`_route_to_plate`'s job.
    """
    top_layer = field.layer_of(top_z)
    plate_eps = max(1e-6, params.layer_height * 0.5)

    blocked_at = None
    for layer in range(top_layer, -1, -1):
        if not field.contains(field.free(bucket, layer), xy):
            blocked_at = layer
            break

    if blocked_at is None:
        return 0.0, False, top_z

    # Something is in the way lower down: stand on whatever is directly below
    # the last clear height. Never above the shaft's own top — when the block
    # starts at the very top layer, the surface immediately beneath the top is
    # what the shaft stands on.
    clear_z = min(top_z, float(field.heights[min(blocked_at + 1, field.n_layers - 1)]))
    probe = np.array([[float(xy[0]), float(xy[1]), clear_z]])
    hit_z, _, hit = ray.z_below(probe)
    if bool(hit[0]) and plate_eps < float(hit_z[0]) < top_z:
        return float(hit_z[0]), True, top_z
    return max(0.0, min(clear_z, top_z) - params.layer_height), False, top_z


# --------------------------------------------------------------------------- #
# cross-links: the part that makes it a scaffold
# --------------------------------------------------------------------------- #


def _link_shafts(field: AvoidanceField, ray: DownRay, shafts: list[Shaft], params: SupportParams):
    """Tie neighbouring shafts together into a lattice.

    Resin slicers run these horizontally. An FDM printer cannot bridge a
    horizontal strut in mid-air, so each link is laid at a chosen diagonal
    inside the printable band instead — the same lattice, an angle the nozzle
    can manage.

    Which shaft braces which is decided in one go for the whole field rather
    than shaft by shaft (:func:`_neighbour_candidates`, :func:`_choose_links`),
    and the rungs are laid on heights the whole field shares
    (:func:`_link_storeys`). Both exist to make the scaffold read as a
    structure. Picking neighbours greedily per shaft, and giving each pair its
    own heights, braces about as well and looks like a scribble.
    """
    parts: list[trimesh.Trimesh] = []
    if not params.brace_enabled or len(shafts) < 2:
        return parts, 0

    angle = _link_angle(params)
    if angle is None:
        return parts, 0
    tan_a = math.tan(math.radians(angle))

    pos = np.array([s.xy for s in shafts])
    candidates = _neighbour_candidates(pos, params)
    if not candidates:
        return parts, 0

    chosen, spare = _choose_links(candidates, len(shafts))
    storeys = _link_storeys(shafts, chosen, tan_a, params)

    made: set[tuple[int, int]] = set()

    def build(i: int, j: int) -> bool:
        rungs = _rungs(field, ray, shafts, i, j, tan_a, storeys, params)
        if not rungs:
            return False
        parts.extend(rungs)
        made.add((min(i, j), max(i, j)))
        return True

    for i, j in chosen:
        build(i, j)

    # Two things leave a shaft with nothing once the graph has had its say: a
    # chosen pair that turns out unbuildable, because the model sits in the way
    # of every rung between them, and a shaft whose neighbours are all far
    # shorter than it is, so no link to one of them has the vertical run to be
    # printable at all. A braced shaft is worth more than a tidy one, so both
    # get a second try — in order of how far the tidy answer has to stretch,
    # and only ever for a shaft that has nothing.
    braced = {i for pair in made for i in pair}
    for i, j in spare:
        if i not in braced or j not in braced:
            if build(i, j):
                braced.update((i, j))
    for i, j in _reach_further(pos, shafts, braced, tan_a, params):
        if i not in braced or j not in braced:
            if build(i, j):
                braced.update((i, j))
    return parts, len(made)


def _reach_further(
    pos: np.ndarray,
    shafts: list[Shaft],
    braced: set[int],
    tan_a: float,
    params: SupportParams,
) -> list[tuple[int, int]]:
    """Pairs for a shaft its own neighbours could not brace, nearest first.

    A link needs ``span · tan(angle)`` of shared height, and the shaft it is
    tied to only offers as much as it has. So a tall shaft in a thicket of
    stubs can be adjacent to five others and still have nowhere to put a strut,
    and it is the tall ones that most need one. Here, and only here, a shaft is
    allowed past its neighbours to whatever within ``brace_max_span`` stands
    tall enough to hold it.
    """
    loose = [i for i in range(len(shafts)) if i not in braced]
    if not loose:
        return []
    kdt = cKDTree(pos)
    out: list[tuple[int, int]] = []
    for i in loose:
        near = sorted(
            (j for j in kdt.query_ball_point(pos[i], params.brace_max_span) if j != i),
            key=lambda j: (float(np.linalg.norm(pos[j] - pos[i])), j),
        )
        out.extend(
            (i, j)
            for j in near
            if _link_band(shafts[i], shafts[j], tan_a, params) is not None
        )
    return out


def _neighbour_candidates(
    pos: np.ndarray, params: SupportParams
) -> list[tuple[float, int, int]]:
    """Every pair of shafts worth bracing, as ``(span, i, j)``, shortest first.

    "Which shafts are neighbours" is a question with a real answer, and it is
    not "the three nearest ones within `brace_max_span`". Nearest-first picks
    the same popular shaft from every side of a crowd, leaves the shaft on the
    far edge of it with nothing, and links two shafts straight over the top of
    a third standing between them.

    The Delaunay triangulation answers it properly: it is the graph of shafts
    that are *actually adjacent* in the field, it is planar — so no two links
    cross in plan — and it does not depend on the order the shafts arrive in.
    Its long thin border triangles are then dropped by the Gabriel test in
    :func:`_spans_a_nearer_shaft`, which is what removes a link reaching over
    an intervening shaft. What is left, on an evenly spaced field, is the grid
    you would have drawn by hand.
    """
    n = len(pos)
    pairs: set[tuple[int, int]] = set()
    if n >= 4:
        try:
            simplices = Delaunay(pos).simplices
        except QhullError:
            simplices = ()  # every shaft on one line: no triangulation exists
        for tri in simplices:
            for u, v in ((0, 1), (1, 2), (2, 0)):
                i, j = int(tri[u]), int(tri[v])
                pairs.add((min(i, j), max(i, j)))
    if not pairs:
        # Too few shafts, or all of them collinear. Offer every pair and let
        # the span and Gabriel filters below do the thinning.
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

    kdt = cKDTree(pos)
    # Overlapping shafts are already one column; nothing to brace.
    min_span = (params.shaft_lower_diameter + params.brace_diameter) * 0.5
    out: list[tuple[float, int, int]] = []
    for i, j in pairs:
        span = float(np.linalg.norm(pos[j] - pos[i]))
        if not min_span < span <= params.brace_max_span:
            continue
        if _spans_a_nearer_shaft(kdt, i, j, span):
            continue
        out.append((span, i, j))
    # Shortest first: a short link needs less vertical run to stay printable,
    # so close pairs are both the cheapest and the most useful. The index
    # tie-break keeps the choice identical from run to run.
    out.sort()
    return out


def _spans_a_nearer_shaft(kdt: cKDTree, i: int, j: int, span: float) -> bool:
    """Is a third shaft standing between these two? (The Gabriel condition.)

    Take the circle that has the link as its diameter. Any shaft inside it is
    closer to both ends than the ends are to each other, so the link is
    reaching over its head — and two short links through that shaft brace the
    same pair better and look like they belong there.
    """
    mid = (kdt.data[i] + kdt.data[j]) * 0.5
    inside = kdt.query_ball_point(mid, span * 0.5 * (1.0 - 1e-6))
    return any(k != i and k != j for k in inside)


def _choose_links(candidates: list[tuple[float, int, int]], n_shafts: int):
    """Split the candidate pairs into the ones to build and the runners-up.

    The cap is on how many *neighbours* a shaft braces against, not on how many
    struts hang off it: a tall pair with a four-rung ladder between them is
    still one neighbour. And it is spent globally, shortest link first, which
    is what makes the result even — per-shaft greed hands the first shaft in
    the list its three best neighbours and leaves the last one isolated.

    A cap can still cut a corner of the field adrift, and an unbraced island is
    the one thing worse than an untidy link, so a second pass puts back the
    shortest link across each remaining split.
    """
    parent = list(range(n_shafts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    chosen: list[tuple[int, int]] = []
    passed_over: list[tuple[int, int]] = []
    degree = [0] * n_shafts
    for _, i, j in candidates:
        if degree[i] < _LINKS_PER_SHAFT and degree[j] < _LINKS_PER_SHAFT:
            chosen.append((i, j))
            degree[i] += 1
            degree[j] += 1
            union(i, j)
        else:
            passed_over.append((i, j))

    spare: list[tuple[int, int]] = []
    for i, j in passed_over:  # still shortest first
        if union(i, j):
            chosen.append((i, j))
        else:
            spare.append((i, j))
    return chosen, spare


def _link_band(a: Shaft, b: Shaft, tan_a: float, params: SupportParams):
    """The heights a link between these two may be *centred* on.

    Start half way up the bases rather than clear of them: on a short support
    that half a base height is the difference between a link and no link, and a
    link end buried in a base is buried in solid material, which costs nothing.
    """
    rise = float(np.linalg.norm(b.xy - a.xy)) * tan_a
    lo = max(a.land_z, b.land_z) + params.foot_height * 0.5
    hi = min(a.top_z, b.top_z)
    if hi - lo < rise:
        return None  # no vertical run to spend; this pair gets no link
    return lo + rise * 0.5, hi - rise * 0.5


def _link_storeys(
    shafts: list[Shaft], edges, tan_a: float, params: SupportParams
) -> np.ndarray:
    """The heights the whole field lays its links at.

    Every pair used to start its own ladder from its own base, so on the sample
    mini the links landed at twenty distinct heights: correct, and visual
    noise. One shared set of heights turns the same links into storeys —
    each a row of struts at one level, all leaning the same way, which is what
    scaffolding looks like and what makes it easy to read and to cut away.

    A grid is the obvious way to get shared heights and the wrong one. Each
    pair can only hold a link inside its own window of heights (:func:`_link_band`)
    — generous on tall shafts, barely there on a short pair over a low overhang
    — and a grid laid at ``brace_interval`` walks straight past the narrow ones,
    which then have to fall off it again to get braced at all.

    So take the heights from the windows instead. The tidiest scaffold is the
    one covering every window with the *fewest distinct heights*, and that is a
    stabbing problem with an exact greedy answer: sort the windows by their
    top, and each time one is still uncovered put a storey at its top, which is
    also the height reaching furthest into the windows still to come. Each
    storey then slides to the middle of the windows it ended up covering — the
    same count of them, but a link sits in its window rather than clinging to
    the ceiling of it.
    """
    bands = [
        band
        for band in (_link_band(shafts[i], shafts[j], tan_a, params) for i, j in edges)
        if band is not None
    ]
    if not bands:
        return np.empty(0)

    levels: list[float] = []
    for lo, hi in sorted(bands, key=lambda b: (b[1], b[0])):
        # Windows arrive in order of their ceiling, so only the newest storey
        # can possibly be inside this one.
        if not levels or levels[-1] < lo:
            levels.append(hi)

    settled = []
    for z in levels:
        covered = [b for b in bands if b[0] <= z <= b[1]]
        middle = float(np.mean([(lo + hi) * 0.5 for lo, hi in covered]))
        floor = max(lo for lo, _ in covered)
        ceiling = min(hi for _, hi in covered)
        settled.append(min(max(middle, floor), ceiling))
    return np.array(sorted(settled))


def _rungs(
    field: AvoidanceField,
    ray: DownRay,
    shafts: list[Shaft],
    i: int,
    j: int,
    tan_a: float,
    storeys: np.ndarray,
    params: SupportParams,
) -> list[trimesh.Trimesh]:
    """Every strut between one pair of shafts — one per storey they both reach."""
    a, b = _uphill(shafts[i], shafts[j])
    band = _link_band(a, b, tan_a, params)
    if band is None:
        return []
    lo, hi = band
    rise = float(np.linalg.norm(b.xy - a.xy)) * tan_a

    levels = _rung_heights(storeys, lo, hi, params)

    out: list[trimesh.Trimesh] = []
    for centre in levels:
        # `_link_ends` solves the top end for itself, so a rung sits centred on
        # its storey only as closely as a routed shaft's lean allows.
        ends = _link_ends(a, b, float(centre) - rise * 0.5, tan_a, b.top_z)
        if ends is None:
            continue
        p0, p1 = ends
        if not _strut_clear(field, ray, p0, p1, params.brace_diameter * 0.5, params):
            continue
        # Both ends sit on the shaft axes, so they are buried inside the
        # shafts. Capping them would put a flat downward face — a 90 degree
        # overhang — inside the solid.
        out.append(
            make_strut(
                p0,
                p1,
                params,
                radius=params.brace_diameter * 0.5,
                cap_bottom=False,
                cap_top=False,
            )
        )
    return out


def _rung_heights(storeys: np.ndarray, lo: float, hi: float, params: SupportParams) -> np.ndarray:
    """Which storeys one pair of shafts hangs a rung on.

    Every storey inside the pair's window, thinned so a tall pair gets a ladder
    rather than a wall of struts. A pair that reaches no storey at all is one
    the storeys were not computed from — a runner-up brought in to rescue an
    unbraced shaft — and it gets a single link at the nearest height it can
    hold, which is still better than standing loose.
    """
    inside = storeys[(storeys >= lo) & (storeys <= hi)]
    if not len(inside):
        if not len(storeys):
            return np.empty(0)
        nearest = float(min(storeys, key=lambda z: abs(z - (lo + hi) * 0.5)))
        return np.array([min(max(nearest, lo), hi)])

    spacing = max(params.brace_interval, params.shaft_lower_diameter * 4.0)
    kept: list[float] = []
    for z in inside:
        if not kept or z - kept[-1] >= spacing:
            kept.append(float(z))
        if len(kept) >= _MAX_RUNGS:
            break
    return np.array(kept)


def _uphill(a: Shaft, b: Shaft) -> tuple[Shaft, Shaft]:
    """Which end a link leaves from, and which it climbs to.

    A diagonal has to lean one way or the other and, for a single link, there
    is nothing to choose between them. Across a storey there is: decide by
    position rather than by whatever order the shafts came in, and the whole
    row leans together instead of each strut picking its own way up.
    """
    ka = (round(float(a.xy[0]), 6), round(float(a.xy[1]), 6))
    kb = (round(float(b.xy[0]), 6), round(float(b.xy[1]), 6))
    return (a, b) if ka <= kb else (b, a)


def _link_ends(a: Shaft, b: Shaft, z: float, tan_a: float, hi: float):
    """Both ends of one cross-link, at the angle the printable band allows.

    A straight shaft is in the same place at every height, so its end of a link
    is wherever it stands. A *routed* shaft is not: its axis has moved by the
    time the link meets it, so the horizontal span — and therefore the vertical
    run that keeps the link at ``angle`` — depends on the very heights being
    solved for. Both shafts lean by at most one ``max_move`` per layer, so the
    dependence is weak and a couple of passes settle it.
    """
    p0 = np.append(a.xy_at(z), z)
    z1 = z
    for _ in range(3):
        span = float(np.linalg.norm(b.xy_at(z1) - p0[:2]))
        z1 = z + span * tan_a
        if z1 > hi:
            return None
    p1 = np.append(b.xy_at(z1), z1)
    if z1 - z <= _EPS:
        return None
    return p0, p1


def _link_angle(params: SupportParams) -> float | None:
    """The angle a cross-link is laid at, above horizontal.

    A tilted strut overhangs by ``90 - angle`` along its sides and ``angle`` at
    its ends, so it only prints inside the band between the two. Within that
    band, take the *shallowest* angle available: every degree shallower is more
    horizontal span for the same vertical run, and vertical run is the scarce
    thing — supports on a low overhang are short, and a link that needs more
    rise than the shafts have simply cannot be placed.
    """
    limit = float(params.printable_overhang_deg)
    lo, hi = 90.0 - limit, limit
    if lo > hi:
        return None
    return float(min(lo + 2.0, hi))  # a couple of degrees of margin


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def _foot_radius(
    shaft: Shaft, foot_h: float, field: AvoidanceField, params: SupportParams
) -> float:
    """The base disc, shrunk to whatever fits beside the model.

    A base is much fatter than the shaft it sits under — 5 mm against 1.2 mm —
    and until shafts routed around obstacles it was never in a position where
    that mattered: a blocked shaft stood on the model, so a shaft that reached
    the plate had reached it down a clear column. A routed shaft comes down
    *beside* whatever it stepped around, hard against the clearance boundary,
    and a full-width disc there reaches straight back into the sculpt.

    Overlap at the plate itself is fine and deliberate — the model's own
    footprint and the feet are all printed flat on glass, and neighbouring feet
    are supposed to merge into a raft. Overlap with a wall standing 10 mm tall
    is not: that fuses the support to the sculpt. So the disc keeps its full
    width wherever there is room, and gives up only as much as it must.
    """
    want = params.foot_diameter * 0.5
    floor = params.shaft_lower_diameter * 0.5
    if foot_h <= _EPS or want <= floor:
        return want

    # Every layer the disc actually occupies, not just the one it stands on.
    top = field.layer_of(shaft.land_z + foot_h)
    room = min(
        field.room(shaft.xy_at(field.heights[i]), i)
        for i in range(field.layer_of(shaft.land_z), max(top, 0) + 1)
    )
    return float(np.clip(room, floor, want))


def _rings_at_every_bend(profile, axis: PathXY):
    """Add a ring at each bend in the axis, keeping the profile's own radii.

    A loft only knows where its axis is at the heights it is given a ring for.
    Give it two — bottom and top — and it draws a straight cylinder between
    them, which on a shaft that bends around the model is a cylinder straight
    *through* the model. So every vertex of the route gets a ring, at whatever
    radius the profile has at that height.
    """
    prof = sorted((float(z), float(r)) for z, r in profile)
    zs = [z for z, _ in prof]
    rs = [r for _, r in prof]
    lo, hi = zs[0], zs[-1]

    extra = [float(z) for z in axis.heights() if lo + _EPS < z < hi - _EPS]
    if not extra:
        return prof

    merged = sorted(set(zs) | set(extra))
    # np.interp needs strictly increasing x; the profile can repeat a height
    # (the foot's disc-to-flare corner), and there the *lower* radius is the
    # one a bend between them should take, so the corner is nudged apart.
    xs = np.asarray(zs, dtype=np.float64)
    xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-12)
    return [(z, float(np.interp(z, xs, rs))) for z in merged]


def _shaft_meshes(
    shaft: Shaft, field: AvoidanceField, params: SupportParams
) -> list[trimesh.Trimesh]:
    """Base, join cone, shaft, arms and tips for one support."""
    out: list[trimesh.Trimesh] = []
    r_low = params.shaft_lower_diameter * 0.5
    r_up = params.shaft_upper_diameter * 0.5

    # Base, flare and shaft as one continuous stack of rings rather than three
    # stacked primitives. Stacking leaves the base's top cap and the shaft's
    # bottom cap face to face, and a flat downward face is a 90 degree overhang
    # however deeply it is buried.
    if shaft.on_model:
        # Landing on the model: end in a tip there too, so the support snaps
        # off at the bottom instead of fusing to the sculpt.
        z0 = shaft.land_z - params.tip_penetration
        tip_len = min(params.tip_length, max(shaft.height * 0.4, params.layer_height))
        profile = [(z0, params.tip_diameter * 0.5), (z0 + tip_len, r_low)]
    else:
        z0 = shaft.land_z
        # A support shorter than the base is not a reason to skip the base —
        # that support needs the adhesion most — but the base may not grow out
        # of the top of the shaft it is supposed to sit under, so it is squashed
        # into whatever height there is.
        foot_h = min(params.foot_height, max(0.0, shaft.top_z - z0 - params.layer_height))
        profile = list(foot_profile(z0, foot_h, _foot_radius(shaft, foot_h, field, params), r_low))

    if shaft.top_z > profile[-1][0] + _EPS:
        profile.append((float(shaft.top_z), r_up))

    axis = PathXY(np.vstack([[*shaft.base_xy, z0], shaft.path]))
    profile = _rings_at_every_bend(profile, axis)
    out.append(mesh_rings(rings_on_axis(axis, profile), params.ring_sections))

    for arm in shaft.arms:
        out.append(
            make_strut(
                arm.attach,
                arm.elbow,
                params,
                radius=params.arm_diameter * 0.5,
                cap_bottom=False,  # buried in the shaft
                cap_top=False,  # the tip continues from here
            )
        )

        contact = arm.contact
        elbow = arm.elbow
        tip_len = float(contact[2] - elbow[2])
        if tip_len > _EPS:
            axis = LineXY(elbow, contact + np.array([0.0, 0.0, params.tip_penetration]))
            profile = tip_profile(
                float(contact[2]), tip_len, params, params.arm_diameter * 0.5
            )
            out.append(
                mesh_rings(rings_on_axis(axis, profile), params.ring_sections, cap_bottom=False)
            )

    return [m for m in out if m is not None and len(m.faces)]
