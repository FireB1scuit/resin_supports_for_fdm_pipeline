"""Stage 3, resin style — the SLA scaffold, dimensioned for FDM.

This is the structure resin slicers build, and the one this project exists to
make printable on filament. It is *not* an FDM organic tree: nothing fuses into
a thickening trunk that wanders down to the plate. An SLA support is a set of
discrete components, and it keeps its shape all the way down::

    contact ── tip ── arm ── shaft ── join cone ── base
                (angled)  (vertical, thin)         (on the plate)

* **tip** — a small cone that meets the surface at roughly a right angle and
  penetrates it slightly, so it snaps off leaving a dot rather than a scar.
* **arm** — the short angled link from the shaft up to the tip. Several arms
  can fan off one shaft; that is how resin slicers keep the shaft count down
  (Lychee calls it parenting, or "optimize supports").
* **shaft** — straight and vertical, ``shaft_upper_diameter`` at the top and
  ``shaft_lower_diameter`` at the bottom. A slight taper, not a trunk.
* **join cone** — the 45 degree flare from the shaft out to the base.
* **base** — the footprint on the plate.
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

Collision and reachability come from :class:`rsupport.tree.AvoidanceField`,
which is independent of what gets built on top of it: a shaft may never pass
through the model, and only rests on the model when the plate is genuinely
unreachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree
from shapely.ops import nearest_points

from .mesh_io import concat
from .raycast import DownRay
from .tree import AvoidanceField, _branch_angle
from .types import SupportBuild, SupportParams, SupportPoint

__all__ = ["build_resin", "Shaft", "Arm"]

_EPS = 1e-9

#: How many neighbours one shaft will brace against. More than this and the
#: scaffold becomes a solid wall that is miserable to cut off.
_LINKS_PER_SHAFT = 3


@dataclass
class Arm:
    """The angled link from a shaft up to one contact point."""

    contact: np.ndarray
    normal: np.ndarray
    attach: np.ndarray  # where it leaves the shaft
    elbow: np.ndarray  # where the arm ends and the tip begins


@dataclass
class Shaft:
    """One vertical strut, its arms, and where it stands."""

    xy: np.ndarray
    top_z: float
    land_z: float
    on_model: bool
    arms: list[Arm] = dc_field(default_factory=list)

    @property
    def height(self) -> float:
        return max(0.0, self.top_z - self.land_z)


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
        parts.extend(_shaft_meshes(shaft, params))

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


def _elbow(pt: SupportPoint, params: SupportParams) -> np.ndarray:
    """The joint between tip and arm, where the tip's own run ends.

    A resin tip leaves the surface roughly perpendicular, so on a downward
    overhang the elbow sits directly below the contact and on a wall it stands
    off to the side — which is exactly the SLA look.

    Everything below is measured from here, never from the contact. The tip has
    already covered that horizontal step, and charging the arm a vertical rise
    for the same distance is what made contacts near the plate impossible to
    support: their shaft top came out below the build plate.
    """
    contact = np.asarray(pt.position, dtype=np.float64)
    return contact - _tip_axis(np.asarray(pt.normal, dtype=np.float64), params)


def _plan_shafts(field: AvoidanceField, ray: DownRay, points, params: SupportParams):
    """Group contacts onto shared shafts and work out where each shaft stands."""
    warnings: list[str] = []
    dropped: list[SupportPoint] = []

    elbows = np.array([_elbow(p, params) for p in points])
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

    shafts = _absorb_neighbours(shafts, params)

    if dropped:
        warnings.append(f"{len(dropped)} contact point(s) had nowhere to stand a shaft")
    return shafts, dropped, warnings


def _absorb_neighbours(shafts: list[Shaft], params: SupportParams) -> list[Shaft]:
    """Fold shafts that are practically touching into one.

    Grouping happens per contact, so two contacts that could not share a shaft
    on the way down can still end up standing a shaft each half a millimetre
    apart. Two shafts closer together than their own diameter are one shaft with
    extra steps: they waste resin, they double the number of feet to snap off,
    and — because a cross-link needs real distance to span — they leave nothing
    for the scaffold to brace against.

    An arm is only handed over if it can still meet its contact at a printable
    angle from the surviving shaft.
    """
    if len(shafts) < 2:
        return shafts

    reach = params.shaft_lower_diameter * 1.5
    pos = np.array([s.xy for s in shafts])
    kdt = cKDTree(pos)

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
            moved = []
            for arm in other.arms:
                d = float(np.linalg.norm(arm.elbow[:2] - host.xy))
                attach_z = arm.elbow[2] - _arm_rise(d, params)
                if attach_z < host.land_z + params.layer_height:
                    break  # this arm cannot reach from over here
                moved.append(
                    Arm(
                        contact=arm.contact,
                        normal=arm.normal,
                        attach=np.array([host.xy[0], host.xy[1], min(attach_z, host.top_z)]),
                        elbow=arm.elbow,
                    )
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
            # The shaft top must clear every arm, and cannot go below the plate.
            if _shaft_top_for(elbows[i, :2], elbows[[*members, j]], params) <= 0.0:
                continue
            members.append(j)
            taken[j] = True
        groups.append(np.array(members))
    return groups


def _arm_rise(dist: float, params: SupportParams) -> float:
    """Vertical run an arm needs to cover `dist` horizontally and stay printable."""
    angle = min(float(params.arm_angle_deg), _branch_angle(params))
    return dist / max(math.tan(math.radians(max(angle, 1.0))), 1e-6)


def _shaft_top_for(xy: np.ndarray, elbows: np.ndarray, params: SupportParams) -> float:
    """Highest the shaft top may sit and still feed every arm."""
    elbows = np.atleast_2d(np.asarray(elbows, dtype=np.float64))
    xy = np.asarray(xy, dtype=np.float64)
    d = np.linalg.norm(elbows[:, :2] - xy, axis=1)
    rises = np.array([_arm_rise(float(v), params) for v in d])
    return float((elbows[:, 2] - rises).min())


def _make_shaft(field: AvoidanceField, ray: DownRay, members, elbows, params: SupportParams):
    """Place one shaft under a group of contacts, or return None."""
    elbows = np.atleast_2d(np.asarray(elbows, dtype=np.float64))
    xy = elbows[:, :2].mean(axis=0)
    top_z = _shaft_top_for(xy, elbows, params)
    # A contact barely clear of the plate still gets a support: the base cone
    # simply runs straight into the tip with no shaft in between. Only refuse
    # when the arm would have to start below the plate.
    if top_z <= 0.0:
        return None

    bucket = field.bucket(params.shaft_lower_diameter * 0.5)
    xy = _settle(field, xy, top_z, bucket)
    if xy is None:
        return None

    top_z = _shaft_top_for(xy, elbows, params)
    if top_z <= 0.0:
        return None

    land_z, on_model, top_z = _drop_shaft(field, ray, xy, top_z, bucket, params)
    if top_z - land_z <= 0.0:
        return None

    shaft = Shaft(xy=xy, top_z=top_z, land_z=land_z, on_model=on_model)
    for pt, elbow in zip(members, elbows):
        elbow = np.asarray(elbow, dtype=np.float64)
        d = float(np.linalg.norm(elbow[:2] - xy))
        attach_z = min(top_z, elbow[2] - _arm_rise(d, params))
        shaft.arms.append(
            Arm(
                contact=np.asarray(pt.position, dtype=np.float64),
                normal=np.asarray(pt.normal, dtype=np.float64),
                attach=np.array([xy[0], xy[1], max(attach_z, land_z)]),
                elbow=elbow,
            )
        )
    return shaft


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

    max_lean = math.tan(math.radians(_branch_angle(params)))
    flat = float(np.linalg.norm(axis[:2]))
    if flat > axis[2] * max_lean:
        # Too shallow to print: keep the direction, steepen the climb.
        scale = (axis[2] * max_lean) / flat
        axis = np.array([axis[0] * scale, axis[1] * scale, axis[2]])
    return axis


def _settle(field: AvoidanceField, xy, z: float, bucket: int):
    """Nudge a shaft position onto standable ground at this height."""
    layer = field.layer_of(z)
    for to_plate in (True, False):
        region = field.standable(bucket, layer, to_plate)
        if region is None or region.is_empty:
            continue
        if field.contains(region, xy):
            return np.asarray(xy, dtype=np.float64)
        q = nearest_points(region, shapely.Point(float(xy[0]), float(xy[1])))[0]
        moved = np.array([q.x, q.y])
        if float(np.linalg.norm(moved - xy)) <= field.max_move * 4.0:
            return moved
    return None


def _drop_shaft(field: AvoidanceField, ray: DownRay, xy, top_z, bucket, params: SupportParams):
    """Take a vertical shaft down as far as it can go.

    Straight down, because that is what an SLA shaft does. The plate is used
    whenever it is reachable; the shaft only stops on the model when the free
    space runs out beneath it.
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
    """
    parts: list[trimesh.Trimesh] = []
    if not params.brace_enabled or len(shafts) < 2:
        return parts, 0

    angle = _link_angle(params)
    if angle is None:
        return parts, 0
    tan_a = math.tan(math.radians(angle))

    pos = np.array([s.xy for s in shafts])
    kdt = cKDTree(pos)
    made: set[tuple[int, int]] = set()
    interval = max(params.brace_interval, params.shaft_lower_diameter * 4.0)

    for i, a in enumerate(shafts):
        # Nearest first: a short link needs less vertical run to stay printable,
        # so the close pairs are both the cheapest and the most useful.
        neighbours = sorted(
            (j for j in kdt.query_ball_point(pos[i], params.brace_max_span) if j != i),
            key=lambda j: float(np.linalg.norm(pos[j] - pos[i])),
        )
        linked_here = 0
        for j in neighbours:
            if linked_here >= _LINKS_PER_SHAFT:
                break
            b = shafts[j]
            key = (min(i, j), max(i, j))
            if key in made:
                continue
            span = float(np.linalg.norm(pos[j] - pos[i]))
            # Overlapping shafts are already one column; nothing to brace.
            if span < (params.shaft_lower_diameter + params.brace_diameter) * 0.5:
                continue

            rise = span * tan_a
            # Start just above the join cones rather than clear of them: on a
            # short support that half millimetre is the difference between a
            # link and no link.
            lo = max(a.land_z, b.land_z) + params.foot_height * 0.5
            hi = min(a.top_z, b.top_z)
            if hi - lo < rise:
                continue

            # One link per interval of height, so tall shafts get a ladder.
            n = 0
            z = lo + rise * 0.5
            while z + rise <= hi and n < 6:
                p0 = np.array([a.xy[0], a.xy[1], z])
                p1 = np.array([b.xy[0], b.xy[1], z + rise])
                if not ray.segment_blocked([p0], [p1])[0]:
                    linked_here += 1
                    # Both ends sit on the shaft axes, so they are buried
                    # inside the shafts. Capping them would put a flat
                    # downward face — a 90 degree overhang — inside the solid.
                    parts.append(
                        _strut(
                            p0,
                            p1,
                            params.brace_diameter * 0.5,
                            params,
                            cap_bottom=False,
                            cap_top=False,
                        )
                    )
                    made.add(key)
                    n += 1
                z += interval
    return parts, len(made)


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


def _strut(a, b, radius: float, params: SupportParams, cap_bottom=True, cap_top=True):
    """A straight strut with horizontal cross-sections.

    Sheared rather than rotated, so the end caps stay flat and horizontal — a
    rotated cap would be an overhang of ``90 - tilt``.
    """
    from .supports import _LineXY, _mesh_rings, _rings_on_axis

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo, hi = (a, b) if a[2] <= b[2] else (b, a)
    if hi[2] - lo[2] <= _EPS:
        return None
    axis = _LineXY(lo, hi)
    profile = [(float(lo[2]), radius), (float(hi[2]), radius)]
    return _mesh_rings(
        _rings_on_axis(axis, profile), params.pillar_sections, cap_bottom, cap_top
    )


def _shaft_meshes(shaft: Shaft, params: SupportParams) -> list[trimesh.Trimesh]:
    """Base, join cone, shaft, arms and tips for one support."""
    from .supports import _LineXY, _foot_profile, _mesh_rings, _rings_on_axis, _tip_profile

    out: list[trimesh.Trimesh] = []
    r_low = params.shaft_lower_diameter * 0.5
    r_up = params.shaft_upper_diameter * 0.5
    x, y = float(shaft.xy[0]), float(shaft.xy[1])

    # Base, join cone and shaft as one continuous stack of rings rather than
    # three stacked primitives. Stacking leaves the base's top cap and the
    # shaft's bottom cap face to face, and a flat downward face is a 90 degree
    # overhang however deeply it is buried.
    if shaft.on_model:
        # Landing on the model: end in a tip there too, so the support snaps
        # off at the bottom instead of fusing to the sculpt.
        z0 = shaft.land_z - params.tip_penetration
        tip_len = min(params.tip_length, max(shaft.height * 0.4, params.layer_height))
        profile = [(z0, params.tip_diameter * 0.5), (z0 + tip_len, r_low)]
    else:
        z0 = 0.0
        profile = list(_foot_profile(0.0, params.foot_height, params.foot_diameter * 0.5, r_low))

    if shaft.top_z > profile[-1][0] + _EPS:
        profile.append((float(shaft.top_z), r_up))

    axis = _LineXY([x, y, z0], [x, y, float(shaft.top_z)])
    out.append(_mesh_rings(_rings_on_axis(axis, profile), params.pillar_sections))

    for arm in shaft.arms:
        strut = _strut(
            arm.attach,
            arm.elbow,
            params.arm_diameter * 0.5,
            params,
            cap_bottom=False,  # buried in the shaft
            cap_top=False,  # the tip continues from here
        )
        if strut is not None:
            out.append(strut)

        contact = arm.contact
        elbow = arm.elbow
        tip_len = float(contact[2] - elbow[2])
        if tip_len > _EPS:
            axis = _LineXY(elbow, contact + np.array([0.0, 0.0, params.tip_penetration]))
            profile = _tip_profile(
                float(contact[2]), tip_len, params, params.arm_diameter * 0.5
            )
            out.append(
                _mesh_rings(_rings_on_axis(axis, profile), params.pillar_sections, cap_bottom=False)
            )

    return [m for m in out if m is not None and len(m.faces)]
