"""Stage 3, tree style — branches that merge on the way down.

The pillar generator gives every contact point its own stick, and then bolts
cross-braces on afterwards to stop the slender ones buckling. A tree is the
better structure: branches start thin at the model, merge into limbs and then
trunks as they descend, and flow *around* obstacles instead of being stopped by
them. Load travels through the junctions, so the structure holds the model
rather than balancing it.

The algorithm follows CuraEngine's tree support (Ghostkeeper's original
CuraEngine PR #655, later rewritten around influence areas and avoidance
sampling), itself descended from Vanek et al., *Clever Support* (2014):

* work top down, one layer at a time;
* a node may move at most ``pitch * tan(branch_angle)`` per layer, which is what
  bounds the branch's lean and therefore keeps it printable;
* nodes that come within reach of each other merge;
* collision is the model's cross-section offset by the branch radius plus a
  clearance, sampled at a handful of radii.

Reachability is the interesting part
------------------------------------
Two of this module's requirements — never pass through the model, and always
prefer the build plate over resting on it — look like separate problems and are
not. Whether the plate is reachable is a question about the whole column of
layers below a node, so it is precomputed bottom-up::

    solid[i]    = model cross-section over [z_i, z_i+1]
    free[r][i]  = everywhere a branch of radius r may stand on layer i
    reach[r][0] = free[r][0]
    reach[r][i] = free[r][i] ∩ reach[r][i-1].buffer(max_move)

``reach[r][i]`` is exactly the set of positions on layer ``i`` from which a
branch of radius ``r`` can still get to the plate without ever entering the
model. Both requirements then fall out of the descent for free:

* a node already inside ``reach`` on the layer below drops straight down;
* a node outside it moves to the nearest point of ``reach``, which is what makes
  a branch curve around an arm rather than stop at it;
* a node can only ever land on the model when ``reach`` is genuinely
  unreachable — not as a preference, but because there is nowhere else to go.

And since every node is always inside ``free`` for its own radius, no branch can
intersect the model at all. Both guarantees are structural rather than a
check-and-reject after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import nearest_points, unary_union

from .mesh_io import concat
from .overhang import slice_polygons
from .raycast import DownRay
from .types import SupportBuild, SupportParams, SupportPoint

__all__ = ["build_tree", "AvoidanceField", "Branch"]

_EPS = 1e-9

#: How many radii the collision field is sampled at. Cura samples too, for the
#: same reason: buffering every layer for every distinct branch radius would be
#: ruinous, and a handful of buckets is visually indistinguishable.
_RADIUS_BUCKETS = 6


# --------------------------------------------------------------------------- #
# the avoidance field
# --------------------------------------------------------------------------- #


class AvoidanceField:
    """Per-layer, per-radius maps of where a branch may stand.

    Built once per model and reused across every branch, which is what keeps
    the whole thing affordable — the expensive part is slicing, and
    :func:`rsupport.overhang.slice_polygons` batches every height into a single
    pass.
    """

    def __init__(self, mesh, params: SupportParams, top_z: float | None = None):
        self.params = params
        self.pitch = max(float(params.tree_layer_pitch), 1e-3)
        self.max_move = self.pitch * math.tan(math.radians(_branch_angle(params)))

        lo, hi = mesh.bounds
        ceiling = float(hi[2] if top_z is None else max(top_z, lo[2] + self.pitch))
        self.n_layers = max(2, int(math.ceil(ceiling / self.pitch)) + 2)
        self.heights = np.arange(self.n_layers, dtype=np.float64) * self.pitch

        # Branches fan outward as they fall, so the working area has to be
        # wider than the model by everything they could travel.
        span = float(ceiling) * math.tan(math.radians(_branch_angle(params)))
        margin = float(np.clip(span + 5.0, 10.0, 80.0))
        self.bed = box(lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin)

        self.radii = self._radius_buckets()
        self._build(mesh)

    # -- construction ------------------------------------------------------ #

    def _radius_buckets(self) -> np.ndarray:
        p = self.params
        lo = p.tip_diameter * 0.5
        hi = max(p.max_branch_diameter * 0.5, lo * 1.5)
        return np.geomspace(lo, hi, _RADIUS_BUCKETS)

    def _build(self, mesh) -> None:
        p = self.params
        # Slice just above each layer boundary: a slice exactly at z=0 catches
        # the model's base plane edge-on and comes back degenerate.
        sample = np.clip(self.heights + self.pitch * 0.02, 1e-4, None)
        sections = slice_polygons(mesh, sample)

        solids = []
        for i in range(self.n_layers):
            here = sections[i] if i < len(sections) else []
            above = sections[i + 1] if i + 1 < len(sections) else []
            merged = [g for g in (*here, *above) if g is not None and not g.is_empty]
            solids.append(unary_union(merged) if merged else None)

        # Simplifying keeps the repeated buffering in the sweep below from
        # compounding vertex counts without bound. Cura exposes the same knob
        # as "collision resolution".
        tol = max(p.nozzle_diameter * 0.25, 1e-3)

        self._free: list[list] = []
        self._reach: list[list] = []
        for r in self.radii:
            grow = float(r) + p.xy_clearance
            free = []
            for solid in solids:
                if solid is None or solid.is_empty:
                    free.append(self.bed)
                else:
                    # Grow by the tolerance as well: simplifying can shave a
                    # corner inward, and the collision area must never end up
                    # smaller than the real one.
                    blocked = solid.buffer(grow + tol, quad_segs=8).simplify(tol)
                    free.append(self.bed.difference(blocked))

            reach = [free[0]]
            for i in range(1, self.n_layers):
                # Simplify the grown region, never the intersection: tidying up
                # afterwards can nudge the boundary back outside `free`, and
                # then a branch judged reachable would in fact be in the model.
                grown = reach[-1].buffer(self.max_move, quad_segs=4).simplify(tol)
                reach.append(free[i].intersection(grown))

            self._free.append(free)
            self._reach.append(reach)

    # -- lookup ------------------------------------------------------------ #

    def bucket(self, radius: float) -> int:
        """Index of the first sampled radius at least as fat as `radius`.

        Rounding up rather than to nearest: a branch judged against a radius
        smaller than its own could be routed into the model.
        """
        idx = int(np.searchsorted(self.radii, float(radius), side="left"))
        return min(idx, len(self.radii) - 1)

    def layer_of(self, z: float) -> int:
        return int(np.clip(round(float(z) / self.pitch), 0, self.n_layers - 1))

    def free(self, bucket: int, layer: int):
        return self._free[bucket][int(np.clip(layer, 0, self.n_layers - 1))]

    def reach(self, bucket: int, layer: int):
        return self._reach[bucket][int(np.clip(layer, 0, self.n_layers - 1))]

    def standable(self, bucket: int, layer: int, to_plate: bool):
        return self.reach(bucket, layer) if to_plate else self.free(bucket, layer)

    def contains(self, region, xy) -> bool:
        if region is None or region.is_empty:
            return False
        return bool(shapely.contains_xy(region, float(xy[0]), float(xy[1])))


def _branch_angle(params: SupportParams) -> float:
    """Lean allowance, clamped so a branch can never out-overhang the limit.

    A branch leaning by `a` degrees overhangs by exactly `a`, so this is the
    entire printability budget for a tree.
    """
    return float(np.clip(params.branch_angle_deg, 1.0, params.printable_overhang_deg - 2.0))


# --------------------------------------------------------------------------- #
# nodes and branches
# --------------------------------------------------------------------------- #


@dataclass
class Branch:
    """One length of tube: a node's path from where it started to where it
    stopped, either at a merge, the plate, or the model."""

    path: list  # [(x, y, z, r)] top -> bottom
    contact: np.ndarray | None = None  # set when this branch starts at the model
    ends_on_plate: bool = False
    ends_on_model: bool = False
    ends_at_merge: bool = False
    land_z: float = 0.0


@dataclass
class _Node:
    xy: np.ndarray
    z: float
    dist_to_tip: float
    merge_r: float
    to_plate: bool
    path: list = field(default_factory=list)
    contact: np.ndarray | None = None

    def radius(self, params: SupportParams) -> float:
        return _radius_at(params, self.merge_r, self.dist_to_tip)

    def record(self, params: SupportParams) -> None:
        self.path.append((float(self.xy[0]), float(self.xy[1]), float(self.z), self.radius(params)))


# --------------------------------------------------------------------------- #
# the descent
# --------------------------------------------------------------------------- #


def build_tree(
    mesh,
    points,
    params: SupportParams | None = None,
    ray: DownRay | None = None,
    field: AvoidanceField | None = None,
) -> SupportBuild:
    """Grow a support tree up to `points` and down to the plate.

    Args:
        mesh: the oriented model, already sitting on z=0.
        points: list[SupportPoint] from stage 2, or an edited version of it.
        params: SupportParams; defaults to the built-in set.
        ray: a DownRay over `mesh`, to reuse across repeated runs.
        field: a prebuilt AvoidanceField. Only depends on the mesh and the
            parameters, not on the points, so it survives point edits.
    """
    params = params or SupportParams()
    points = [p for p in points]
    if not points:
        return SupportBuild(mesh=trimesh.Trimesh(), n_points=0)

    ray = ray or DownRay(mesh)
    top_z = max(float(p.position[2]) for p in points)
    field = field or AvoidanceField(mesh, params, top_z=top_z)

    branches, dropped, warnings, n_merges = _descend(field, ray, points, params)
    parts = [_branch_mesh(b, params) for b in branches]
    parts = [m for m in parts if m is not None and len(m.faces)]

    return SupportBuild(
        mesh=concat(*parts),
        n_points=len(points) - len(dropped),
        n_braces=n_merges,  # a merge is a tree's version of a brace
        dropped=dropped,
        warnings=warnings,
    )


def _descend(field: AvoidanceField, ray: DownRay, points, params: SupportParams):
    """Walk every node from its contact down to the plate or the model."""
    spawn: dict[int, list[SupportPoint]] = {}
    for p in points:
        z = float(p.position[2]) - params.tip_length
        layer = field.layer_of(max(z, 0.0))
        spawn.setdefault(layer, []).append(p)

    branches: list[Branch] = []
    dropped: list[SupportPoint] = []
    warnings: list[str] = []
    live: list[_Node] = []
    n_merges = 0
    n_demoted = 0
    n_model_landings = 0

    top = max(spawn) if spawn else 0
    for i in range(top, -1, -1):
        for pt in spawn.get(i, ()):
            node = _spawn(field, params, pt, i)
            if node is None:
                dropped.append(pt)
            else:
                live.append(node)

        if not live:
            continue

        if i == 0:
            for node in live:
                branches.append(_finish(node, params, on_plate=True, land_z=0.0))
            live = []
            break

        # Targets first, then merging, then commit. Merging has to happen
        # *inside* the step: two nodes join by choosing the same target, so the
        # move that brings them together is bounded by the same one-layer
        # budget as any other move. Joining them after they had both already
        # moved would jump a tube sideways at constant height, which is a 90
        # degree overhang however short it is.
        moving: list[_Node] = []
        targets: list[np.ndarray] = []
        for node in live:
            outcome = _step_target(field, ray, params, node, i - 1, live)
            if isinstance(outcome, tuple):
                land_z, on_model = outcome
                if on_model:
                    n_model_landings += 1
                branches.append(_finish(node, params, on_plate=not on_model, land_z=land_z))
                continue
            moving.append(node)
            targets.append(outcome)

        live, merged_branches, merged_here = _merge(
            field, params, moving, targets, float(field.heights[i - 1])
        )
        branches.extend(merged_branches)
        n_merges += merged_here

        n_demoted += sum(1 for n in live if not n.to_plate)

    for node in live:  # anything still standing at the bottom
        branches.append(_finish(node, params, on_plate=True, land_z=0.0))

    if n_model_landings:
        warnings.append(
            f"{n_model_landings} branch(es) rest on the model; the plate was not "
            "reachable from their contact"
        )
    if dropped:
        warnings.append(f"{len(dropped)} contact point(s) had nowhere to put a branch")
    return branches, dropped, warnings, n_merges


def _spawn(field: AvoidanceField, params: SupportParams, pt: SupportPoint, layer: int):
    """Start a node just under a contact point.

    The contact itself is *on* the model, so it is inside the collision area by
    definition. The tip is allowed to be there — that is its whole job — but the
    branch below it is not, so the node steps out to the nearest standable spot.
    That step bends the tip slightly, which is why it is capped.
    """
    contact = np.asarray(pt.position, dtype=np.float64)
    xy = contact[:2].copy()
    z = field.heights[layer]
    node = _Node(
        xy=xy,
        z=float(z),
        dist_to_tip=max(0.0, float(contact[2]) - float(z)),
        merge_r=params.tip_diameter * 0.5,
        to_plate=True,
        contact=contact,
    )

    bucket = field.bucket(node.radius(params))
    budget = params.tip_length * math.tan(math.radians(_branch_angle(params))) + field.max_move

    for to_plate in (True, False):
        region = field.standable(bucket, layer, to_plate)
        if region is None or region.is_empty:
            continue
        if field.contains(region, xy):
            node.to_plate = to_plate
            node.record(params)
            return node
        q = nearest_points(region, shapely.Point(float(xy[0]), float(xy[1])))[0]
        moved = np.array([q.x, q.y])
        if float(np.linalg.norm(moved - xy)) <= budget:
            node.xy = moved
            node.to_plate = to_plate
            node.record(params)
            return node
    return None


def _step_target(
    field: AvoidanceField, ray: DownRay, params: SupportParams, node: _Node, j: int, live
):
    """Where this node wants to be on layer `j`.

    Returns an ``(x, y)`` target, or ``(land_z, on_model)`` when the node has
    arrived and should be closed off. Nothing is committed here — the caller
    merges targets first.
    """
    z_next = float(field.heights[j])

    # A node that cannot reach the plate is on its way to resting on the model:
    # stop as soon as there is model directly beneath it.
    if not node.to_plate:
        land_z, on_model = _surface_below(ray, node.xy, node.z, params)
        if on_model and z_next <= land_z + params.tree_layer_pitch:
            return land_z, True

    # Judge collision one radius sample fatter than the branch actually is.
    #
    # Without the headroom a branch sits still while it thickens, and then the
    # radius sample steps up, the safe region jumps inward, and it is suddenly
    # too far outside to catch up in one layer — so it gives up on the plate and
    # settles on the model instead. Carrying a sample of slack means it has
    # already drifted clear by the time it grows into that width.
    lookahead = node.dist_to_tip + (node.z - z_next)
    bucket = min(
        field.bucket(_radius_at(params, node.merge_r, lookahead)) + 1,
        len(field.radii) - 1,
    )

    target = _target_xy(field, params, node, j, bucket, live)
    if target is None:
        if node.to_plate:
            node.to_plate = False  # demote; try again on the next layer down
            return node.xy.copy()
        land_z, on_model = _surface_below(ray, node.xy, node.z, params)
        return (land_z, True) if on_model else (0.0, False)
    return target


def _target_xy(field: AvoidanceField, params: SupportParams, node: _Node, j: int, bucket: int, live):
    """Where this node should sit on layer `j`."""
    region = field.standable(bucket, j, node.to_plate)
    if region is None or region.is_empty:
        return None

    p = shapely.Point(float(node.xy[0]), float(node.xy[1]))
    max_move = field.max_move

    if field.contains(region, node.xy):
        goal = _attraction(params, node, live)
        if goal is None:
            return node.xy
        step = goal - node.xy
        dist = float(np.linalg.norm(step))
        if dist <= _EPS:
            return node.xy
        step = step / dist * min(dist, max_move * _attract_fraction(params))
        cand = node.xy + step
        if field.contains(region, cand):
            return cand
        return node.xy  # the pull would put it in the model; stay put

    # Outside: head for the nearest standable point. The reachability sweep
    # guarantees this is within one move for any node that was standable on the
    # layer above, so this is the step that curves a branch around an obstacle.
    q = nearest_points(region, p)[0]
    goal = np.array([q.x, q.y])
    dist = float(np.linalg.norm(goal - node.xy))
    if dist <= max_move + _EPS:
        return goal

    # Too far to arrive this layer. Head that way anyway rather than giving up:
    # the branch has more layers to run, and abandoning the plate here is what
    # produces a support resting on the model when the ground was reachable all
    # along. Only refuse if the step would put it inside the model.
    cand = node.xy + (goal - node.xy) / dist * max_move
    if field.contains(field.free(bucket, j), cand):
        return cand
    return None


def _attract_fraction(params: SupportParams) -> float:
    """How much of a layer's movement budget is spent chasing a neighbour."""
    return 0.35 + 0.45 * float(np.clip(params.merge_strength, 0.0, 1.0))


def _attraction(params: SupportParams, node: _Node, live) -> np.ndarray | None:
    """The neighbour this node should drift toward, if any is worth chasing.

    Without this branches descend in parallel and never quite touch. Cura calls
    it pulling branches towards each other; it is what turns a bundle of sticks
    into a tree.
    """
    strength = float(np.clip(params.merge_strength, 0.0, 1.0))
    if strength <= 0.0 or len(live) < 2:
        return None
    search = params.support_spacing * (0.75 + 2.5 * strength)

    best = None
    best_d = search
    for other in live:
        if other is node:
            continue
        d = float(np.linalg.norm(other.xy - node.xy))
        if d < best_d:
            best_d, best = d, other
    return None if best is None else best.xy.copy()


def _merge(
    field: AvoidanceField,
    params: SupportParams,
    nodes: list[_Node],
    targets: list[np.ndarray],
    z_next: float,
):
    """Commit every node to the layer below, fusing those that arrive together.

    Two nodes merge by *choosing the same target*, so the move that brings them
    together costs no more than any other move and stays inside the branch
    angle. The junction is refused if either node would have to travel more
    than one layer's budget to reach it — the pull in :func:`_attraction` will
    close the gap over the next few layers instead.

    The trunk keeps the deeper ``dist_to_tip`` so it goes on thickening, and a
    radius that conserves cross-section.
    """

    def commit(node: _Node, xy) -> None:
        node.dist_to_tip += node.z - z_next
        node.xy = np.asarray(xy, dtype=np.float64)
        node.z = z_next
        node.record(params)

    if not nodes:
        return [], [], 0
    if len(nodes) == 1:
        commit(nodes[0], targets[0])
        return nodes, [], 0

    strength = float(np.clip(params.merge_strength, 0.0, 1.0))
    slack = 1.0 + 1.5 * strength

    tpos = np.asarray(targets, dtype=np.float64)
    radii = np.array([n.radius(params) for n in nodes])
    kdt = cKDTree(tpos)
    search = float(radii.max() * 2.0 * slack) + _EPS
    layer = field.layer_of(z_next)

    taken = [False] * len(nodes)
    keep: list[_Node] = []
    finished: list[Branch] = []
    n_merged = 0

    for a in np.argsort(-radii):  # fattest first: trunks recruit branches
        if taken[a]:
            continue
        partner, best_d = -1, np.inf
        for b in kdt.query_ball_point(tpos[a], search):
            if b == a or taken[b]:
                continue
            d = float(np.linalg.norm(tpos[b] - tpos[a]))
            if d <= (radii[a] + radii[b]) * slack and d < best_d:
                best_d, partner = d, b
        if partner < 0:
            continue

        na, nb = nodes[a], nodes[partner]
        wa, wb = radii[a] ** 2, radii[partner] ** 2
        xy = (tpos[a] * wa + tpos[partner] * wb) / max(wa + wb, _EPS)

        # Neither branch may lean further than a normal step to get here.
        budget = field.max_move + _EPS
        if np.linalg.norm(xy - na.xy) > budget or np.linalg.norm(xy - nb.xy) > budget:
            continue

        merged_r = min(params.max_branch_diameter * 0.5, math.hypot(radii[a], radii[partner]))
        to_plate = na.to_plate and nb.to_plate
        if not field.contains(field.standable(field.bucket(merged_r), layer, to_plate), xy):
            continue  # the junction itself would sit inside the model

        taken[a] = taken[partner] = True
        n_merged += 1

        for child in (na, nb):
            commit(child, xy)
            # Bury the open end a little way inside the trunk. The tube gets no
            # bottom cap: a flat downward face is a 90 degree overhang wherever
            # it is, and this one would be pure interior anyway.
            x, y, z, r = child.path[-1]
            buried = max(0.0, z - field.pitch * 0.4)  # never below the plate
            if buried < z - _EPS:
                child.path.append((x, y, buried, r))
            finished.append(
                Branch(path=list(child.path), contact=child.contact, ends_at_merge=True)
            )

        keep.append(
            _Node(
                xy=np.asarray(xy, dtype=np.float64),
                z=float(z_next),
                dist_to_tip=max(na.dist_to_tip, nb.dist_to_tip),
                merge_r=merged_r,
                to_plate=to_plate,
                path=[(float(xy[0]), float(xy[1]), float(z_next), merged_r)],
                contact=None,
            )
        )

    for i, node in enumerate(nodes):
        if not taken[i]:
            commit(node, targets[i])
            keep.append(node)
    return keep, finished, n_merged


def _radius_at(params: SupportParams, merge_r: float, dist_to_tip: float) -> float:
    """Branch radius: thickens with depth below the tip, capped."""
    grown = params.tip_diameter * 0.5 + dist_to_tip * math.tan(
        math.radians(params.diameter_angle_deg)
    )
    return float(min(params.max_branch_diameter * 0.5, max(merge_r, grown)))


def _surface_below(ray: DownRay, xy, z: float, params: SupportParams):
    """Highest model surface under a node, or the plate."""
    probe = np.array([[float(xy[0]), float(xy[1]), float(z)]])
    hit_z, _, hit = ray.z_below(probe)
    plate_eps = max(1e-6, params.layer_height * 0.5)
    if bool(hit[0]) and float(hit_z[0]) > plate_eps:
        return float(hit_z[0]), True
    return 0.0, False


def _finish(node: _Node, params: SupportParams, on_plate: bool, land_z: float) -> Branch:
    """Close a branch off at the plate or on the model."""
    sink = params.tip_penetration if not on_plate else 0.0
    z_bot = max(0.0, float(land_z) - sink)
    if node.path and z_bot < node.path[-1][2] - _EPS:
        node.path.append((float(node.xy[0]), float(node.xy[1]), z_bot, node.radius(params)))
    return Branch(
        path=list(node.path),
        contact=node.contact,
        ends_on_plate=on_plate,
        ends_on_model=not on_plate,
        land_z=float(land_z),
    )


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def _branch_mesh(branch: Branch, params: SupportParams) -> trimesh.Trimesh | None:
    """Loft one branch, plus its tip and its foot.

    Reuses the ring machinery in `supports`: a branch is already a list of
    ``(x, y, z, radius)``, which is exactly what `_mesh_rings` lofts, drifting
    centres and all.
    """
    from .supports import _LineXY, _foot_profile, _mesh_rings, _rings_on_axis, _tip_profile

    if len(branch.path) < 2 and branch.contact is None:
        return None

    rings = np.array(branch.path, dtype=np.float64)[::-1]  # _mesh_rings wants bottom-up
    parts = []

    if len(rings) >= 2:
        parts.append(
            _mesh_rings(rings, params.pillar_sections, cap_bottom=not branch.ends_at_merge)
        )

    if branch.contact is not None:
        top = branch.path[0]
        base = np.array([top[0], top[1], top[2]])
        contact = np.asarray(branch.contact, dtype=np.float64)
        tip_len = float(contact[2] - base[2])
        if tip_len > _EPS:
            axis = _LineXY(base, contact + np.array([0.0, 0.0, params.tip_penetration]))
            profile = _tip_profile(float(contact[2]), tip_len, params, float(top[3]))
            parts.append(
                _mesh_rings(
                    _rings_on_axis(axis, profile),
                    params.pillar_sections,
                    # The branch tube's top ring is already here. Capping the
                    # tip's underside as well would bury a flat downward face —
                    # a 90 degree overhang — inside its own support.
                    cap_bottom=len(rings) < 2,
                )
            )

    if branch.ends_on_plate or branch.ends_on_model:
        bottom = branch.path[-1]
        r_top = float(bottom[3])
        if branch.ends_on_plate:
            height = params.foot_height
            r_base = max(params.foot_diameter * 0.5, r_top * 1.6)
        else:
            height = params.pad_height
            r_base = max(params.pad_diameter * 0.5, r_top * 1.2)
        z0 = float(bottom[2])
        axis = _LineXY([bottom[0], bottom[1], z0], [bottom[0], bottom[1], z0 + height])
        parts.append(
            _mesh_rings(_rings_on_axis(axis, _foot_profile(z0, height, r_base, r_top)),
                        params.pillar_sections)
        )

    parts = [m for m in parts if m is not None and len(m.faces)]
    if not parts:
        return None
    return concat(*parts)
