"""Stage 2: deciding *where* supports touch the model.

The output is a plain ``list[SupportPoint]`` and nothing else. No cones, no
pillars, no meshes. The UI edits this list between stages - click to delete a
support, shift-click to add one - and stage 3 turns whatever survives into
geometry. Keeping this stage pure data is what makes that round trip cheap.

Three sources feed the list, in order, each one seeing what the previous ones
already placed:

1. **Islands** (``forced=True``). A cross-section that begins in mid-air has to
   be held, wherever it is and whatever angle its faces are at. These are
   placed first and are never thinned away by anything downstream.
2. **Overhang coverage.** Blue-noise sampling across the faces flagged by
   :func:`rsupport.overhang.overhang_mask`, with the minimum spacing tightened
   where the overhang is steeper: a face pointing straight down sags far worse
   than one leaning at 50 degrees, so it gets roughly twice the point density.
3. **Span fill.** A 0.3 mm contact tip cannot bridge. Anything still further
   than ``params.max_unsupported_span`` from a support point gets one.

Every dimension comes from the passed ``SupportParams``; the only constants in
this file are dimensionless ratios (see CLAUDE.md).

No spatial index is used: ``rtree``/``embree`` are absent, so surface queries
go through :class:`rsupport.raycast.DownRay` and point queries through
``scipy.spatial.cKDTree``.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .overhang import find_islands, overhang_mask, overhang_severity
from .raycast import DownRay
from .types import SupportParams, SupportPoint

__all__ = [
    "place_points",
    "prune_points",
    "sample_overhang_candidates",
]

# --- dimensionless tuning constants -------------------------------------------
# Slice islands at this multiple of the layer height. See find_islands' docs for
# the cost measurement behind the choice; the Z precision given up here is won
# back by its bisection refinement.
ISLAND_STEP_LAYERS = 4

# A face is resting on the build plate, not overhanging, if its whole extent is
# within this many layers of the plate.
BED_LAYERS = 2.0

# The build plate. Fixed, not read off the model: stage 1 sets the model down at
# z=0 or floats it above by ``lift_height``, and either way the plate stays here.
PLATE_Z = 0.0

# Minimum spacing at severity 1 (straight down) as a fraction of
# params.support_spacing. 0.5 means flat undersides get twice the density of a
# face sitting right on the overhang threshold.
SEVERITY_TIGHTEN = 0.5

# Span fill also covers faces up to this multiple of the overhang angle, i.e.
# shallower faces that the mask skipped but that still sag over a long run.
# 1.0 would restrict span fill to the strict overhang region, at which point it
# almost never fires, because the blue-noise pass is already maximal at a
# spacing tighter than max_unsupported_span.
SPAN_ANGLE_SCALE = 1.25

# How many candidates to throw at the dart-throwing pass per final point.
OVERSAMPLE = 20
MIN_CANDIDATES = 512
MAX_CANDIDATES = 60_000

# Weight of the detail metric when ordering candidates. 0 ignores detail; 1
# makes it as important as the overhang severity itself.
DETAIL_PRIORITY_WEIGHT = 0.5

_SEED = 0


# ------------------------------------------------------------------- helpers


def _as_face_array(detail, mesh) -> np.ndarray | None:
    """Normalise the optional per-face detail metric to (F,) in 0..1.

    Accepts an (F,) array or a callable taking the mesh and returning one, so
    ``detail=rsupport.detail.face_detail`` works as well as a precomputed
    array. Anything else, or a wrong-length array, is ignored rather than
    raising - detail is an ordering hint, not a correctness input.
    """
    n_faces = len(mesh.faces)
    if detail is None:
        return None
    if callable(detail):
        try:
            detail = detail(mesh)
        except Exception:
            return None
    arr = np.asarray(detail, dtype=np.float64).ravel()
    if arr.size != n_faces:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    span = float(arr.max() - arr.min())
    if span <= 0:
        return np.zeros(n_faces)
    return (arr - arr.min()) / span


def _bed_faces(mesh, params: SupportParams) -> np.ndarray:
    """(F,) bool: faces lying on the build plate.

    The underside of a cone standing on its base points straight down and would
    otherwise be the single most heavily supported surface on the model.

    Measured against the plate at z=0, *not* against the model's own lowest
    point. Those are the same thing only when the model is set down flat; with
    ``lift_height`` the model floats, and then its underside is genuinely
    printing into air and wants supporting like any other 90 degree overhang.
    """
    tri_z = np.asarray(mesh.triangles, dtype=np.float64)[:, :, 2]
    return tri_z.max(axis=1) <= PLATE_Z + params.layer_height * BED_LAYERS


def _supportable(points: np.ndarray, mesh, params: SupportParams, ray: DownRay) -> np.ndarray:
    """(N,) bool: can a pillar actually be built under each point?

    Two rejections, both about places a support would be pointless:

    * The point sits on the *inside* of the model - the underside of a crossbar
      directly above the post that already holds it up. Probed by sampling one
      layer height below the surface and asking the parity test.
    * There is less than a tip's length of clear air beneath it. Stage 3 could
      not fit a contact cone in there even if it wanted to.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    probe = points.copy()
    probe[:, 2] -= params.layer_height
    buried = ray.inside(probe)

    z_below, _, hit = ray.z_below(points)
    # Nothing below means the plate is what is below, whether or not the model
    # is sitting on it — a lifted model has air under its whole footprint.
    floor = np.where(hit, z_below, PLATE_Z)
    clearance = points[:, 2] - floor

    return (~buried) & (clearance > params.tip_length)


def _project_to_surface(
    mesh, ray: DownRay, xy: np.ndarray, z: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the model surface at ``xy`` nearest height ``z``, facing downward.

    Islands are found by slicing, which gives a location but not a face. This
    walks the vertical column at that XY and picks the crossing closest to
    ``z``, preferring downward-facing surfaces - the underside of the floating
    feature is what wants holding, not its top.
    """
    query = np.array([[xy[0], xy[1], z]], dtype=np.float64)
    _, faces, heights = ray.column_hits(query)
    if len(faces) == 0:
        return None

    normals = np.asarray(mesh.face_normals, dtype=np.float64)[faces]
    downward = normals[:, 2] < 0.0
    dist = np.abs(heights - z)
    pool = np.nonzero(downward)[0]
    if len(pool) == 0:
        pool = np.arange(len(faces))
    best = pool[np.argmin(dist[pool])]

    position = np.array([xy[0], xy[1], heights[best]], dtype=np.float64)
    return position, normals[best]


def _radii(severity: np.ndarray, params: SupportParams) -> np.ndarray:
    """Per-point minimum spacing, tightened as the overhang gets steeper."""
    threshold = float(np.cos(np.radians(np.clip(params.overhang_angle_deg, 0.0, 90.0))))
    denom = max(1.0 - threshold, 1e-6)
    steep = np.clip((severity - threshold) / denom, 0.0, 1.0)
    return params.support_spacing * (1.0 - SEVERITY_TIGHTEN * steep)


# ------------------------------------------------------------------ sampling


def sample_overhang_candidates(
    mesh,
    params: SupportParams,
    angle_deg: float | None = None,
    count: int | None = None,
    seed: int = _SEED,
    filter_supportable: bool = True,
    ray=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense random points on the faces that need support.

    Exposed because the same candidate set is what a coverage check has to be
    measured against - the tests use it, and so does the span-fill pass.

    Args:
        mesh: a ``trimesh.Trimesh`` in print orientation.
        params: dimensions.
        angle_deg: overhang angle to select faces with. Defaults to
            ``params.overhang_angle_deg``.
        count: number of raw samples. Defaults to a density derived from the
            overhang area and the tightest spacing.
        seed: RNG seed, so stage 2 is reproducible.
        filter_supportable: drop points that are buried inside the model or have
            no room for a contact tip beneath them.
        ray: an existing :class:`rsupport.raycast.DownRay`; built if omitted.

    Returns:
        ``(points (N,3), face_index (N,), severity (N,))``.
    """
    angle = params.overhang_angle_deg if angle_deg is None else float(angle_deg)
    severity = overhang_severity(mesh)
    mask = overhang_mask(mesh, angle) & ~_bed_faces(mesh, params)

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    weights = np.where(mask, areas, 0.0)
    total = float(weights.sum())
    empty = (np.zeros((0, 3)), np.zeros(0, dtype=np.intp), np.zeros(0))
    if total <= 0:
        return empty

    if count is None:
        tightest = params.support_spacing * (1.0 - SEVERITY_TIGHTEN)
        count = int(OVERSAMPLE * total / max(tightest * tightest, 1e-9))
        count = int(np.clip(count, MIN_CANDIDATES, MAX_CANDIDATES))
    if count <= 0:
        return empty

    points, faces = trimesh.sample.sample_surface(
        mesh, count, face_weight=weights, seed=seed
    )
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.intp)

    if filter_supportable:
        ok = _supportable(points, mesh, params, ray if ray is not None else DownRay(mesh))
        points, faces = points[ok], faces[ok]

    return points, faces, severity[faces]


def _thin(
    points: np.ndarray,
    radii: np.ndarray,
    priority: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Variable-radius dart throwing over a fixed candidate set.

    Classic greedy Poisson elimination: build one static KD-tree over every
    candidate, walk them in priority order, and each time one is accepted kill
    every candidate inside its radius. One tree, no rebuilds, O(N k).

    ``seeds`` are points already committed (the island contacts); candidates
    inside their radius are killed before anything else is considered, so no
    overhang point ever lands on top of a forced one.

    Returns:
        Indices of the accepted candidates.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.intp)

    alive = np.ones(n, dtype=bool)
    if len(seeds):
        d_seed, _ = cKDTree(seeds).query(points)
        alive &= d_seed >= radii

    tree = cKDTree(points)
    accepted: list[int] = []
    for i in np.argsort(-priority, kind="stable"):
        if not alive[i]:
            continue
        alive[i] = False
        accepted.append(int(i))
        killed = tree.query_ball_point(points[i], radii[i])
        alive[killed] = False
    return np.asarray(sorted(accepted), dtype=np.intp)


def _span_fill(
    coverage: np.ndarray,
    supported: np.ndarray,
    max_span: float,
    limit: int,
) -> list[int]:
    """Indices into ``coverage`` to promote so nothing is left to bridge.

    Greedy farthest-point insertion: repeatedly take whatever is furthest from
    any support and support it. The distance array is updated in place rather
    than rebuilding a tree, so each insertion is O(N).
    """
    if len(coverage) == 0:
        return []

    if len(supported):
        dist, _ = cKDTree(supported).query(coverage)
    else:
        dist = np.full(len(coverage), np.inf)

    added: list[int] = []
    while len(added) < limit:
        i = int(np.argmax(dist))
        if not (dist[i] > max_span):
            break
        added.append(i)
        dist = np.minimum(dist, np.linalg.norm(coverage - coverage[i], axis=1))
    return added


# ------------------------------------------------------------------ pipeline


def place_points(mesh, params: SupportParams, ray=None, detail=None) -> list[SupportPoint]:
    """The full stage-2 pipeline: islands + overhang coverage + span fill.

    Args:
        mesh: a ``trimesh.Trimesh``, already in its print orientation with its
            lowest point at the build plate.
        params: a :class:`rsupport.types.SupportParams`.
        ray: an existing :class:`rsupport.raycast.DownRay` for this mesh.
            Built if omitted; pass one in when re-running stage 2 with new
            placement parameters, since it is the expensive part to rebuild.
        detail: optional (F,) per-face detail-density metric from
            :mod:`rsupport.detail`. Only affects *ordering*: where two
            candidates compete for the same spot, the one on plainer geometry
            wins, which pushes contact scars off a miniature's face and onto
            its cloak. It never removes a needed support.

    Returns:
        Support points, islands first. Positions lie on the model surface and
        normals are the outward face normal there.
    """
    if not isinstance(params, SupportParams):
        raise TypeError("params must be a SupportParams (see rsupport.presets)")
    if ray is None:
        ray = DownRay(mesh)
    detail_arr = _as_face_array(detail, mesh)

    points: list[SupportPoint] = []

    # --- 1. islands: non-negotiable, placed before anything can crowd them ---
    islands = find_islands(
        mesh,
        params.layer_height,
        params.island_min_area,
        step=params.layer_height * ISLAND_STEP_LAYERS,
    )
    for island in islands:
        found = _project_to_surface(mesh, ray, island.centroid[:2], island.z)
        if found is None:
            position = np.asarray(island.centroid, dtype=np.float64)
            normal = np.array([0.0, 0.0, -1.0])
        else:
            position, normal = found
        points.append(
            SupportPoint(position=position, normal=normal, forced=True, source="island")
        )

    island_xyz = (
        np.array([p.position for p in points], dtype=np.float64)
        if points
        else np.zeros((0, 3))
    )

    # --- 2. overhang coverage: blue noise, denser where it points down more ---
    cand, cand_faces, cand_sev = sample_overhang_candidates(mesh, params, ray=ray)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    if len(cand):
        radii = _radii(cand_sev, params)
        priority = cand_sev.astype(np.float64)
        if detail_arr is not None:
            priority = priority - DETAIL_PRIORITY_WEIGHT * detail_arr[cand_faces]
        for i in _thin(cand, radii, priority, island_xyz):
            points.append(
                SupportPoint(
                    position=cand[i].copy(),
                    normal=normals[cand_faces[i]].copy(),
                    forced=False,
                    source="overhang",
                )
            )

    # --- 3. span fill: nothing left further than a tip can bridge ------------
    relaxed_angle = min(90.0, params.overhang_angle_deg * SPAN_ANGLE_SCALE)
    shallow, shallow_faces, _ = sample_overhang_candidates(
        mesh, params, angle_deg=relaxed_angle, seed=_SEED + 1, ray=ray
    )
    if len(shallow) or len(cand):
        coverage = np.vstack([c for c in (cand, shallow) if len(c)])
        coverage_faces = np.concatenate(
            [f for f, c in ((cand_faces, cand), (shallow_faces, shallow)) if len(c)]
        )
        supported = (
            np.array([p.position for p in points], dtype=np.float64)
            if points
            else np.zeros((0, 3))
        )
        for i in _span_fill(
            coverage, supported, params.max_unsupported_span, limit=len(coverage)
        ):
            points.append(
                SupportPoint(
                    position=coverage[i].copy(),
                    normal=normals[coverage_faces[i]].copy(),
                    forced=False,
                    source="span",
                )
            )

    return points


def prune_points(points, min_spacing: float) -> list[SupportPoint]:
    """Thin a point list while keeping every forced point.

    Forced points are seeded first, so an island contact can never be the one
    that gets dropped. The remainder keep their original order, which makes the
    result stable when the UI re-runs this after an edit.

    Args:
        points: the list to thin.
        min_spacing: mm. Kept points are at least this far apart, except where
            two forced points were already closer.

    Returns:
        A new list; the SupportPoint objects themselves are not copied.
    """
    points = list(points)
    if min_spacing <= 0 or not points:
        return points

    kept_idx = [i for i, p in enumerate(points) if p.forced]
    coords = [np.asarray(points[i].position, dtype=np.float64) for i in kept_idx]

    for i, p in enumerate(points):
        if p.forced:
            continue
        pos = np.asarray(p.position, dtype=np.float64)
        if coords and np.min(np.linalg.norm(np.asarray(coords) - pos, axis=1)) < min_spacing:
            continue
        kept_idx.append(i)
        coords.append(pos)

    return [points[i] for i in sorted(kept_idx)]
