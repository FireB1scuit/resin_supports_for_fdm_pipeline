"""Stage 1: pick a print orientation.

A pose is described entirely by which direction of the *original* mesh ends up
pointing at the build plate. Everything else — the spin about Z, where the model
sits on the bed — is irrelevant to how well it prints, so every candidate here is
generated as a "down direction", turned into a rotation, and dropped to the bed.

The candidate set is the standard one (any resting pose puts a convex-hull facet
on the plate) plus three additions that matter for miniatures: the largest flat
patch in the mesh, which is almost always an integral round base and is what a
human would set flat; the PCA principal axis stood upright; and lean variants of
those, tilted a few degrees at a time so the detail-bearing front rides up and
the supports land on the back.

Scoring is a weighted sum of normalised terms — see :data:`WEIGHTS`. Lower wins.

**The weights are a taste problem, not a correctness problem.** They need tuning
against a shelf of real models, and the whole reason :func:`orientations` returns
a ranked list rather than a single answer is so a bad pick costs the user one
click. Tune here; do not scatter magic numbers through the scoring functions.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .detail import detail_direction, face_detail
from .types import Orientation, SupportParams

# `overhang` is being written in parallel; scoring degrades gracefully without
# it rather than refusing to run.
try:  # pragma: no cover - depends on sibling module landing
    from .overhang import find_islands
except ImportError:  # pragma: no cover
    find_islands = None

__all__ = [
    "WEIGHTS",
    "apply",
    "best_orientation",
    "candidate_matrices",
    "orientations",
    "score_orientation",
]

#: Term weights. Every term is normalised to roughly 0..1 before weighting, so
#: these numbers are directly comparable and mean the same thing on a 28 mm mini
#: as on a 200 mm dragon.
#:
#: ``detail_penalty`` dominates deliberately: keeping support scars off a
#: miniature's face is the entire point of this preset, and we would rather pay
#: for more support material than sand a nose flat. ``stability`` is next
#: because a pose that tips over does not print at all. ``support_cost`` is the
#: baseline unit. ``height`` is a mild tiebreak (shorter is faster and less
#: wobbly) and must stay mild or it lays every model on its side.
WEIGHTS: dict[str, float] = {
    "support_cost": 1.0,
    "detail_penalty": 6.0,
    "stability": 2.0,
    "height": 0.3,
    "island_count": 1.5,
}

#: Subtracted from the score of the "flat-base" candidate. Sized to win near
#: ties only — it should never drag a genuinely bad pose to the top.
FLAT_BASE_BONUS = 0.25

#: Added to `stability` when the centre of mass projects outside the bed-contact
#: hull. Large: this pose falls over.
COM_OUTSIDE_PENALTY = 3.0

#: Two candidate down-directions closer than this are the same pose.
DEDUPE_ANGLE_DEG = 2.0

#: Scoring every candidate is cheap but not free; cap the set.
MAX_CANDIDATES = 150

#: Lean variants step this far apart, up to ``params.max_lean_deg``.
LEAN_STEP_DEG = 5.0

#: A flat patch smaller than this fraction of total surface area is a facet, not
#: a base, and gets no bonus.
MIN_FLAT_PATCH_FRACTION = 0.02

#: Island detection slices the model, which is far too slow to run on every
#: candidate. Only the best few from the cheap pass get scored with islands, and
#: even then the slice pitch is coarsened so no model exceeds this many layers.
ISLAND_SCORING_LAYERS = 60
ISLAND_REFINE_COUNT = 8

_Z = np.array([0.0, 0.0, 1.0])
_DOWN = np.array([0.0, 0.0, -1.0])
_EPS = 1e-12


# --------------------------------------------------------------------- helpers


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros(3)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation (det +1) taking unit vector `a` onto unit vector `b`."""
    a, b = _unit(a), _unit(b)
    if not a.any() or not b.any():
        return np.eye(3)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s <= _EPS:
        if c > 0:
            return np.eye(3)
        # Antiparallel: a half turn about any axis perpendicular to `a`.
        axis = _unit(np.cross(a, _Z if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])))
        return trimesh.transformations.rotation_matrix(np.pi, axis)[:3, :3]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _pose_matrix(mesh, down: np.ndarray) -> np.ndarray:
    """4x4 taking `mesh` into the pose where `down` faces the build plate.

    The result already includes the drop-to-bed translation: lowest point at
    z=0, bounding box centred on the origin in XY. Matches
    ``mesh_io.drop_to_bed``.
    """
    rot = _rotation_between(down, _DOWN)
    verts = np.asarray(mesh.vertices, dtype=np.float64) @ rot.T
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [-(lo[0] + hi[0]) * 0.5, -(lo[1] + hi[1]) * 0.5, -lo[2]]
    return matrix


def _principal_axis(mesh) -> np.ndarray:
    """Longest axis of the model, by area-weighted PCA of face centroids.

    Weighting by area rather than counting vertices keeps a densely tessellated
    head from outvoting a smooth torso.
    """
    centres = np.asarray(mesh.triangles_center, dtype=np.float64)
    weights = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(weights.sum())
    if len(centres) < 2 or total <= _EPS:
        return _Z.copy()
    mean = (weights @ centres) / total
    d = centres - mean
    cov = (d * weights[:, None]).T @ d / total
    _, vecs = np.linalg.eigh(cov)
    return _unit(vecs[:, -1])


def _largest_flat_patch(mesh) -> tuple[np.ndarray, float] | None:
    """Outward normal and area of the biggest coplanar patch, if there is one.

    Greedy clustering seeded from the largest faces: take the biggest unclaimed
    face, sweep up every face sharing its normal *and* its plane offset, and
    move on. Both tests are needed — the top and bottom of a disc base are
    parallel but are not the same patch.

    A patch only counts if it is the *outermost* thing along its own normal.
    Otherwise the top face of a base disc, or any large internal partition in a
    model assembled from overlapping shells, would qualify as somewhere the
    model could rest — and it cannot, because the rest of the model is in the
    way.
    """
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(areas) == 0:
        return None
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    centres = np.asarray(mesh.triangles_center, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    offsets = np.einsum("ij,ij->i", normals, centres)

    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= _EPS:
        return None
    plane_tol = diagonal * 1e-3
    normal_tol = np.cos(np.radians(3.0))

    used = np.zeros(len(areas), dtype=bool)
    order = np.argsort(areas)[::-1]
    best_normal = None
    best_area = 0.0

    # Only the largest faces are worth seeding from; a flat base is never made
    # exclusively of the mesh's smallest triangles.
    for seed in order[: min(len(order), 200)]:
        if used[seed]:
            continue
        mask = (
            (normals @ normals[seed] >= normal_tol)
            & (np.abs(offsets - offsets[seed]) <= plane_tol)
            & ~used
        )
        used |= mask
        patch_area = float(areas[mask].sum())
        if patch_area <= best_area:
            continue
        # Extremality check, deferred until the patch is a contender because it
        # costs a pass over every vertex.
        if offsets[seed] < float((verts @ normals[seed]).max()) - plane_tol:
            continue
        best_area = patch_area
        best_normal = _unit((areas[mask, None] * normals[mask]).sum(axis=0))

    if best_normal is None or not best_normal.any():
        return None
    if best_area < float(areas.sum()) * MIN_FLAT_PATCH_FRACTION:
        return None
    return best_normal, best_area


def _lean_axes(rot: np.ndarray, detail_dir: np.ndarray) -> list[np.ndarray]:
    """Axes to tilt about so the detail-bearing side rides upward.

    Returns axes in *posed* space (apply after `rot`). Falls back to both
    horizontal axes, in both directions, when the model has no detail bias — a
    symmetric model gives a zero ``detail_direction`` and there is then no
    reason to prefer one lean over another.
    """
    if detail_dir is not None and np.asarray(detail_dir).any():
        posed = rot @ np.asarray(detail_dir, dtype=np.float64)
        horizontal = np.array([posed[0], posed[1], 0.0])
        if float(np.linalg.norm(horizontal)) > 1e-6:
            # Rotating about h x z by +theta raises the detail side. (Rodrigues:
            # d(v.z)/dtheta = |h| > 0 for this axis.)
            return [_unit(np.cross(horizontal, _Z))]
    return [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]


# ------------------------------------------------------------------ candidates


def candidate_matrices(mesh, params: SupportParams) -> list[tuple[np.ndarray, str]]:
    """Poses worth scoring, each as a (4x4 matrix, label) pair.

    Matrices already include the drop-to-bed translation, so they can be handed
    straight to :func:`score_orientation` or :func:`apply`.

    Ordering is by priority, not quality: the specially-labelled candidates come
    first so that when two candidates collapse together during deduplication it
    is the meaningful label — "flat-base", "upright" — that survives.
    """
    down_dirs: list[tuple[np.ndarray, str]] = []

    flat = _largest_flat_patch(mesh)
    base_rot = None
    if flat is not None:
        base_normal = flat[0]
        down_dirs.append((base_normal, "flat-base"))
        base_rot = _rotation_between(base_normal, _DOWN)

    axis = _principal_axis(mesh)
    upright_rot = _rotation_between(-axis, _DOWN)
    down_dirs.append((-axis, "upright"))
    down_dirs.append((axis, "upright-flipped"))

    # Lean variants. Tilting the model away from a resting pose is the move that
    # walks overhangs off the front and onto the back, which no hull facet does.
    detail_dir = detail_direction(mesh)
    max_lean = max(0.0, float(params.max_lean_deg))
    steps = np.arange(LEAN_STEP_DEG, max_lean + 1e-6, LEAN_STEP_DEG)
    for label, rot in (("upright", upright_rot), ("base", base_rot)):
        if rot is None:
            continue
        for lean_axis in _lean_axes(rot, detail_dir):
            for deg in steps:
                tilt = trimesh.transformations.rotation_matrix(np.radians(deg), lean_axis)[:3, :3]
                leaned = tilt @ rot
                # The down-direction of a rotation R is whatever R maps to -Z.
                down_dirs.append((-leaned[2, :], f"{label}-lean-{int(round(deg))}"))

    # The standard candidate set: any way the object can rest is a hull facet
    # against the plate. Largest facets first — they are the plausible poses.
    hull = mesh.convex_hull
    hull_normals = np.asarray(hull.face_normals, dtype=np.float64)
    hull_areas = np.asarray(hull.area_faces, dtype=np.float64)
    for idx in np.argsort(hull_areas)[::-1]:
        down_dirs.append((hull_normals[idx], "hull"))

    return _build_unique(mesh, down_dirs)


def _build_unique(mesh, down_dirs) -> list[tuple[np.ndarray, str]]:
    """Deduplicate down-directions by angle, then build matrices."""
    cos_tol = np.cos(np.radians(DEDUPE_ANGLE_DEG))
    kept: list[np.ndarray] = []
    out: list[tuple[np.ndarray, str]] = []
    for raw, label in down_dirs:
        d = _unit(raw)
        if not d.any():
            continue
        if kept and float(np.max(np.asarray(kept) @ d)) >= cos_tol:
            continue
        kept.append(d)
        out.append((_pose_matrix(mesh, d), label))
        if len(out) >= MAX_CANDIDATES:
            break
    return out


# --------------------------------------------------------------------- scoring


def score_orientation(
    mesh,
    matrix: np.ndarray,
    params: SupportParams,
    detail: np.ndarray | None = None,
    *,
    islands: bool = True,
) -> tuple[float, dict]:
    """Score one pose. Lower is better.

    Args:
        mesh: the mesh in its *original* pose.
        matrix: 4x4 transform into the pose being scored.
        params: the active parameter set.
        detail: precomputed :func:`~rsupport.detail.face_detail`, to avoid
            recomputing it for every candidate.
        islands: run per-layer island detection. Slicing is expensive, so the
            cheap first pass turns this off and the term contributes 0.

    Returns:
        ``(score, terms)``. ``terms`` holds the *unweighted* normalised value of
        every term, so the UI can explain a decision.
    """
    if detail is None:
        detail = face_detail(mesh)

    matrix = np.asarray(matrix, dtype=np.float64)
    rot = matrix[:3, :3]
    verts = np.asarray(mesh.vertices, dtype=np.float64) @ rot.T + matrix[:3, 3]
    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ rot.T
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Heights are measured from whatever the actual lowest point is, so a caller
    # passing a rotation without the drop still gets a sane answer.
    bed_z = float(verts[:, 2].min())
    height = float(verts[:, 2].max()) - bed_z
    total_area = float(areas.sum())
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(hi - lo))
    if total_area <= _EPS or diagonal <= _EPS:
        return 0.0, {k: 0.0 for k in WEIGHTS}

    face_z = verts[faces, 2].mean(axis=1) - bed_z

    # --- overhangs -------------------------------------------------------
    # A face needs support when its normal tips more than `overhang_angle_deg`
    # off vertical, i.e. when its downward component passes sin(angle).
    downward = -normals[:, 2]
    threshold = float(np.sin(np.radians(params.overhang_angle_deg)))
    severity = np.clip((downward - threshold) / max(1.0 - threshold, _EPS), 0.0, 1.0)
    # Faces resting on the plate are held by the plate, not by supports.
    severity = np.where(face_z > params.layer_height, severity, 0.0)

    weighted = areas * severity
    support_cost = float((weighted * face_z).sum()) / (total_area * max(height, _EPS))
    detail_penalty = float((weighted * detail).sum()) / total_area

    # --- stability -------------------------------------------------------
    contact = (face_z <= params.layer_height) & (downward > 0.0)
    contact_area = float((areas[contact] * downward[contact]).sum())
    silhouette = _hull_area_2d(verts[:, :2])
    footprint = contact_area / silhouette if silhouette > _EPS else 0.0
    stability = 1.0 - float(np.clip(footprint, 0.0, 1.0))

    com = _centre_of_mass(mesh) @ rot.T + matrix[:3, 3]
    contact_pts = verts[np.unique(faces[contact].ravel()), :2] if contact.any() else np.zeros((0, 2))
    if not _point_in_hull(com[:2], contact_pts):
        stability += COM_OUTSIDE_PENALTY

    # --- the rest --------------------------------------------------------
    height_term = height / diagonal

    island_term = 0.0
    if islands:
        island_term = _island_term(mesh, matrix, params, height)

    terms = {
        "support_cost": support_cost,
        "detail_penalty": detail_penalty,
        "stability": stability,
        "height": height_term,
        "island_count": island_term,
    }
    score = sum(WEIGHTS[k] * v for k, v in terms.items())
    return float(score), terms


def _centre_of_mass(mesh) -> np.ndarray:
    """Centre of mass, falling back to an area-weighted centroid.

    ``center_mass`` is only meaningful on a watertight mesh, and plenty of
    real-world STLs are not.
    """
    try:
        if mesh.is_watertight and mesh.volume > 0:
            return np.asarray(mesh.center_mass, dtype=np.float64)
    except Exception:
        pass
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(areas.sum())
    if total <= _EPS:
        return np.asarray(mesh.centroid, dtype=np.float64)
    return (areas @ np.asarray(mesh.triangles_center, dtype=np.float64)) / total


def _hull_area_2d(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # 2D "volume" is area
    except (QhullError, ValueError):
        return 0.0


def _point_in_hull(point: np.ndarray, points: np.ndarray) -> bool:
    """Does `point` fall inside the 2D convex hull of `points`?

    A degenerate contact region — one vertex, or a line of them — cannot hold
    anything up, so it counts as outside.
    """
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return False
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return False
    return bool(np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 0.0))


def _island_term(mesh, matrix: np.ndarray, params: SupportParams, height: float) -> float:
    """Fraction of layers that start a new island.

    Every island is a forced support: geometry that appears out of thin air with
    nothing beneath it. Fewer is better.
    """
    if find_islands is None or height <= _EPS:
        return 0.0
    pitch = max(params.support_layer_height, height / ISLAND_SCORING_LAYERS)
    try:
        posed = mesh.copy()
        posed.apply_transform(matrix)
        found = find_islands(posed, pitch, params.island_min_area)
    except Exception:
        # `overhang` is a sibling module under active development. A failure
        # there must not take orientation down with it.
        return 0.0
    n_layers = max(1.0, height / pitch)
    return float(min(1.0, len(found) / n_layers))


# ----------------------------------------------------------------------- entry


def orientations(mesh, params: SupportParams, top_k: int = 3) -> list[Orientation]:
    """Best-first candidate poses.

    Runs in two passes: every candidate gets the cheap terms, then only the
    leaders are re-scored with island detection, which needs the mesh actually
    sliced and is far too slow to run 150 times.
    """
    candidates = candidate_matrices(mesh, params)
    if not candidates:
        return []

    detail = face_detail(mesh)

    coarse: list[tuple[float, np.ndarray, str, dict]] = []
    for matrix, label in candidates:
        score, terms = score_orientation(mesh, matrix, params, detail, islands=False)
        score, terms = _apply_bonus(score, terms, label)
        coarse.append((score, matrix, label, terms))
    coarse.sort(key=lambda item: item[0])

    refine_n = max(top_k, ISLAND_REFINE_COUNT)
    results: list[Orientation] = []
    for score, matrix, label, terms in coarse[:refine_n]:
        if find_islands is not None:
            score, terms = score_orientation(mesh, matrix, params, detail, islands=True)
            score, terms = _apply_bonus(score, terms, label)
        results.append(Orientation(matrix=matrix, score=score, terms=terms, label=label))

    results.sort(key=lambda o: o.score)
    return results[: max(1, top_k)]


def _apply_bonus(score: float, terms: dict, label: str) -> tuple[float, dict]:
    """A human sets an integral base flat, so let it win near-ties."""
    bonus = -FLAT_BASE_BONUS if label == "flat-base" else 0.0
    terms = dict(terms)
    terms["bonus"] = bonus
    return score + bonus, terms


def best_orientation(mesh, params: SupportParams) -> Orientation:
    """The single best pose."""
    found = orientations(mesh, params, top_k=1)
    if found:
        return found[0]
    # Degenerate mesh: identity, dropped to the bed, is still a valid answer.
    return Orientation(matrix=np.eye(4), score=float("inf"), terms={}, label="identity")


def apply(mesh, orientation) -> trimesh.Trimesh:
    """Return a copy of `mesh` transformed into `orientation`.

    Accepts an :class:`~rsupport.types.Orientation` or a bare 4x4 matrix.
    """
    matrix = getattr(orientation, "matrix", orientation)
    out = mesh.copy()
    out.apply_transform(np.asarray(matrix, dtype=np.float64))
    return out
