"""Per-face "detail density" — where the sculpted detail lives on a model.

The orientation scorer needs to know the difference between a miniature's face,
hands and front crest and the flat back of its cloak, so it can prefer poses
that put support contacts on the boring side. There is no semantic
understanding available here, so we use the geometric signature of sculpted
detail: *lots of angular variation packed into a small area*.

The metric is a discrete curvature density. For every interior edge, the
integrated mean curvature is proportional to ``edge_length * dihedral_angle``.
Summing that over a face's edges and dividing by the face's own area gives a
quantity with units of 1/length — high where the surface wrinkles tightly, low
on big flat panels and on gentle, large-radius curves. Multiplying by the
model's diagonal makes it dimensionless, so the numbers mean the same thing on
a 28 mm mini and a 200 mm terrain piece.

Two rounds of smoothing over face adjacency turn the raw per-face values into
regions, which is what we actually want to reason about: one accidentally sharp
triangle is noise, a hundred of them next to each other is a face.

Known limitation: this cannot tell *sculpted* detail from *mechanical* detail.
A cube's edges are genuinely high curvature and score high; so do the teeth of
a gear. For organic miniatures — the target of this project — it behaves.
"""

from __future__ import annotations

import numpy as np

__all__ = ["face_detail", "detail_direction"]

#: Rounds of neighbour averaging. Two is enough to read as regions without
#: washing out the boundary between a detailed front and a plain back.
SMOOTH_ITERATIONS = 2

#: Values are divided by this percentile of the smoothed field and clipped, so
#: the output lands in roughly 0..1 regardless of how wrinkly the model is.
#: Using a percentile rather than the max keeps a handful of degenerate
#: triangles from flattening the whole scale.
NORMALISE_PERCENTILE = 97.0

#: Sliver triangles have near-zero area and would divide to infinity. Clamp the
#: denominator to a fraction of the mean face area — relative, so it carries no
#: implied millimetre value.
_MIN_AREA_FRACTION = 0.01

_EPS = 1e-12


def face_detail(mesh) -> np.ndarray:
    """Per-face detail density.

    Args:
        mesh: a ``trimesh.Trimesh``.

    Returns:
        (F,) float array, >= 0 and normalised to roughly 0..1. Higher means
        "this neighbourhood is sculpted detail"; 0 means flat panel.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.float64)

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    mean_area = float(areas.mean()) if len(areas) else 0.0
    denom = np.maximum(areas, max(mean_area * _MIN_AREA_FRACTION, _EPS))

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    raw = np.zeros(n_faces, dtype=np.float64)

    if len(adjacency):
        angles = np.abs(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
        edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        edge_len = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)

        # Integrated mean curvature of the edge, split evenly between the two
        # faces that share it.
        contribution = 0.5 * edge_len * angles
        np.add.at(raw, adjacency[:, 0], contribution)
        np.add.at(raw, adjacency[:, 1], contribution)

    raw /= denom

    # Make it dimensionless: a scaled copy of a model must score identically.
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= _EPS:
        return np.zeros(n_faces, dtype=np.float64)
    raw *= diagonal

    smoothed = _smooth(raw, adjacency, n_faces, SMOOTH_ITERATIONS)

    reference = float(np.percentile(smoothed, NORMALISE_PERCENTILE))
    if not np.isfinite(reference) or reference <= _EPS:
        return np.zeros(n_faces, dtype=np.float64)

    return np.clip(smoothed / reference, 0.0, 1.0)


def detail_direction(mesh) -> np.ndarray:
    """Unit vector pointing at the model's most detail-bearing side.

    An area-and-detail-weighted sum of face normals: the side carrying more
    sculpted surface pulls the resultant toward itself. For a miniature this
    comes out pointing roughly out of its chest and face.

    Returns:
        (3,) unit vector, or a zero vector when the model is detail-symmetric
        (a sphere, a gear, a well-balanced bust) and no side wins. **Callers
        must handle the zero vector** — see ``orient._lean_axes``.
    """
    if len(mesh.faces) == 0:
        return np.zeros(3, dtype=np.float64)

    detail = face_detail(mesh)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    weights = areas * detail
    total = float(weights.sum())
    if total <= _EPS:
        return np.zeros(3, dtype=np.float64)

    resultant = weights @ normals
    magnitude = float(np.linalg.norm(resultant))
    # A symmetric model cancels to numerical noise; that is not a direction.
    if magnitude <= total * 1e-6:
        return np.zeros(3, dtype=np.float64)
    return resultant / magnitude


def _smooth(values: np.ndarray, adjacency: np.ndarray, n_faces: int, rounds: int) -> np.ndarray:
    """Average each face with its neighbours, `rounds` times."""
    if rounds <= 0 or len(adjacency) == 0:
        return values

    counts = np.zeros(n_faces, dtype=np.float64)
    np.add.at(counts, adjacency[:, 0], 1.0)
    np.add.at(counts, adjacency[:, 1], 1.0)
    counts = np.maximum(counts, 1.0)

    out = values
    for _ in range(rounds):
        neighbour_sum = np.zeros(n_faces, dtype=np.float64)
        np.add.at(neighbour_sum, adjacency[:, 0], out[adjacency[:, 1]])
        np.add.at(neighbour_sum, adjacency[:, 1], out[adjacency[:, 0]])
        # Half own value, half neighbourhood mean: keeps the field from
        # diffusing away entirely over repeated rounds.
        out = 0.5 * out + 0.5 * (neighbour_sum / counts)
    return out
