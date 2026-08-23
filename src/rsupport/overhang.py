"""Which faces are overhanging, and which cross-sections start in mid-air.

Two separate questions, both answered here because both are "what does gravity
object to?":

* **Overhang faces** — a face whose normal points far enough downward that the
  extruder would be laying plastic over nothing. Cheap: one dot product.
* **Islands** — a slice contour that has no material anywhere beneath it. The
  outstretched tip of a sword, the lower hem of a cape, a floating gem. These
  are not "steep", they are *absent*: the first layer of an island prints into
  thin air, so every island is a mandatory support regardless of angle.

Nothing here builds geometry or makes placement decisions; that is
:mod:`rsupport.sampling`'s job.

Angle convention
----------------
``severity`` is ``n · (-Z)``, the cosine of the angle between a face normal and
straight down. ``overhang_mask(mesh, a)`` flags faces with
``severity >= cos(a)``, i.e. faces whose normal is within ``a`` degrees of
straight down. At the default 45 deg this is exactly the familiar slicer rule.
Away from 45 the sense is *inverted* relative to a slicer's "overhang angle"
slider: here a **larger** angle flags **more** faces. That matches this repo's
own presets (``mini_0.2_dense`` raises the angle to 55 to hold more) and
matches PLAN.md's wording, "n · (-Z) beyond the overhang angle".

No spatial index is used anywhere in this module: ``rtree``/``embree`` are not
installed (see CLAUDE.md), so ``mesh.section`` and friends that need them are
avoided in favour of ``mesh_multiplane`` plus a hand-rolled loop walker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize, unary_union
from trimesh.intersections import mesh_multiplane

__all__ = [
    "overhang_severity",
    "overhang_mask",
    "Island",
    "find_islands",
    "slice_polygons",
]


# ---------------------------------------------------------------- face angles


def overhang_severity(mesh) -> np.ndarray:
    """How badly each face points downward.

    Args:
        mesh: a ``trimesh.Trimesh``.

    Returns:
        (F,) float in 0..1. ``0`` for a vertical wall or anything facing
        upward, ``1`` for a face pointing straight down. Equal to the cosine of
        the angle between the face normal and -Z, clipped at zero.
    """
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"expected (F,3) face normals, got {normals.shape}")
    sev = np.nan_to_num(-normals[:, 2], nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sev, 0.0, 1.0)


def overhang_mask(mesh, angle_deg: float) -> np.ndarray:
    """Faces steep enough to need holding up.

    Args:
        mesh: a ``trimesh.Trimesh``.
        angle_deg: faces whose normal lies within this many degrees of straight
            down are flagged. See the module docstring on the sign convention.

    Returns:
        (F,) bool.
    """
    angle = float(np.clip(angle_deg, 0.0, 90.0))
    threshold = float(np.cos(np.radians(angle)))
    # Tolerance so a face sitting exactly on the threshold (a 45 deg chamfer
    # modelled exactly) is flagged rather than lost to float noise.
    return overhang_severity(mesh) >= threshold - 1e-9


# ------------------------------------------------------------------- slicing


def _snap_digits(mesh) -> int:
    """Decimal places for welding slice vertices, scaled to the model."""
    scale = float(getattr(mesh, "scale", 1.0)) or 1.0
    # Adjacent triangles compute their shared edge crossing in opposite
    # directions, so the two copies differ by an ULP or two - many orders of
    # magnitude below this tolerance, which stays far away from real detail.
    return int(np.clip(round(-np.log10(scale * 1e-9)), 3, 12))


def _walk_rings(vertices: np.ndarray, edges: np.ndarray) -> list[np.ndarray] | None:
    """Chain a soup of 2D segments into closed loops.

    A clean cross-section of a watertight mesh is a set of simple closed loops,
    so every vertex has exactly two incident edges and the loops can be walked
    in linear time. That is several times faster than handing the segments to
    shapely's ``polygonize``, which has to node an arbitrary planar graph.

    Returns:
        A list of index arrays, one per loop, or ``None`` if the section is not
        made of degree-2 vertices (non-manifold or self-intersecting input) -
        the caller should fall back to shapely.
    """
    n_vert = len(vertices)
    degree = np.bincount(edges.reshape(-1), minlength=n_vert)
    used = degree > 0
    if not used.any():
        return []
    if degree.max() != 2 or degree[used].min() != 2:
        return None

    neighbour = np.full((n_vert, 2), -1, dtype=np.intp)
    slot = np.zeros(n_vert, dtype=np.intp)
    for a, b in edges:
        neighbour[a, slot[a]] = b
        slot[a] += 1
        neighbour[b, slot[b]] = a
        slot[b] += 1

    seen = np.zeros(n_vert, dtype=bool)
    rings: list[np.ndarray] = []
    for start in np.nonzero(used)[0]:
        if seen[start]:
            continue
        loop = [int(start)]
        seen[start] = True
        prev, cur = int(start), int(neighbour[start, 0])
        while cur != -1 and cur != start:
            if seen[cur]:  # ran into another loop: input was not clean
                return None
            seen[cur] = True
            loop.append(cur)
            n0, n1 = neighbour[cur]
            nxt = int(n1) if int(n0) == prev else int(n0)
            prev, cur = cur, nxt
        if cur != start:  # open chain
            return None
        if len(loop) >= 3:
            rings.append(np.asarray(loop, dtype=np.intp))
    return rings


def _nest(rings: list[Polygon]) -> list[Polygon]:
    """Turn a flat list of simple rings into polygons with holes.

    A ring is a hole when it sits at odd depth in the containment tree. Only
    *full* containment counts: two rings that merely overlap (which happens all
    the time on concatenated, self-intersecting STLs) are both kept as solid
    regions rather than one being mistaken for a hole in the other.
    """
    n = len(rings)
    if n <= 1:
        return rings

    areas = np.array([p.area for p in rings], dtype=np.float64)
    geoms = np.empty(n, dtype=object)
    geoms[:] = rings

    parent = np.full(n, -1, dtype=np.intp)
    for i in range(n):
        enclosing = shapely.contains(geoms, rings[i])
        enclosing &= areas > areas[i] * (1.0 + 1e-12)
        cand = np.nonzero(enclosing)[0]
        if len(cand):
            parent[i] = cand[np.argmin(areas[cand])]

    depth = np.zeros(n, dtype=np.intp)
    for i in np.argsort(-areas):  # parents are larger, so always resolved first
        depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1

    out: list[Polygon] = []
    for i in range(n):
        if depth[i] % 2:
            continue
        holes = [
            rings[j].exterior.coords
            for j in range(n)
            if parent[j] == i and depth[j] % 2 == 1
        ]
        poly = Polygon(rings[i].exterior.coords, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "Polygon" and not poly.is_empty:
            out.append(poly)
        elif poly.geom_type == "MultiPolygon":
            out.extend(g for g in poly.geoms if not g.is_empty)
    return out


def _section_polygons(segments: np.ndarray, digits: int) -> list[Polygon]:
    """One plane's ``(n, 2, 2)`` segment array -> filled shapely polygons."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.size == 0:
        return []

    points = segments.reshape(-1, 2)
    unique, inverse = trimesh.grouping.unique_rows(points, digits=digits)
    verts = points[unique]
    edges = inverse.reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return []
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    loops = _walk_rings(verts, edges)
    if loops is None:
        # Messy section (non-manifold, or two shells crossing exactly on the
        # plane). Let shapely sort it out; its faces are used as-is, which can
        # fill an internal hole but never invents or loses an outline.
        lines = MultiLineString([[tuple(verts[a]), tuple(verts[b])] for a, b in edges])
        return [g for g in polygonize(unary_union(lines)) if g.area > 0.0]

    rings: list[Polygon] = []
    for loop in loops:
        poly = Polygon(verts[loop])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "Polygon" and poly.area > 0.0:
            rings.append(poly)
        elif poly.geom_type == "MultiPolygon":
            rings.extend(g for g in poly.geoms if g.area > 0.0)
    return _nest(rings)


def slice_polygons(mesh, heights) -> list[list[Polygon]]:
    """Cross-section the mesh at a set of absolute Z heights.

    Args:
        mesh: a ``trimesh.Trimesh``.
        heights: (m,) absolute Z values.

    Returns:
        A list of m lists of shapely polygons, in XY. Polygons carry their
        holes. Uses ``trimesh.intersections.mesh_multiplane``, which caches the
        vertex/plane dot product across all m planes - so one call with m
        heights is far cheaper than m calls with one.
    """
    heights = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    if len(heights) == 0:
        return []
    lines, _, _ = mesh_multiplane(
        mesh, np.zeros(3), np.array([0.0, 0.0, 1.0]), heights
    )
    digits = _snap_digits(mesh)
    return [_section_polygons(seg, digits) for seg in lines]


# ------------------------------------------------------------------- islands


@dataclass
class Island:
    """A cross-section that begins with nothing underneath it.

    Attributes:
        z: height at which the island starts. Accurate to ``layer_height`` when
            ``find_islands`` is left to refine (the default); otherwise
            accurate to the slicing step.
        centroid: (3,) a point inside the island's cross-section, at height
            ``z``. Not projected onto the surface - :mod:`rsupport.sampling`
            does that with a raycast.
        area: mm^2 of the cross-section that triggered the detection.
        polygon: shapely Polygon in XY, at the slice where the island was
            detected.
    """

    z: float
    centroid: np.ndarray
    area: float
    polygon: object


def _interior_point(poly: Polygon) -> np.ndarray:
    """Centroid if it is inside, otherwise a guaranteed-interior point.

    A crescent-shaped cape hem has its centroid out in the open air, which
    would put the support pillar next to the model instead of under it.
    """
    c = poly.centroid
    if poly.contains(c):
        return np.array([c.x, c.y], dtype=np.float64)
    r = poly.representative_point()
    return np.array([r.x, r.y], dtype=np.float64)


def _touches_at(mesh, poly: Polygon, z: float) -> bool:
    """Is there any cross-section at height ``z`` touching ``poly``?"""
    return any(q.intersects(poly) for q in slice_polygons(mesh, [z])[0])


def _refine_start_z(mesh, poly: Polygon, z_lo: float, z_hi: float, tol: float) -> float:
    """Bisect for the height where ``poly``'s feature first appears.

    ``z_lo`` is known to have nothing under the island and ``z_hi`` is the
    slice that detected it. Each step costs one single-plane slice, and there
    are only ``log2(step / layer_height)`` of them per island - two, at the
    default 4x step - so this buys back the resolution the coarse pass gave up
    at a cost proportional to the (small) island count rather than the model
    height.
    """
    lo, hi = float(z_lo), float(z_hi)
    guard = 0
    while hi - lo > tol and guard < 24:
        guard += 1
        mid = 0.5 * (lo + hi)
        if _touches_at(mesh, poly, mid):
            hi = mid
        else:
            lo = mid
    return hi


def find_islands(
    mesh,
    layer_height: float,
    min_area: float = 0.05,
    *,
    step: float | None = None,
    refine: bool = True,
) -> list[Island]:
    """Cross-sections that start in mid-air.

    An outstretched sword tip, the lower hem of a cape, a gem floating clear of
    the body. Each one is a mandatory support: its first printed layer has no
    material beneath it at all, so no amount of overhang-angle sampling nearby
    will save it.

    The test is per layer: slice, then for every polygon ask whether it
    intersects *any* polygon in the layer below. If not, it started in air. The
    bottom-most layer is skipped, since it rests on the build plate.

    Args:
        mesh: a ``trimesh.Trimesh``, already in its print orientation.
        layer_height: the print layer height. Sets the refinement resolution,
            and the default slicing step.
        min_area: mm^2. Cross-sections smaller than this are slivers from a
            plane grazing a curved surface, not real islands.
        step: Z distance between slice planes. Defaults to ``layer_height``.
            **Cost**: slicing dominates stage 2. Measured on a 41k-face, 41 mm
            miniature, a full 0.08 mm pass is 512 planes at ~1.5 s; a 4x step
            is 128 planes at ~0.4 s. :mod:`rsupport.sampling` therefore calls
            this with ``step = 4 * layer_height`` and lets ``refine`` recover
            the Z precision. The tradeoff of a coarse step is that an island
            shorter in Z than one step can fall between two planes and be
            missed entirely; at 0.32 mm that only loses features thinner than a
            few layers, which cannot be supported by a 0.3 mm tip anyway.
        refine: bisect each detected island back down to ``layer_height``
            precision. Free when ``step == layer_height``.

    Returns:
        Islands, ordered bottom-up.
    """
    layer_height = float(layer_height)
    if layer_height <= 0:
        raise ValueError("layer_height must be positive")
    step = layer_height if step is None else float(step)
    if step <= 0:
        raise ValueError("step must be positive")

    lo, hi = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    if hi - lo <= step:
        return []

    heights = np.arange(lo + step * 0.5, hi, step)
    if len(heights) < 2:
        return []

    layers = slice_polygons(mesh, heights)

    islands: list[Island] = []
    for i in range(1, len(layers)):
        below = layers[i - 1]
        if not layers[i]:
            continue
        tree = shapely.STRtree(below) if below else None
        for poly in layers[i]:
            if poly.area < min_area:
                continue
            if tree is not None and len(tree.query(poly, predicate="intersects")):
                continue
            z = float(heights[i])
            if refine and step > layer_height:
                z = _refine_start_z(mesh, poly, float(heights[i - 1]), z, layer_height)
            xy = _interior_point(poly)
            islands.append(
                Island(
                    z=z,
                    centroid=np.array([xy[0], xy[1], z], dtype=np.float64),
                    area=float(poly.area),
                    polygon=poly,
                )
            )

    islands.sort(key=lambda isl: isl.z)
    return islands
