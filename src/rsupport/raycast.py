"""Straight-down raycasting against a triangle mesh.

Every ray this project casts points along -Z: "what does this support land on?",
"is this vertex the lowest thing in its column?", "does this pillar pass through
the model?". That is a much smaller problem than general raycasting, so instead
of a BVH (and a native dependency for it) we project every triangle to XY once,
bucket the projections into a uniform grid, and answer a query with a vectorized
point-in-triangle test plus a plane solve.

The whole thing is numpy — no embree, no compiled extension, and it stays fast
because queries are answered in one batched pass rather than one ray at a time.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DownRay"]

_EPS = 1e-9


class DownRay:
    """Grid-accelerated downward raycaster.

    Args:
        mesh: anything with ``.triangles`` as an (F, 3, 3) array — a
            ``trimesh.Trimesh`` works directly.
        target_cells: rough number of grid cells to aim for. The grid is sized
            from the model's XY footprint so cells stay near triangle scale.
    """

    def __init__(self, mesh, target_cells: int = 128 * 128):
        tri = np.asarray(mesh.triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            raise ValueError(f"expected (F,3,3) triangles, got {tri.shape}")
        self.tri = tri
        self.n_faces = len(tri)

        self._a = tri[:, 0, :]
        self._b = tri[:, 1, :]
        self._c = tri[:, 2, :]

        lo = tri.reshape(-1, 3).min(axis=0)
        hi = tri.reshape(-1, 3).max(axis=0)
        self.bounds = np.array([lo, hi])

        span = np.maximum(hi[:2] - lo[:2], 1e-6)
        # Square-ish cells, count scaled to the footprint's aspect ratio.
        area_per_cell = (span[0] * span[1]) / max(target_cells, 1)
        cell = max(np.sqrt(area_per_cell), 1e-6)
        self.cell = cell
        self.origin = lo[:2] - cell * 0.5
        self.nx = int(np.ceil(span[0] / cell)) + 2
        self.ny = int(np.ceil(span[1] / cell)) + 2

        self._build_grid()

    # ------------------------------------------------------------------ build

    def _build_grid(self) -> None:
        """Bucket every triangle into the cells its XY bounding box covers."""
        tmin = self.tri[:, :, :2].min(axis=1)
        tmax = self.tri[:, :, :2].max(axis=1)

        i0 = np.clip(((tmin[:, 0] - self.origin[0]) / self.cell).astype(np.int64), 0, self.nx - 1)
        i1 = np.clip(((tmax[:, 0] - self.origin[0]) / self.cell).astype(np.int64), 0, self.nx - 1)
        j0 = np.clip(((tmin[:, 1] - self.origin[1]) / self.cell).astype(np.int64), 0, self.ny - 1)
        j1 = np.clip(((tmax[:, 1] - self.origin[1]) / self.cell).astype(np.int64), 0, self.ny - 1)

        w = i1 - i0 + 1
        h = j1 - j0 + 1
        counts = w * h
        total = int(counts.sum())

        faces = np.repeat(np.arange(self.n_faces, dtype=np.int64), counts)
        # Position of each entry within its own face's block of cells.
        block_start = np.cumsum(counts) - counts
        k = np.arange(total, dtype=np.int64) - np.repeat(block_start, counts)
        w_rep = np.repeat(w, counts)
        cx = np.repeat(i0, counts) + (k % w_rep)
        cy = np.repeat(j0, counts) + (k // w_rep)
        cells = cy * self.nx + cx

        order = np.argsort(cells, kind="stable")
        self._cell_sorted = cells[order]
        self._face_sorted = faces[order]

    # ------------------------------------------------------------------ query

    def _candidates(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (point_index, face_index) pairs worth testing, flattened."""
        n = len(xy)
        ix = ((xy[:, 0] - self.origin[0]) / self.cell).astype(np.int64)
        iy = ((xy[:, 1] - self.origin[1]) / self.cell).astype(np.int64)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        cells = np.where(inside, iy * self.nx + ix, -1)

        starts = np.searchsorted(self._cell_sorted, cells, side="left")
        ends = np.searchsorted(self._cell_sorted, cells, side="right")
        cnt = np.where(inside, ends - starts, 0)
        total = int(cnt.sum())
        if total == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        pidx = np.repeat(np.arange(n, dtype=np.int64), cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        fidx = self._face_sorted[np.repeat(starts, cnt) + offs]
        return pidx, fidx

    def column_hits(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All surface crossings in the vertical column through each point.

        Returns three parallel flat arrays ``(point_index, face_index, z)``,
        one entry per triangle the column passes through. Order is arbitrary.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        pidx, fidx = self._candidates(pts[:, :2])
        if len(pidx) == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty(0)

        p = pts[pidx, :2]
        a = self._a[fidx]
        b = self._b[fidx]
        c = self._c[fidx]

        # Barycentric coordinates of p within the XY projection of the triangle.
        v0 = c[:, :2] - a[:, :2]
        v1 = b[:, :2] - a[:, :2]
        v2 = p - a[:, :2]

        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)

        denom = d00 * d11 - d01 * d01
        # Degenerate in projection: an exactly vertical triangle contributes no
        # crossing, so drop it rather than dividing by zero.
        ok = np.abs(denom) > _EPS
        denom = np.where(ok, denom, 1.0)

        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        # Small tolerance so a point landing exactly on a shared edge still hits.
        tol = 1e-9
        hit = ok & (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)

        pidx, fidx = pidx[hit], fidx[hit]
        u, v = u[hit], v[hit]
        az, bz, cz = a[hit, 2], b[hit, 2], c[hit, 2]
        z = az + u * (cz - az) + v * (bz - az)
        return pidx, fidx, z

    def z_below(
        self, points: np.ndarray, eps: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Highest surface strictly below each point.

        Returns ``(z, face, hit)`` where ``hit`` is False for points with
        nothing underneath — those land on the build plate.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        pidx, fidx, z = self.column_hits(pts)

        best_z = np.full(len(pts), -np.inf)
        best_f = np.full(len(pts), -1, dtype=np.int64)

        keep = z < (pts[pidx, 2] - eps) if len(pidx) else np.zeros(0, dtype=bool)
        if keep.any():
            pidx, fidx, z = pidx[keep], fidx[keep], z[keep]
            np.maximum.at(best_z, pidx, z)
            # Recover which face produced each winning z.
            is_best = z >= best_z[pidx] - 1e-12
            best_f[pidx[is_best]] = fidx[is_best]

        hit = np.isfinite(best_z)
        best_z = np.where(hit, best_z, 0.0)
        return best_z, best_f, hit

    def z_above(
        self, points: np.ndarray, eps: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lowest surface strictly above each point. Mirror of :meth:`z_below`."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        pidx, fidx, z = self.column_hits(pts)

        best_z = np.full(len(pts), np.inf)
        best_f = np.full(len(pts), -1, dtype=np.int64)

        keep = z > (pts[pidx, 2] + eps) if len(pidx) else np.zeros(0, dtype=bool)
        if keep.any():
            pidx, fidx, z = pidx[keep], fidx[keep], z[keep]
            np.minimum.at(best_z, pidx, z)
            is_best = z <= best_z[pidx] + 1e-12
            best_f[pidx[is_best]] = fidx[is_best]

        hit = np.isfinite(best_z)
        best_z = np.where(hit, best_z, 0.0)
        return best_z, best_f, hit

    def inside(self, points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Parity test: is each point inside the (closed) mesh?

        Counts crossings above the point. Odd means inside. Unreliable on open
        or self-intersecting meshes, which is fine — it is only used as a
        cheap veto when routing pillars.

        Crossings at the same height are merged first. A column passing exactly
        along a shared edge — the diagonal across a box's top face, say — hits
        both adjacent triangles and would otherwise count one surface crossing
        twice, flipping the parity and reporting solid as empty.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        pidx, _, z = self.column_hits(pts)
        counts = np.zeros(len(pts), dtype=np.int64)
        if len(pidx):
            above = z > (pts[pidx, 2] + eps)
            pidx, z = pidx[above], z[above]
            if len(pidx):
                tol = max(float(np.ptp(self.bounds[:, 2])), 1.0) * 1e-9
                order = np.lexsort((z, pidx))
                pidx, z = pidx[order], z[order]
                first = np.empty(len(z), dtype=bool)
                first[0] = True
                first[1:] = (pidx[1:] != pidx[:-1]) | (np.diff(z) > tol)
                np.add.at(counts, pidx[first], 1)
        return (counts % 2) == 1

    def segment_blocked(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        samples: int = 12,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Does each segment pass through solid model?

        Sampled rather than exact: fine for deciding whether a pillar needs to
        be tilted, and it costs one batched query instead of a swept solid test.
        """
        starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
        ends = np.atleast_2d(np.asarray(ends, dtype=np.float64))
        n = len(starts)
        if n == 0:
            return np.zeros(0, dtype=bool)

        t = np.linspace(0.0, 1.0, samples)[1:-1]  # skip endpoints: they touch by design
        pts = starts[:, None, :] + (ends - starts)[:, None, :] * t[None, :, None]
        flags = self.inside(pts.reshape(-1, 3), eps=eps).reshape(n, -1)
        return flags.any(axis=1)
