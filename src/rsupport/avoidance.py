"""Where a support may stand: the bottom-up reachability sweep.

Every strut the generator places — a shaft dropping to the plate, an arm
reaching up to a contact, a cross-link tying two shafts together — has to
answer two questions, and they look like separate problems but are not:

1. **Does this pass through the model?**
2. **Can this reach the build plate, or must it rest on the model?**

Whether the plate is reachable is a question about the *whole column of layers
below* a position, so a local "is there something directly beneath me" test
cannot answer it. It is precomputed instead, bottom-up::

    solid[i]    = model cross-section over [z_i, z_i+1]
    free[r][i]  = everywhere a strut of radius r may stand on layer i
    reach[r][0] = free[r][0]
    reach[r][i] = free[r][i] ∩ dilate(reach[r][i-1], max_move)

``reach[r][i]`` is exactly the set of positions on layer ``i`` from which a
strut of radius ``r`` can still get to the plate without ever entering the
model, given that it may travel at most ``max_move`` sideways per layer. Both
guarantees then fall out of the descent for free:

* a position already inside ``reach`` on the layer below drops straight down;
* a position outside it moves to the nearest point of ``reach``, which is what
  makes a shaft step around an arm rather than stop at it;
* a shaft can only ever land on the model when ``reach`` is genuinely
  unreachable — not as a preference, but because there is nowhere else to go.

And since every position is always inside ``free`` for its own radius, no
support can intersect the model at all. Both guarantees are structural rather
than a check-and-reject after the fact.

The sweep is deliberately independent of what gets built on top of it. It knows
about radii and clearances, not about tips, arms or cross-links.

The approach is CuraEngine's tree-support avoidance (Ghostkeeper's original
CuraEngine PR #655, later rewritten around influence areas), itself descended
from Vanek et al., *Clever Support* (2014). Collision is sampled at a handful of
radii for the same reason Cura samples: rebuilding the field for every distinct
strut width would be ruinous, and a few buckets are visually indistinguishable.

Raster, not polygons
--------------------

The sets above are held as **boolean masks on a fixed XY lattice**, not as
shapely polygons. The sweep used to be a chain of
``buffer`` → ``simplify`` → ``difference`` → ``intersection``, once per layer
per radius, and that chain was both the slowest part of a build and the
fragile one: repeatedly buffering an already-simplified ring is the classic way
to hand GEOS a near-degenerate edge, and GEOS 3.12 answers with
``TopologyException: depth mismatch``. Under Emscripten that exception is not
catchable from Python — it unwinds straight past the interpreter — so the
polygon sweep cannot run in a browser at all.

On a lattice the whole thing collapses into one Euclidean distance transform
per layer:

* ``dist[i]`` — millimetres from each cell to the nearest solid cell — is
  computed **once** and answers every radius at once, because ``free[r][i]`` is
  just ``dist[i] >= r + clearance``. The old code paid for a separate buffer per
  radius per layer;
* the dilation in the sweep is another distance transform; and
* ``room`` — "how fat may a foot standing here be" — becomes a lookup into
  ``dist`` rather than an exact point-to-polygon distance query.

Slicing and polygonisation are untouched: :func:`rsupport.overhang.slice_polygons`
is proven and was never part of the fragile chain. GEOS is still used, but only
for ``contains_xy`` — point-in-polygon location, not overlay — which is the
robust half of the library.

**Every quantisation here rounds toward refusing a position, never toward
allowing one.** The solid is over-marked by a cell, distances are measured from
that over-marked set, thresholds carry a half-cell margin, and the dilation
radius is rounded down. So the rasterised ``free`` is always a *subset* of the
true one: the field may be a shade pessimistic — a support declined where a hair
more room existed — but it can never route a strut into the model. That
asymmetry is what lets the self-printability invariant survive the change.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import box

from .overhang import slice_polygons
from .types import SupportParams

__all__ = ["AvoidanceField", "Region", "strut_lean"]

#: How many radii the collision field is sampled at.
_RADIUS_BUCKETS = 6

#: Lattice pitch as a fraction of the nozzle. Everything the field has to
#: resolve — the clearance, a strut radius, one layer's sideways travel — is a
#: multiple of the nozzle, so the cell that resolves them is too. At the default
#: 0.4 mm nozzle this is a 0.15 mm cell, which puts ~7 cells across the 1 mm
#: blade in ``tests/test_pipeline.py``.
_CELL_RATIO = 0.375

#: Ceiling on lattice cells per layer. A tall model with a wide footprint could
#: otherwise ask for a grid big enough to matter in a browser tab; past this the
#: cell is coarsened instead, which costs room and never safety.
_MAX_CELLS = 1_200_000


class _Lattice:
    """The fixed XY grid every layer's mask is sampled on."""

    __slots__ = ("cell", "origin", "nx", "ny", "_xs", "_ys")

    def __init__(self, lo_xy, hi_xy, cell: float):
        self.cell = float(cell)
        self.origin = np.asarray(lo_xy, dtype=np.float64)
        span = np.maximum(np.asarray(hi_xy, dtype=np.float64) - self.origin, self.cell)
        self.nx = int(math.ceil(span[0] / self.cell)) + 1
        self.ny = int(math.ceil(span[1] / self.cell)) + 1
        self._xs = self.origin[0] + np.arange(self.nx, dtype=np.float64) * self.cell
        self._ys = self.origin[1] + np.arange(self.ny, dtype=np.float64) * self.cell

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def centres(self, sl_y=slice(None), sl_x=slice(None)):
        """Cell-centre coordinates over a window, as (X, Y) meshes."""
        return np.meshgrid(self._xs[sl_x], self._ys[sl_y], indexing="xy")

    def index(self, x, y):
        """Nearest cell index for each coordinate, plus an in-bounds mask."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        ix = np.rint((x - self.origin[0]) / self.cell).astype(np.intp)
        iy = np.rint((y - self.origin[1]) / self.cell).astype(np.intp)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1), inside

    def window(self, bounds):
        """Index slices covering a (minx, miny, maxx, maxy) box, clipped.

        Padded by a cell on each side so a polygon edge landing between centres
        still has its neighbours tested.
        """
        minx, miny, maxx, maxy = bounds
        x0 = max(int(math.floor((minx - self.origin[0]) / self.cell)) - 1, 0)
        y0 = max(int(math.floor((miny - self.origin[1]) / self.cell)) - 1, 0)
        x1 = min(int(math.ceil((maxx - self.origin[0]) / self.cell)) + 2, self.nx)
        y1 = min(int(math.ceil((maxy - self.origin[1]) / self.cell)) + 2, self.ny)
        if x1 <= x0 or y1 <= y0:
            return None
        return slice(y0, y1), slice(x0, x1)

    def position(self, ix: int, iy: int) -> np.ndarray:
        return np.array(
            [self.origin[0] + ix * self.cell, self.origin[1] + iy * self.cell],
            dtype=np.float64,
        )


class Region:
    """A standable set on one layer: a boolean mask plus the lattice under it.

    Everywhere off the lattice is standable. The grid is padded past the model
    by more than any strut's radius plus its clearance, so a cell out there is
    beyond anything the model could block — which also means the border ring is
    always set, and a nearest-point query therefore always has an answer.
    """

    __slots__ = ("mask", "lattice", "_indices")

    def __init__(self, mask: np.ndarray, lattice: _Lattice):
        self.mask = mask
        self.lattice = lattice
        self._indices = None

    @property
    def is_empty(self) -> bool:
        """Never true — everywhere off the lattice is standable, so there is
        always somewhere. Kept because callers guard on it."""
        return False

    @property
    def area(self) -> float:
        """Square millimetres inside the lattice. Off-lattice ground is
        unbounded and deliberately not counted: this exists to compare two
        regions on the same lattice, not to measure the world."""
        return float(self.mask.sum()) * self.lattice.cell**2

    def without(self, other: "Region") -> "Region":
        """This region minus `other` — the raster's set difference."""
        return Region(self.mask & ~other.mask, self.lattice)

    def contains_xy(self, x, y) -> np.ndarray:
        ix, iy, inside = self.lattice.index(x, y)
        return np.where(inside, self.mask[iy, ix], True)

    def contains(self, xy) -> bool:
        return bool(self.contains_xy(float(xy[0]), float(xy[1]))[0])

    def nearest(self, xy) -> np.ndarray:
        """Closest standable position to `xy`, or `xy` itself if already so."""
        xy = np.asarray(xy, dtype=np.float64)
        ix, iy, inside = self.lattice.index(xy[0], xy[1])
        if not bool(inside[0]) or bool(self.mask[iy[0], ix[0]]):
            return xy
        if self._indices is None:
            # The feature transform is only ever paid for on a layer where a
            # strut actually had to dodge, which is the rare case.
            self._indices = ndimage.distance_transform_edt(
                ~self.mask, return_distances=False, return_indices=True
            )
        fy, fx = self._indices
        return self.lattice.position(int(fx[iy[0], ix[0]]), int(fy[iy[0], ix[0]]))


class AvoidanceField:
    """Per-layer, per-radius maps of where a support may stand.

    Built once per model and reused across every strut, which is what keeps the
    whole thing affordable — the expensive part is slicing, and
    :func:`rsupport.overhang.slice_polygons` batches every height into a single
    pass.
    """

    def __init__(self, mesh, params: SupportParams, top_z: float | None = None):
        self.params = params
        self.pitch = max(float(params.collision_pitch), 1e-3)
        self.max_move = self.pitch * math.tan(math.radians(strut_lean(params)))

        lo, hi = mesh.bounds
        ceiling = float(hi[2] if top_z is None else max(top_z, lo[2] + self.pitch))
        self.n_layers = max(2, int(math.ceil(ceiling / self.pitch)) + 2)
        self.heights = np.arange(self.n_layers, dtype=np.float64) * self.pitch

        self.radii = self._radius_buckets()

        # The lattice only has to cover the model plus everything a strut could
        # be pushed out to. Past that ring nothing is ever blocked, so the old
        # lean-sized bed — tens of millimetres of empty margin in every
        # direction — would be rasterising air.
        margin = 2.0 * (float(self.radii[-1]) + params.xy_clearance) + 4.0 * self.max_move
        self.bed = box(lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin)

        cell = self._cell_size(margin, lo, hi)
        self._lattice = _Lattice(
            (lo[0] - margin, lo[1] - margin), (hi[0] + margin, hi[1] + margin), cell
        )

        # Half a cell diagonal: the worst a query point can sit from the centre
        # that answered for it. Every threshold carries it, so quantisation
        # costs room rather than safety.
        self._slack = self._lattice.cell * math.sqrt(2.0) * 0.5

        self._reach_cache: dict[int, np.ndarray] = {}
        self._build(mesh)

    # -- construction ------------------------------------------------------ #

    def _radius_buckets(self) -> np.ndarray:
        p = self.params
        lo = p.tip_diameter * 0.5
        hi = max(p.max_strut_diameter * 0.5, lo * 1.5)
        return np.geomspace(lo, hi, _RADIUS_BUCKETS)

    def _cell_size(self, margin: float, lo, hi) -> float:
        cell = max(self.params.nozzle_diameter * _CELL_RATIO, 1e-3)
        # The sweep dilates by one layer's travel, so the cell has to divide it
        # a few times over or rounding that radius down would stall the sweep.
        cell = min(cell, self.max_move / 3.0)

        span_x = float(hi[0] - lo[0]) + 2.0 * margin
        span_y = float(hi[1] - lo[1]) + 2.0 * margin
        n = (span_x / cell + 1.0) * (span_y / cell + 1.0)
        if n > _MAX_CELLS:
            cell *= math.sqrt(n / _MAX_CELLS)
        return cell

    def _build(self, mesh) -> None:
        lat = self._lattice
        # Slice just above each layer boundary: a slice exactly at z=0 catches
        # the model's base plane edge-on and comes back degenerate.
        sample = np.clip(self.heights + self.pitch * 0.02, 1e-4, None)
        sections = slice_polygons(mesh, sample)

        # Millimetres from every cell to the nearest solid cell, per layer. One
        # transform answers every radius bucket, which is the whole reason this
        # is cheaper than the polygon sweep it replaced.
        self._dist = np.empty((self.n_layers, lat.ny, lat.nx), dtype=np.float32)
        for i in range(self.n_layers):
            here = sections[i] if i < len(sections) else []
            above = sections[i + 1] if i + 1 < len(sections) else []
            merged = [g for g in (*here, *above) if g is not None and not g.is_empty]
            solid = self._rasterise(merged)
            if solid is None:
                self._dist[i] = np.float32(np.inf)
            else:
                self._dist[i] = ndimage.distance_transform_edt(
                    ~solid, sampling=lat.cell
                ).astype(np.float32)

    def _rasterise(self, polygons) -> np.ndarray | None:
        """Mark every cell the cross-section touches.

        Two passes, because neither alone is safe. Testing cell *centres* fills
        the interior but loses any feature thinner than a cell — and losing
        solid is the one error that matters, since it would hand a strut a route
        straight through a blade. Walking the *outline* catches those: however
        thin a feature is, its boundary still crosses cells, so every sliver
        gets marked whatever its width.

        Dilating the centre hits instead would also catch them, but at the price
        of a full cell of phantom clearance everywhere — which the boundary pass
        does not cost, since a boundary cell straddles the real surface anyway
        and sits inside the half-cell margin ``_slack`` already carries.

        Returns None when the layer is empty, which lets the caller skip the
        transform rather than measure distances to nothing.
        """
        lat = self._lattice
        solid = np.zeros(lat.shape, dtype=bool)
        hit_any = False
        for poly in polygons:
            win = lat.window(poly.bounds)
            if win is None:
                continue
            sy, sx = win
            gx, gy = lat.centres(sy, sx)
            hit = shapely.contains_xy(poly, gx.ravel(), gy.ravel())
            if hit.any():
                solid[sy, sx] |= hit.reshape(gx.shape)
                hit_any = True
            hit_any |= self._mark_outline(poly, solid)
        return solid if hit_any else None

    def _mark_outline(self, poly, solid: np.ndarray) -> bool:
        """Stamp every cell the polygon's rings pass through.

        The rings are walked at half-cell steps, which is fine enough that
        consecutive samples cannot skip a cell.
        """
        lat = self._lattice
        marked = False
        for ring in (poly.exterior, *poly.interiors):
            xy = np.asarray(ring.coords, dtype=np.float64)
            if len(xy) < 2:
                continue
            seg = xy[1:] - xy[:-1]
            steps = np.maximum(
                np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / (lat.cell * 0.5)), 1.0
            ).astype(np.intp)

            # One densified point per step, without a Python loop over segments.
            total = int(steps.sum())
            which = np.repeat(np.arange(len(steps), dtype=np.intp), steps)
            offset = np.repeat(np.cumsum(steps) - steps, steps)
            t = (np.arange(total, dtype=np.float64) - offset) / np.repeat(steps, steps)
            pts = xy[:-1][which] + seg[which] * t[:, None]

            ix, iy, inside = lat.index(pts[:, 0], pts[:, 1])
            if inside.any():
                solid[iy[inside], ix[inside]] = True
                marked = True
        return marked

    def _free_stack(self, bucket: int) -> np.ndarray:
        return self._dist >= self._need(bucket)

    def _need(self, bucket: int) -> float:
        return float(self.radii[bucket]) + self.params.xy_clearance + self._slack

    def _reach_stack(self, bucket: int) -> np.ndarray:
        """The reachability sweep for one radius, computed on first use.

        Most builds only ever ask about a couple of radii — a tip and a shaft —
        so sweeping all six up front would be paying for four of them twice
        over: once in time, once in the memory to hold them.
        """
        cached = self._reach_cache.get(bucket)
        if cached is not None:
            return cached

        free = self._free_stack(bucket)
        # The transform measures exact Euclidean distance in cells, so this is
        # one layer's travel converted to the same units — neither rounded up
        # (which would claim more lean than a strut has) nor padded down.
        span_cells = self.max_move / self._lattice.cell

        reach = np.empty_like(free)
        reach[0] = free[0]
        for i in range(1, self.n_layers):
            prev = reach[i - 1]
            if not prev.any():
                reach[i] = False
                continue
            grown = ndimage.distance_transform_edt(~prev) <= span_cells
            reach[i] = free[i] & grown

        self._reach_cache[bucket] = reach
        return reach

    # -- lookup ------------------------------------------------------------ #

    def bucket(self, radius: float) -> int:
        """Index of the first sampled radius at least as fat as `radius`.

        Rounding up rather than to nearest: a strut judged against a radius
        smaller than its own could be routed into the model.
        """
        idx = int(np.searchsorted(self.radii, float(radius), side="left"))
        return min(idx, len(self.radii) - 1)

    def layer_of(self, z: float) -> int:
        return int(np.clip(round(float(z) / self.pitch), 0, self.n_layers - 1))

    def free(self, bucket: int, layer: int) -> Region:
        layer = int(np.clip(layer, 0, self.n_layers - 1))
        return Region(self._dist[layer] >= self._need(bucket), self._lattice)

    def reach(self, bucket: int, layer: int) -> Region:
        layer = int(np.clip(layer, 0, self.n_layers - 1))
        return Region(self._reach_stack(bucket)[layer], self._lattice)

    def standable(self, bucket: int, layer: int, to_plate: bool) -> Region:
        return self.reach(bucket, layer) if to_plate else self.free(bucket, layer)

    def room(self, xy, layer: int) -> float:
        """Fattest radius anything standing at `xy` on this layer could have.

        The bucketed ``free`` maps answer the question the other way round —
        given a radius, where may it stand — which is the right shape for
        routing a strut of known width. A base disc is the opposite case: it is
        placed wherever the shaft came down, and the question is how much of it
        fits. Buckets are far too coarse to answer that (they stop at
        ``max_strut_diameter``, and a foot is several times fatter), so this
        reads the distance to the model off the layer's transform and takes the
        clearance off it.
        """
        layer = int(np.clip(layer, 0, self.n_layers - 1))
        ix, iy, inside = self._lattice.index(float(xy[0]), float(xy[1]))
        if not bool(inside[0]):
            return float("inf")
        gap = float(self._dist[layer, iy[0], ix[0]])
        if not math.isfinite(gap):
            return float("inf")
        return max(0.0, gap - self.params.xy_clearance - self._slack)

    def contains(self, region: Region | None, xy) -> bool:
        if region is None or region.is_empty:
            return False
        return region.contains(xy)

    def nearest(self, region: Region, xy) -> np.ndarray:
        """Closest position in `region` to `xy`; `xy` itself if already inside."""
        return region.nearest(xy)


def strut_lean(params: SupportParams) -> float:
    """Lean allowance off vertical, clamped below the printable limit.

    A strut leaning by `a` degrees overhangs by exactly `a`, so this is the
    entire printability budget for anything that does not run straight up.
    """
    return float(np.clip(params.strut_lean_deg, 1.0, params.printable_overhang_deg - 2.0))
