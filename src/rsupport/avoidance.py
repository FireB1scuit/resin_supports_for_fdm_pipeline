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
    reach[r][i] = free[r][i] ∩ reach[r][i-1].buffer(max_move)

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
radii for the same reason Cura samples: buffering every layer for every distinct
radius would be ruinous, and a few buckets are visually indistinguishable.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from .overhang import slice_polygons
from .types import SupportParams

__all__ = ["AvoidanceField", "strut_lean"]

#: How many radii the collision field is sampled at.
_RADIUS_BUCKETS = 6


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

        # Struts fan outward as they descend, so the working area has to be
        # wider than the model by everything they could travel.
        span = float(ceiling) * math.tan(math.radians(strut_lean(params)))
        margin = float(np.clip(span + 5.0, 10.0, 80.0))
        self.bed = box(lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin)

        self.radii = self._radius_buckets()
        self._build(mesh)

    # -- construction ------------------------------------------------------ #

    def _radius_buckets(self) -> np.ndarray:
        p = self.params
        lo = p.tip_diameter * 0.5
        hi = max(p.max_strut_diameter * 0.5, lo * 1.5)
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
                # then a position judged reachable would in fact be in the model.
                grown = reach[-1].buffer(self.max_move, quad_segs=4).simplify(tol)
                reach.append(free[i].intersection(grown))

            self._free.append(free)
            self._reach.append(reach)

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


def strut_lean(params: SupportParams) -> float:
    """Lean allowance off vertical, clamped below the printable limit.

    A strut leaning by `a` degrees overhangs by exactly `a`, so this is the
    entire printability budget for anything that does not run straight up.
    """
    return float(np.clip(params.strut_lean_deg, 1.0, params.printable_overhang_deg - 2.0))
