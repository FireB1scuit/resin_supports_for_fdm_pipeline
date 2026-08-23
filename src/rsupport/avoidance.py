"""Which collision backend answers "where may a support stand".

There are two, and they compute the same thing from the same slices:

* :mod:`rsupport.avoidance_polygon` — shapely regions, the original. Faster,
  and the one every desktop and container build uses.
* :mod:`rsupport.avoidance_raster` — boolean masks on a lattice. Slower, a
  shade more conservative, and the only one that runs in a browser.

Two exist because the polygon sweep **cannot run under Emscripten at all**. Its
per-layer chain of ``buffer`` → ``difference`` → ``intersection`` trips a GEOS
overlay bug on the sculpts we test, and the GEOS that Pyodide ships is 3.12
against the 3.13 a desktop shapely brings. That would merely be a crash, except
Emscripten turns the C++ ``TopologyException`` into an unwind that passes
straight through the interpreter: ``except BaseException`` does not catch it,
and the whole Python runtime dies with it. There is no degrading gracefully and
no retrying, so the browser needs a backend that never calls those operations.

Measured, on the sample mini, against the polygon sweep:

============  ==========================  ========================
preset        drops (polygon → raster)    build (polygon → raster)
============  ==========================  ========================
mini_0.2      6 → 7                       0.65s → 2.32s
mini_0.25     5 → 4                       0.52s → 1.29s
mini_0.4      3 → 3                       0.47s → 0.68s
dense         26 → 29                     1.13s → 4.52s
lifted        5 → 2                       0.75s → 2.53s
templar       10 → 10                     1.02s → 1.45s
============  ==========================  ========================

Violations are 0 under both: every difference lands on *dropped contacts*, the
safe side, because the lattice rounds toward refusing a position. The gap is
quantisation and nothing else — at a 0.0375 mm cell the raster field matches the
polygon field exactly, and takes 15.4s to do it. That is the whole trade, and it
is why the faster one stays the default rather than being retired.

**A browser build therefore places slightly fewer supports on a dense preset
than the desktop does.** That is a real difference in output, not a rounding
detail, and it is the price of the thing running client-side at all.
"""

from __future__ import annotations

import os
import sys

from .avoidance_polygon import PolygonAvoidanceField, PolygonRegion
from .avoidance_raster import RasterAvoidanceField, RasterRegion
from .types import strut_lean

__all__ = [
    "AvoidanceField",
    "PolygonAvoidanceField",
    "PolygonRegion",
    "RasterAvoidanceField",
    "RasterRegion",
    "backend_name",
    "select_backend",
    "strut_lean",
]

#: Override the automatic choice — ``polygon`` or ``raster``. Set it to run the
#: suite against the backend this platform would not otherwise pick, which is
#: the only way the browser path gets tested on a desktop.
_ENV_VAR = "RSUPPORT_AVOIDANCE"

_BACKENDS = {
    "polygon": PolygonAvoidanceField,
    "raster": RasterAvoidanceField,
}


def select_backend(name: str | None = None) -> type:
    """The collision field class for `name`, or for this platform.

    Emscripten gets the raster field because the polygon one cannot survive
    there; everything else gets the polygon field because it is faster and
    marginally less conservative.
    """
    if name is None:
        name = os.environ.get(_ENV_VAR) or (
            "raster" if sys.platform == "emscripten" else "polygon"
        )
    name = name.strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown avoidance backend {name!r} — expected one of {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]


def backend_name(cls: type | None = None) -> str:
    cls = cls or AvoidanceField
    return "raster" if cls is RasterAvoidanceField else "polygon"


#: Bound to a class, not a factory, so annotations and ``isinstance`` still work
#: and every existing ``from .avoidance import AvoidanceField`` keeps its
#: meaning.
AvoidanceField = select_backend()
