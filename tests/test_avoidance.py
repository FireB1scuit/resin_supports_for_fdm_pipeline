"""The reachability sweep every support is routed through.

`free` and `reach` are what make the two structural guarantees structural: a
support never enters the model because it is always inside `free` for its own
radius, and it only rests on the model when `reach` is genuinely empty beneath
it. The relationships pinned here are what the generator relies on.

Every case runs against **both** collision backends. The raster one is what a
browser build uses and what a desktop build never would, so without this it
would only ever be exercised under Emscripten — which is exactly where a
failure is hardest to see. The two disagree slightly on how much room they
find; they may not disagree on any of the properties below.
"""

from __future__ import annotations

import pytest

from rsupport import mesh_io, presets
from rsupport.avoidance import select_backend, strut_lean
from test_supports import table

PARAMS = presets.from_nozzle(0.2)

BACKENDS = ["polygon", "raster"]


@pytest.fixture(params=BACKENDS)
def field(request):
    """An AvoidanceField over the `table` scene, once per backend."""
    model = mesh_io.drop_to_bed(table())
    return select_backend(request.param)(model, PARAMS, top_z=12.0)


def test_reachable_is_a_subset_of_free(field):
    """Reaching the plate from a spot implies being able to stand there."""
    for layer in (0, 5, 12):
        for bucket in (0, len(field.radii) - 1):
            free = field.free(bucket, layer)
            reach = field.reach(bucket, layer)
            assert reach.without(free).area < 1e-6


def test_the_bottom_layer_is_reachable_wherever_it_is_free(field):
    assert field.reach(0, 0).area == pytest.approx(field.free(0, 0).area, rel=1e-9)


def test_fatter_struts_get_less_room(field):
    thin = field.free(0, 8).area
    fat = field.free(len(field.radii) - 1, 8).area
    assert fat < thin


def test_bucket_rounds_the_radius_up(field):
    """Judging a strut against a radius smaller than its own would route it
    into the model, so the lookup must never round down."""
    for r in (field.radii[0], float(field.radii[2] * 0.99), float(field.radii[-1] * 2)):
        assert field.radii[field.bucket(r)] >= r - 1e-9 or field.bucket(r) == len(field.radii) - 1


def test_a_point_in_the_open_has_room_for_anything(field):
    """`room` sizes a base disc, so it has to answer for a spot the model is
    nowhere near — and both backends have to agree that the answer is 'plenty'.
    The raster one measures distance only inside a band, and reads everything
    past it as unobstructed; this is what pins that the band is wide enough."""
    lo, hi = field.bed.bounds[:2], field.bed.bounds[2:]
    corner = (lo[0] + 1e-3, lo[1] + 1e-3)
    assert field.room(corner, 4) >= PARAMS.foot_diameter * 0.5


def test_strut_lean_is_clamped_below_the_printable_limit():
    """A strut may never be allowed to out-overhang what the printer can do."""
    params = PARAMS.with_(strut_lean_deg=85.0, printable_overhang_deg=50.0)
    assert strut_lean(params) <= 48.0
