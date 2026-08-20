"""The reachability sweep every support is routed through.

`free` and `reach` are what make the two structural guarantees structural: a
support never enters the model because it is always inside `free` for its own
radius, and it only rests on the model when `reach` is genuinely empty beneath
it. The relationships pinned here are what the generator relies on.
"""

from __future__ import annotations

import pytest

from rsupport import mesh_io, presets
from rsupport.avoidance import AvoidanceField, strut_lean
from test_supports import table

PARAMS = presets.from_nozzle(0.2)


def test_reachable_is_a_subset_of_free():
    """Reaching the plate from a spot implies being able to stand there."""
    model = mesh_io.drop_to_bed(table())
    field = AvoidanceField(model, PARAMS, top_z=12.0)
    for layer in (0, 5, 12):
        for bucket in (0, len(field.radii) - 1):
            free = field.free(bucket, layer)
            reach = field.reach(bucket, layer)
            assert reach.difference(free.buffer(1e-6)).area < 1e-6


def test_the_bottom_layer_is_reachable_wherever_it_is_free():
    model = mesh_io.drop_to_bed(table())
    field = AvoidanceField(model, PARAMS, top_z=12.0)
    assert field.reach(0, 0).area == pytest.approx(field.free(0, 0).area, rel=1e-9)


def test_fatter_struts_get_less_room():
    model = mesh_io.drop_to_bed(table())
    field = AvoidanceField(model, PARAMS, top_z=12.0)
    thin = field.free(0, 8).area
    fat = field.free(len(field.radii) - 1, 8).area
    assert fat < thin


def test_bucket_rounds_the_radius_up():
    """Judging a strut against a radius smaller than its own would route it
    into the model, so the lookup must never round down."""
    model = mesh_io.drop_to_bed(table())
    field = AvoidanceField(model, PARAMS, top_z=12.0)
    for r in (field.radii[0], float(field.radii[2] * 0.99), float(field.radii[-1] * 2)):
        assert field.radii[field.bucket(r)] >= r - 1e-9 or field.bucket(r) == len(field.radii) - 1


def test_strut_lean_is_clamped_below_the_printable_limit():
    """A strut may never be allowed to out-overhang what the printer can do."""
    params = PARAMS.with_(strut_lean_deg=85.0, printable_overhang_deg=50.0)
    assert strut_lean(params) <= 48.0
