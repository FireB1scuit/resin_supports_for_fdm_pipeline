"""Tree supports — the branching generator.

Three things are being asserted here, in order of how much they matter:

1. **A branch never passes through the model.** Sampled densely along every
   segment of every branch. This is a guarantee, not a preference.
2. **The plate wins whenever it is reachable.** A branch resting on the model
   is only ever allowed when there is genuinely no way down.
3. **Branches actually merge.** A tree that never merges is the old stick
   forest with extra machinery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from rsupport import mesh_io, presets
from rsupport.raycast import DownRay
from rsupport.tree import AvoidanceField, _descend, build_tree
from test_supports import bracket, down_point, printability_report, table

# Trees are a selectable style, not the default any more — resin is.
PARAMS = presets.from_nozzle(0.2).with_(support_style="tree")


def grow(model, points, params=None):
    """Run the descent and hand back everything it produced."""
    params = params or PARAMS
    ray = DownRay(model)
    top = max(float(p.position[2]) for p in points)
    field = AvoidanceField(model, params, top_z=top)
    branches, dropped, warnings, merges = _descend(field, ray, points, params)
    return branches, dropped, warnings, merges, ray, field


def bar_points(z=10.0, params=None):
    """A row of contacts under the table's ledge."""
    return [down_point(float(x), 0.0, z) for x in np.arange(-7.0, 7.1, 1.5) if abs(x) > 2.0]


def hollow_box(outer=20.0, inner=12.0) -> trimesh.Trimesh:
    """A sealed box with a void inside — the plate is unreachable from in there."""
    shell = trimesh.creation.box(extents=[outer, outer, outer])
    cavity = trimesh.creation.box(extents=[inner, inner, inner])
    cavity.invert()
    return mesh_io.drop_to_bed(mesh_io.concat(shell, cavity))


# --------------------------------------------------------------------------- #
# 1. the hard constraint
# --------------------------------------------------------------------------- #


def _pierces(branches, ray, samples=12):
    """Count sampled points along the branches that are inside the model.

    Skips the last segment of a model landing, which is deliberately sunk
    `tip_penetration` into the surface so its base is not a floating layer.
    """
    hits = 0
    for b in branches:
        path = np.asarray(b.path, dtype=float)
        for k in range(len(path) - 1):
            if b.ends_on_model and k == len(path) - 2:
                continue
            a, c = path[k][:3], path[k + 1][:3]
            t = np.linspace(0.05, 0.95, samples)[:, None]
            hits += int(ray.inside(a + (c - a) * t).sum())
    return hits


def test_no_branch_passes_through_the_model():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    from rsupport import sampling

    points = sampling.place_points(model, PARAMS)
    branches, _, _, _, ray, _ = grow(model, points)
    assert _pierces(branches, ray) == 0


def test_no_branch_passes_through_an_awkward_shape():
    """The C-bracket: everything below a contact is model, and the only way out
    is sideways."""
    model = mesh_io.drop_to_bed(bracket())
    points = [down_point(-1.0, 0.0, 12.0), down_point(-3.0, 1.0, 12.0)]
    branches, _, _, _, ray, _ = grow(model, points)
    assert branches
    assert _pierces(branches, ray) == 0


def test_branches_never_lean_past_the_branch_angle():
    """Lean is the entire printability budget: a branch leaning by `a` degrees
    overhangs by exactly `a`."""
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    from rsupport import sampling

    branches, *_ = grow(model, sampling.place_points(model, PARAMS))
    worst = 0.0
    for b in branches:
        path = np.asarray(b.path, dtype=float)
        run = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        drop = -np.diff(path[:, 2])
        moving = drop > 1e-9
        if moving.any():
            worst = max(worst, float(np.degrees(np.arctan2(run[moving], drop[moving])).max()))
    assert worst <= PARAMS.branch_angle_deg + 1.0, f"worst lean {worst:.1f} deg"


# --------------------------------------------------------------------------- #
# 2. the plate wins when it can
# --------------------------------------------------------------------------- #


def test_nothing_rests_on_the_model_when_the_plate_is_clear():
    model = mesh_io.drop_to_bed(table(bar_z=10.0))
    branches, _, _, _, _, _ = grow(model, bar_points())
    assert sum(b.ends_on_model for b in branches) == 0
    assert sum(b.ends_on_plate for b in branches) > 0


def test_a_branch_escapes_sideways_rather_than_resting_on_the_model():
    """Under the C-bracket's arm there is model all the way down, but the slot
    is open at the sides. The old pillar generator lands on the shelf; a branch
    should walk out and take the plate."""
    model = mesh_io.drop_to_bed(bracket())
    points = [down_point(-1.0, 0.0, 12.0), down_point(-3.0, 1.0, 12.0)]
    branches, _, _, _, _, _ = grow(model, points)

    landed = [b for b in branches if b.ends_on_plate or b.ends_on_model]
    assert landed
    assert all(b.ends_on_plate for b in landed), "the plate was reachable and was not used"

    # It can only have got there by travelling out of the slot.
    drift = max(np.abs(np.asarray(b.path)[:, 1]).max() for b in branches)
    assert drift > 3.0, f"expected a sideways escape, drifted only {drift:.1f} mm"


def test_a_sealed_cavity_falls_back_to_resting_on_the_model():
    """The fallback still has to work: inside a sealed void there is no way
    down, so the branch rests on the floor of the cavity and says so."""
    model = hollow_box()
    ceiling = model.bounds[1][2] - 4.0
    branches, _, warnings, _, _, _ = grow(model, [down_point(0.0, 0.0, ceiling)])

    landed = [b for b in branches if b.ends_on_plate or b.ends_on_model]
    assert landed and all(b.ends_on_model for b in landed)
    assert any("rest on the model" in w for w in warnings)
    assert landed[0].land_z == pytest.approx(4.0, abs=0.6), "should sit on the cavity floor"


# --------------------------------------------------------------------------- #
# 3. it is actually a tree
# --------------------------------------------------------------------------- #


def test_branches_merge_into_far_fewer_feet_than_contacts():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    from rsupport import sampling

    points = sampling.place_points(model, PARAMS)
    branches, _, _, merges, _, _ = grow(model, points)
    feet = sum(b.ends_on_plate for b in branches)

    assert merges > 0
    assert feet < len(points) * 0.6, f"{feet} feet for {len(points)} contacts is a stick forest"


def test_merge_strength_controls_how_much_they_merge():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    from rsupport import sampling

    counts = []
    for strength in (0.0, 0.5, 1.0):
        params = PARAMS.with_(merge_strength=strength)
        points = sampling.place_points(model, params)
        branches, *_ = grow(model, points, params)
        counts.append(sum(b.ends_on_plate for b in branches))

    assert counts[0] > counts[1] > counts[2], f"feet should fall as merging rises: {counts}"


def test_a_trunk_is_fatter_than_the_branches_feeding_it():
    """Cross-section is conserved on merge, so a trunk carries real section."""
    model = mesh_io.drop_to_bed(table(bar_z=14.0))
    branches, *_ = grow(model, bar_points(z=14.0))

    tips = [b for b in branches if b.contact is not None]
    trunks = [b for b in branches if b.contact is None]
    assert trunks, "nothing merged"
    widest_tip = max(np.asarray(b.path)[:, 3].max() for b in tips)
    widest_trunk = max(np.asarray(b.path)[:, 3].max() for b in trunks)
    assert widest_trunk > widest_tip


def test_branch_radius_never_widens_going_up():
    """A profile that widens upward is an overhang. Every branch must taper the
    other way."""
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    branches, *_ = grow(model, bar_points(z=12.0))
    for b in branches:
        path = np.asarray(b.path, dtype=float)  # top -> bottom
        assert np.all(np.diff(path[:, 3]) >= -1e-9), "radius shrinks going down"


# --------------------------------------------------------------------------- #
# the avoidance field itself
# --------------------------------------------------------------------------- #


def test_reachable_is_a_subset_of_free():
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


def test_fatter_branches_get_less_room():
    model = mesh_io.drop_to_bed(table())
    field = AvoidanceField(model, PARAMS, top_z=12.0)
    thin = field.free(0, 8).area
    fat = field.free(len(field.radii) - 1, 8).area
    assert fat < thin


def test_branch_angle_is_clamped_below_the_printable_limit():
    """A branch may never be allowed to out-overhang what the printer can do."""
    from rsupport.tree import _branch_angle

    params = PARAMS.with_(branch_angle_deg=85.0, printable_overhang_deg=50.0)
    assert _branch_angle(params) <= 48.0


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_output_is_printable_and_deterministic():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)
    a = build_tree(model, points, PARAMS)
    b = build_tree(model, points, PARAMS)
    assert len(a.mesh.faces) == len(b.mesh.faces)

    rep = printability_report(a.mesh, PARAMS, model)
    assert rep["total"] > 0
    assert rep["violations"] == 0, f"{rep['violations']}/{rep['total']} faces overhang"


def test_no_points_means_no_geometry():
    model = mesh_io.drop_to_bed(table())
    assert len(build_tree(model, [], PARAMS).mesh.faces) == 0


def test_build_supports_dispatches_on_style():
    from rsupport.supports import build_supports

    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)

    sizes = {
        style: len(build_supports(model, points, PARAMS.with_(support_style=style)).mesh.faces)
        for style in ("resin", "tree", "pillar")
    }
    assert len(set(sizes.values())) == 3, f"styles should differ: {sizes}"
    assert presets.from_nozzle(0.2).support_style == "resin"


def test_geometry_stays_on_the_right_side_of_the_plate():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    build = build_tree(model, bar_points(z=12.0), PARAMS)
    assert build.mesh.bounds[0][2] >= -1e-6
    assert build.mesh.bounds[1][2] <= model.bounds[1][2] + PARAMS.tip_penetration + 1e-6
