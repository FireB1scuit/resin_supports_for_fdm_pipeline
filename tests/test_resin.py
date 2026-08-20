"""The resin scaffold — the only support structure this project builds.

Four things are asserted here, in order of how much they matter:

1. **A shaft never passes through the model.** This is a guarantee, not a
   preference.
2. **The plate wins whenever it is reachable.** A shaft resting on the model is
   only ever allowed when there is genuinely no way down.
3. **It stays a scaffold.** Shafts stay thin and separate, arms fan off them to
   share, and cross-links tie the whole thing together. Nothing fuses into a
   thickening trunk — that would be an organic tree, which is the wrong
   structure for this project.
4. **The output is deterministic and stays on the right side of the plate.**
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from rsupport import mesh_io, presets, sampling
from rsupport.avoidance import AvoidanceField
from rsupport.raycast import DownRay
from rsupport.resin import _plan_shafts, build_resin
from test_supports import bracket, down_point, printability_report, table

PARAMS = presets.from_nozzle(0.2)


def plan(model, points, params=None):
    """Run the planning half and hand back everything it produced."""
    params = params or PARAMS
    ray = DownRay(model)
    top = max(float(p.position[2]) for p in points)
    field = AvoidanceField(model, params, top_z=top)
    shafts, dropped, warnings = _plan_shafts(field, ray, points, params)
    return shafts, dropped, warnings, ray, field


def bar_points(z=10.0):
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


def _pierces(shafts, ray, samples=12):
    """Count sampled points along the shafts that are inside the model.

    Skips the bottom of a model landing, which is deliberately sunk
    `tip_penetration` into the surface so its base is not a floating layer.
    """
    hits = 0
    for s in shafts:
        lo = s.land_z + PARAMS.tip_length if s.on_model else s.land_z
        if s.top_z - lo <= 1e-9:
            continue
        z = np.linspace(lo, s.top_z, samples)
        pts = np.column_stack([np.full(samples, s.xy[0]), np.full(samples, s.xy[1]), z])
        hits += int(ray.inside(pts).sum())
    return hits


def test_no_shaft_passes_through_the_model():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    points = sampling.place_points(model, PARAMS)
    shafts, _, _, ray, _ = plan(model, points)
    assert shafts
    assert _pierces(shafts, ray) == 0


def test_no_shaft_passes_through_an_awkward_shape():
    """The C-bracket: everything below a contact is model, and the only way out
    is sideways."""
    model = mesh_io.drop_to_bed(bracket())
    points = [down_point(-1.0, 0.0, 12.0), down_point(-3.0, 1.0, 12.0)]
    shafts, _, _, ray, _ = plan(model, points)
    assert shafts
    assert _pierces(shafts, ray) == 0


def test_geometry_never_enters_the_model():
    """The same guarantee, checked on the meshed result rather than the plan.

    Two bands are excluded, both deliberate rather than convenient: the tips at
    the top, which are sunk ``tip_penetration`` into the surface on purpose, and
    the plate itself, where the flared adhesion foot and the model's own
    footprint are both printed flat on glass and may overlap.
    """
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    build = build_resin(model, bar_points(z=12.0), PARAMS)
    ray = DownRay(model)
    v = build.mesh.vertices
    clear = v[(v[:, 2] > PARAMS.foot_height) & (v[:, 2] < 12.0 - PARAMS.tip_penetration * 2)]
    assert len(clear)
    assert not ray.inside(clear).any()


# --------------------------------------------------------------------------- #
# 2. the plate wins when it can
# --------------------------------------------------------------------------- #


def test_nothing_rests_on_the_model_when_the_plate_is_clear():
    model = mesh_io.drop_to_bed(table(bar_z=10.0))
    shafts, _, _, _, _ = plan(model, bar_points())
    assert shafts
    assert sum(s.on_model for s in shafts) == 0


def test_a_sealed_cavity_falls_back_to_resting_on_the_model():
    """The fallback still has to work: inside a sealed void there is no way
    down, so the shaft rests on the floor of the cavity and says so."""
    model = hollow_box()
    ceiling = model.bounds[1][2] - 4.0
    build = build_resin(model, [down_point(0.0, 0.0, ceiling)], PARAMS)

    assert build.mesh.bounds[0][2] == pytest.approx(4.0, abs=0.6), "should sit on the cavity floor"
    assert any("stand on the model" in w for w in build.warnings)


# --------------------------------------------------------------------------- #
# 3. it is a scaffold, not a tree
# --------------------------------------------------------------------------- #


def test_parenting_controls_how_many_shafts_stand_on_the_plate():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))

    counts = []
    for strength in (0.0, 0.5, 1.0):
        params = PARAMS.with_(parenting=strength)
        points = sampling.place_points(model, params)
        shafts, *_ = plan(model, points, params)
        counts.append(len(shafts))

    assert counts[0] > counts[-1], f"shafts should fall as parenting rises: {counts}"


def test_arms_share_shafts_rather_than_one_shaft_per_contact():
    model = mesh_io.drop_to_bed(table(bar_z=14.0))
    points = bar_points(z=14.0)
    shafts, *_ = plan(model, points)
    assert sum(len(s.arms) for s in shafts) == len(points)
    assert len(shafts) < len(points), "no contact was parented onto a shared shaft"


def test_a_shaft_never_thickens_into_a_trunk():
    """The whole point of the style: shafts stay the same slim taper however
    many arms they carry. If this starts failing, the generator is drifting
    into organic-tree territory."""
    model = mesh_io.drop_to_bed(table(bar_z=14.0))
    build = build_resin(model, bar_points(z=14.0), PARAMS)

    v = build.mesh.vertices
    # Ignore the flared feet at the plate and the arms fanning out up top.
    band = v[(v[:, 2] > PARAMS.foot_height * 2) & (v[:, 2] < 14.0 - PARAMS.tip_length * 3)]
    assert len(band)
    # Every ring in that band belongs to a shaft, so no ring may be wider than
    # the shaft's own bottom diameter.
    from scipy.spatial import cKDTree

    axes = np.array([s.xy for s in plan(model, bar_points(z=14.0))[0]])
    dist, _ = cKDTree(axes).query(band[:, :2])
    assert dist.max() <= PARAMS.shaft_lower_diameter * 0.5 + 1e-6


def test_shafts_are_cross_linked_into_a_lattice():
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    build = build_resin(model, bar_points(z=28.0), PARAMS)
    assert build.n_braces > 0, "tall neighbouring shafts must brace each other"


def test_cross_links_can_be_switched_off():
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    build = build_resin(model, bar_points(z=28.0), PARAMS.with_(brace_enabled=False))
    assert build.n_braces == 0


# --------------------------------------------------------------------------- #
# 4. contract
# --------------------------------------------------------------------------- #


def test_output_is_printable_and_deterministic():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)
    a = build_resin(model, points, PARAMS)
    b = build_resin(model, points, PARAMS)
    assert len(a.mesh.faces) == len(b.mesh.faces)

    rep = printability_report(a.mesh, PARAMS, model)
    assert rep["total"] > 0
    assert rep["violations"] == 0, f"{rep['violations']}/{rep['total']} faces overhang"


def test_no_points_means_no_geometry():
    model = mesh_io.drop_to_bed(table())
    assert len(build_resin(model, [], PARAMS).mesh.faces) == 0


def test_geometry_stays_on_the_right_side_of_the_plate():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    build = build_resin(model, bar_points(z=12.0), PARAMS)
    assert build.mesh.bounds[0][2] >= -1e-6
    assert build.mesh.bounds[1][2] <= model.bounds[1][2] + PARAMS.tip_penetration + 1e-6


def test_build_supports_builds_the_scaffold():
    """There is one structure; the entry point must produce exactly it."""
    from rsupport.supports import build_supports

    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)
    assert (
        len(build_supports(model, points, PARAMS).mesh.faces)
        == len(build_resin(model, points, PARAMS).mesh.faces)
    )


# --------------------------------------------------------------------------- #
# 5. the model in the air
# --------------------------------------------------------------------------- #


def test_the_bases_of_a_lifted_model_overlap_into_a_raft():
    """Why the base is as wide as it is.

    With the model held in the air a shaft lands roughly every
    ``support_spacing``, so a base wider than that spacing means neighbouring
    footprints *merge*. What ends up on the glass is one sheet rather than a
    field of separate discs, and a scaffold of 1.2 mm shafts stays put.
    """
    lift = 5.0
    params = PARAMS.with_(lift_height=lift)
    assert params.foot_diameter > params.support_spacing, (
        "a base narrower than the support spacing could not overlap anything"
    )
    model = mesh_io.drop_to_bed(trimesh.creation.box(extents=[14, 14, 10]), lift=lift)

    points = sampling.place_points(model, params)
    shafts, _, _, _, _ = plan(model, points, params)
    feet = np.array([s.xy for s in shafts if not s.on_model])
    assert len(feet) > 4, "a lifted slab should stand on a forest of shafts"

    d = np.linalg.norm(feet[:, None] - feet[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    touching = d.min(axis=1) < params.foot_diameter
    assert touching.mean() > 0.9, (
        f"only {touching.mean():.0%} of bases overlap a neighbour; that is a field "
        "of discs, not a raft"
    )


def test_a_lifted_model_is_held_off_the_plate_by_supports_alone():
    """Nothing of the model may touch the plate, and every support must reach
    it — the whole point of lifting is that the scaffold carries the model."""
    lift = 5.0
    params = PARAMS.with_(lift_height=lift)
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"), lift=lift)
    assert model.bounds[0][2] == pytest.approx(lift, abs=1e-6)

    build = build_resin(model, sampling.place_points(model, params), params)
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert build.mesh.bounds[1][2] <= model.bounds[1][2] + params.tip_penetration + 1e-6
    rep = printability_report(build.mesh, params, model)
    assert rep["violations"] == 0

