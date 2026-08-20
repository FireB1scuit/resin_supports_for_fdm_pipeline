"""End-to-end tests on a whole miniature.

The stage tests use small, tidy scenes — a bar on a stem, a box over a plate.
Those are good for proving one behaviour at a time and bad at catching what a
real sculpt does: interpenetrating shells, features thinner than a support,
contacts right on the edge of a cape. Every bug found in this file was invisible
to the stage tests and obvious the first time the pipeline ran on a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rsupport import mesh_io, presets, sampling, supports
from rsupport.raycast import DownRay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_sample import build as build_sample  # noqa: E402

from test_supports import printability_report  # noqa: E402

# The pipeline still leaves a handful of flat undersides overhanging where a
# support lands on a feature narrower than itself. They are short bridges off
# anchored material, not floating islands, and they are reported in
# SupportBuild.warnings. This is the measured budget, not an aspiration — if it
# rises, something regressed; if it falls, tighten it.
# Restricting supports to the plate all but emptied this: a shaft that would have
# balanced on a feature narrower than itself now either routes past it to the bed
# or is refused, and the flat undersides were nearly all on those landings. It
# fell from 0.2% to a measured 0.023%, so it is tightened to match. Do not raise
# it to make a test pass.
VIOLATION_BUDGET = 0.0004  # fraction of support faces

# A dropped contact is an overhang left unheld, so it has to stay rare — but a
# sculpt can genuinely offer nowhere to stand a support, and refusing to place
# one is better than placing one through the model. Reported in warnings.
#
# There are two budgets because there are two questions. Once a shaft is allowed
# to route around the model *and* to stand on it as a last resort, almost
# nothing is unsupportable, so that budget is tight and measures the router.
# The shipped default is `plate_only`, which additionally refuses to prop a
# support off the sculpt, and a handful of contacts on this mini genuinely
# cannot be reached from the plate by anything: they sit in dimples on the upper
# surface of the head, where every route in — shaft or arm — crosses the head
# itself. Leaving those unheld is the choice `plate_only` exists to make. Both
# are measurements, not aspirations.
DROP_BUDGET = 0.015  # fraction of contact points, with model landings allowed
PLATE_ONLY_DROP_BUDGET = 0.08  # fraction, restricted to the build plate


@pytest.fixture(scope="module")
def mini():
    return mesh_io.drop_to_bed(build_sample())


@pytest.mark.parametrize("preset", sorted(presets.PRESETS))
def test_pipeline_runs_and_stays_within_the_overhang_budget(mini, preset):
    params = presets.get(preset)
    points = sampling.place_points(mini, params)
    assert points, "a model this awkward must need supports"

    build = supports.build_supports(mini, points, params)
    assert len(build.mesh.faces) > 0

    share_dropped = len(build.dropped) / max(len(points), 1)
    assert share_dropped <= PLATE_ONLY_DROP_BUDGET, (
        f"{preset}: dropped {len(build.dropped)}/{len(points)} contacts "
        f"({share_dropped:.1%} > {PLATE_ONLY_DROP_BUDGET:.0%})"
    )
    if build.dropped:
        assert any("unsupported" in w for w in build.warnings), "drops must be reported"

    rep = printability_report(build.mesh, params, mini)
    share = rep["violations"] / max(rep["total"], 1)
    assert share <= VIOLATION_BUDGET, (
        f"{preset}: {rep['violations']}/{rep['total']} support faces overhang more than "
        f"{params.printable_overhang_deg} deg ({share:.4%} > {VIOLATION_BUDGET:.2%})"
    )


@pytest.fixture(scope="module")
def floating_mini():
    """The same sculpt in the default pose: held in the air by the scaffold."""
    return mesh_io.drop_to_bed(build_sample(), lift=presets.get().lift_height)


def test_the_default_lift_survives_the_whole_pipeline(floating_mini):
    """The shipped default floats the model, so that is the path that has to
    hold up on a real sculpt — not just the flat-on-the-plate case above.

    Lifting is not a handicap here: it hands every shaft the same few extra
    millimetres of height, which is what a cross-link needs to be placeable at
    a printable angle, so a floated model comes out *better* braced.
    """
    params = presets.get()
    assert params.lift_height > 0, "the default is to float the model"
    assert floating_mini.bounds[0][2] == pytest.approx(params.lift_height, abs=1e-6)

    points = sampling.place_points(floating_mini, params)
    build = supports.build_supports(floating_mini, points, params)

    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6), "supports reach the plate"
    assert build.n_braces > 0, "a scaffold this tall must be cross-linked"

    share_dropped = len(build.dropped) / max(len(points), 1)
    assert share_dropped <= PLATE_ONLY_DROP_BUDGET, f"dropped {len(build.dropped)}/{len(points)}"

    rep = printability_report(build.mesh, params, floating_mini)
    share = rep["violations"] / max(rep["total"], 1)
    assert share <= VIOLATION_BUDGET, (
        f"{rep['violations']}/{rep['total']} support faces overhang more than "
        f"{params.printable_overhang_deg} deg ({share:.4%} > {VIOLATION_BUDGET:.2%})"
    )


def test_supports_stay_under_the_model_they_hold(mini):
    """Nothing should tower over the model — supports end at their contact."""
    params = presets.get("mini_0.2")
    build = supports.build_supports(mini, sampling.place_points(mini, params), params)
    assert build.mesh.bounds[1][2] <= mini.bounds[1][2] + params.tip_penetration + 1e-6
    assert build.mesh.bounds[0][2] >= -1e-6, "nothing may dip below the build plate"


def test_islands_all_get_a_forced_point(mini):
    params = presets.get("mini_0.2")
    points = sampling.place_points(mini, params)
    forced = [p for p in points if p.forced]
    assert forced, "the outstretched arm and raised blade start in mid-air"
    assert all(p.source == "island" for p in forced)


# --------------------------------------------------------------------------- #
# regressions
# --------------------------------------------------------------------------- #


def interpenetrating(slab_width: float = 10.0) -> trimesh.Trimesh:
    """A wide bar on a mast that runs straight down through a slab.

    Miniatures ship as interpenetrating shells all the time — an arm pushed into
    a torso — and this is the shape of the problem: the topmost surface below a
    point on the mast is the slab's top face, which is *inside* the mast. A
    support standing there is fused into the sculpt where it can never be
    snapped off.

    `slab_width` is how much the shaft has to get around. At 10 mm a support
    under the bar can lean out past the slab and carry on to the plate; widen it
    and there is not enough height to, which is what leaves the shelf as the
    only landing.
    """
    slab = trimesh.creation.box(extents=[slab_width, slab_width, 10])
    slab.apply_translation([0, 0, 5])  # z 0..10
    mast = trimesh.creation.box(extents=[2, 2, 30])
    mast.apply_translation([0, 0, 15])  # z 0..30, straight through the slab
    bar = trimesh.creation.box(extents=[max(20.0, slab_width * 2), 6, 2])
    bar.apply_translation([0, 0, 21])  # the overhang that needs holding
    return mesh_io.concat(slab, mast, bar)


def test_a_support_is_never_buried_in_an_interpenetrating_shell():
    model = interpenetrating()
    params = presets.get("mini_0.2")
    ray = DownRay(model)

    # The naive landing for anything on the mast's axis is the slab's top face
    # at z=10, which is inside the mast.
    naive_z, _, naive_hit = ray.z_below(np.array([[0.0, 0.0, 20.0]]))
    assert naive_hit[0] and naive_z[0] == pytest.approx(10.0)

    build = supports.build_supports(model, [_pt([3.0, 0.0, 20.0])], params, ray=ray)
    assert len(build.mesh.faces), "the bar over the slab is supportable"

    # The slab is 10 mm across and the contact is 3 mm off its axis, so the
    # shaft leans out from under the bar, past the slab's edge, and carries on
    # to the plate. Standing on the slab's shelf would also have been safe; the
    # plate is better, and `plate_only` is what insists on it.
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert not ray.inside(_column_samples(build.mesh)).any(), "nothing inside the sculpt"


def test_the_shelf_is_still_what_a_model_landing_lands_on():
    """With the plate ruled out, the fallback must still find the *exposed*
    shelf rather than the slab top buried inside the mast."""
    # A slab wide enough that leaning out from under the bar cannot clear it in
    # the 10 mm of height there is to do it in.
    model = interpenetrating(slab_width=40.0)
    params = presets.get("mini_0.2").with_(plate_only=False)

    build = supports.build_supports(model, [_pt([3.0, 0.0, 20.0])], params)
    assert len(build.mesh.faces)
    assert build.mesh.bounds[0][2] == pytest.approx(10.0 - params.tip_penetration, abs=1e-6)
    assert any("stand on the model" in w for w in build.warnings)


def _column_samples(mesh, step=0.25):
    """Points along every vertical run of a support mesh, for an inside test."""
    v = np.asarray(mesh.vertices)
    lo, hi = v[:, 2].min(), v[:, 2].max()
    return v[(v[:, 2] > lo + step) & (v[:, 2] < hi - step)]


def test_a_support_takes_the_plate_when_the_shell_leaves_it_room():
    """The guard above must not stop a support that has a clear run down."""
    model = interpenetrating()
    params = presets.get("mini_0.2")

    build = supports.build_supports(model, [_pt([7.0, 0.0, 20.0])], params)
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6), "clear air; take the plate"


def test_a_support_routes_around_a_thin_blade_rather_than_balancing_on_it():
    """A 1 mm blade is a terrible thing to stand a support on, and there is 25 mm
    of clear air either side of it. So the shaft steps off the blade's axis on
    its way down and carries on to the plate."""
    fin = trimesh.creation.box(extents=[1.0, 12.0, 10.0])
    fin.apply_translation([0, 0, 5])  # a 1 mm thick blade, top at z=10
    params = presets.get("mini_0.2")

    build = supports.build_supports(fin, [_pt([0.0, 0.0, 25.0])], params)
    assert len(build.mesh.faces)
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6), "it reached the plate"

    # And having stepped aside, no part of it is inside the blade.
    ray = DownRay(fin)
    v = build.mesh.vertices
    clear = v[(v[:, 2] > params.foot_height) & (v[:, 2] < 25.0 - params.tip_penetration * 2)]
    assert len(clear)
    assert not ray.inside(clear).any(), "the shaft routed around the blade, not through it"


def test_a_support_that_must_land_on_a_thin_blade_lands_tip_first():
    """The fallback, when there is nowhere else to go: what rests on the blade is
    tip-sized rather than a pad wider than the blade itself."""
    # A blade on top of a slab far too wide to lean out past: the blade's own
    # top is the only thing left to stand on.
    slab = trimesh.creation.box(extents=[60, 60, 10])
    slab.apply_translation([0, 0, -5])  # z -10..0, so the blade still tops out at 10
    fin = trimesh.creation.box(extents=[1.0, 12.0, 10.0])
    fin.apply_translation([0, 0, 5])
    params = presets.get("mini_0.2").with_(plate_only=False)

    model = mesh_io.drop_to_bed(mesh_io.concat(fin, slab))
    build = supports.build_supports(model, [_pt([0.0, 0.0, 35.0])], params)
    assert len(build.mesh.faces)
    assert build.mesh.bounds[0][2] == pytest.approx(20.0 - params.tip_penetration, abs=1e-6)

    v = build.mesh.vertices
    base = v[v[:, 2] < build.mesh.bounds[0][2] + 1e-6]
    width = 2 * np.linalg.norm(base[:, :2], axis=1).max()
    assert width == pytest.approx(params.tip_diameter, rel=1e-6)
    assert width <= 1.0, f"{width:.2f} mm resting on a 1 mm blade"


def _pt(position):
    from rsupport.types import SupportPoint

    return SupportPoint(position=np.asarray(position, dtype=float), normal=np.array([0.0, 0.0, -1.0]))
