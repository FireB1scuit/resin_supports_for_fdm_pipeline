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
VIOLATION_BUDGET = 0.002  # fraction of support faces

# A dropped contact is an overhang left unheld, so it has to stay rare — but a
# sculpt can genuinely offer nowhere to stand a support, and refusing to place
# one is better than placing one through the model. Reported in warnings.
DROP_BUDGET = 0.02  # fraction of contact points


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
    assert share_dropped <= DROP_BUDGET, (
        f"{preset}: dropped {len(build.dropped)}/{len(points)} contacts "
        f"({share_dropped:.1%} > {DROP_BUDGET:.0%})"
    )
    if build.dropped:
        assert any("nowhere" in w for w in build.warnings), "drops must be reported"

    rep = printability_report(build.mesh, params, mini)
    share = rep["violations"] / max(rep["total"], 1)
    assert share <= VIOLATION_BUDGET, (
        f"{preset}: {rep['violations']}/{rep['total']} support faces overhang more than "
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


def interpenetrating() -> trimesh.Trimesh:
    """A wide bar on a mast that runs straight down through a slab.

    Miniatures ship as interpenetrating shells all the time — an arm pushed into
    a torso — and this is the shape of the problem: the topmost surface below a
    point on the mast is the slab's top face, which is *inside* the mast. A
    support standing there is fused into the sculpt where it can never be
    snapped off.
    """
    slab = trimesh.creation.box(extents=[10, 10, 10])
    slab.apply_translation([0, 0, 5])  # z 0..10
    mast = trimesh.creation.box(extents=[2, 2, 30])
    mast.apply_translation([0, 0, 15])  # z 0..30, straight through the slab
    bar = trimesh.creation.box(extents=[20, 6, 2])
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

    # It rests on the slab's exposed shelf, sunk by exactly the tip penetration
    # and no further: nothing was driven down through the shelf.
    assert build.mesh.bounds[0][2] == pytest.approx(10.0 - params.tip_penetration, abs=1e-6)


def test_a_support_takes_the_plate_when_the_shell_leaves_it_room():
    """The guard above must not stop a support that has a clear run down."""
    model = interpenetrating()
    params = presets.get("mini_0.2")

    build = supports.build_supports(model, [_pt([7.0, 0.0, 20.0])], params)
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6), "clear air; take the plate"


def test_a_support_on_a_thin_blade_does_not_hang_off_it():
    """Landing on an arm or a blade barely wider than the support itself is
    normal on a miniature. A resin shaft ends in a tip there, so what rests on
    the blade is tip-sized rather than a pad wider than the blade."""
    fin = trimesh.creation.box(extents=[1.0, 12.0, 10.0])
    fin.apply_translation([0, 0, 5])  # a 1 mm thick blade, top at z=10
    params = presets.get("mini_0.2")

    build = supports.build_supports(fin, [_pt([0.0, 0.0, 25.0])], params)
    assert len(build.mesh.faces)
    assert build.mesh.bounds[0][2] == pytest.approx(10.0 - params.tip_penetration, abs=1e-6)

    v = build.mesh.vertices
    base = v[v[:, 2] < build.mesh.bounds[0][2] + 1e-6]
    width = 2 * np.linalg.norm(base[:, :2], axis=1).max()
    assert width == pytest.approx(params.tip_diameter, rel=1e-6)
    assert width <= 1.0, f"{width:.2f} mm resting on a 1 mm blade"


def _pt(position):
    from rsupport.types import SupportPoint

    return SupportPoint(position=np.asarray(position, dtype=float), normal=np.array([0.0, 0.0, -1.0]))
