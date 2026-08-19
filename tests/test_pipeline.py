"""End-to-end tests on a whole miniature.

The stage tests use small, tidy scenes — a bar on a stem, a box over a plate.
Those are good for proving one behaviour at a time and bad at catching what a
real sculpt does: interpenetrating shells, features thinner than a pillar,
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
from rsupport.supports import _fit_pads, _landing, _route_columns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_sample import build as build_sample  # noqa: E402

from test_supports import printability_report  # noqa: E402

# The pipeline still leaves a handful of flat pad undersides overhanging where a
# support lands on a feature narrower than its own pillar. They are short
# bridges off anchored material, not floating islands, and they are reported in
# SupportBuild.warnings. This is the measured budget, not an aspiration — if it
# rises, something regressed; if it falls, tighten it.
VIOLATION_BUDGET = 0.002  # fraction of support faces


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
    assert not build.dropped, f"dropped {len(build.dropped)} points that should have been routable"

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


def test_landing_skips_a_surface_buried_in_another_shell():
    """A pillar may not stand on a surface that is inside the model.

    Miniatures ship as interpenetrating shells all the time — an arm pushed
    into a torso. The topmost surface below a contact is often the part of one
    shell buried inside another, and a pillar landing there is fused into the
    sculpt where it can never be snapped off.
    """
    slab = trimesh.creation.box(extents=[10, 10, 10])
    slab.apply_translation([0, 0, 5])  # z 0..10
    mast = trimesh.creation.box(extents=[2, 2, 30])
    mast.apply_translation([0, 0, 15])  # z 0..30, straight through the slab
    model = mesh_io.concat(slab, mast)
    ray = DownRay(model)

    xy = np.array([[0.0, 0.0]])
    from_z = np.array([20.0])  # inside the mast, above the slab

    # The naive answer is the slab's top face at z=10 — which is inside the mast.
    naive_z, _, naive_hit = ray.z_below(np.array([[0.0, 0.0, 20.0]]))
    assert naive_hit[0] and naive_z[0] == pytest.approx(10.0)

    z, on_model = _landing(ray, xy, from_z, plate_eps=1e-3)
    assert not on_model[0], "landed on a surface buried inside the mast"
    assert z[0] == pytest.approx(0.0)


def test_landing_still_finds_an_exposed_surface():
    """The guard above must not reject ordinary landings."""
    slab = trimesh.creation.box(extents=[10, 10, 10])
    slab.apply_translation([0, 0, 5])
    ray = DownRay(slab)

    z, on_model = _landing(ray, np.array([[0.0, 0.0]]), np.array([20.0]), plate_eps=1e-3)
    assert on_model[0]
    assert z[0] == pytest.approx(10.0)


def test_pad_shrinks_to_the_feature_it_lands_on():
    """A pad wider than what it stands on hangs off the edge.

    Landing on an arm or a blade barely wider than the pillar is normal on a
    miniature, so the pad is measured against the material rather than assumed.
    """
    fin = trimesh.creation.box(extents=[1.0, 12.0, 10.0])
    fin.apply_translation([0, 0, 5])  # a 1 mm thick blade, top at z=10
    ray = DownRay(fin)
    params = presets.get("mini_0.2")

    contact = np.array([0.0, 0.0, 25.0])
    columns, _, _ = _route_columns(ray, [_pt(contact)], params)
    assert len(columns) == 1
    col = columns[0]
    assert col.on_model and col.land_z == pytest.approx(10.0)

    # Default pad is 2 mm across; the blade is 1 mm thick. It has to give.
    assert col.base_r > 0.0
    assert col.base_r * 2 <= 1.0 + 1e-6, f"pad {col.base_r * 2:.2f} mm wide on a 1 mm blade"


def test_wide_landing_keeps_the_full_pad():
    """The shrinking must not fire when there is material to rest on."""
    slab = trimesh.creation.box(extents=[20, 20, 10])
    slab.apply_translation([0, 0, 5])
    ray = DownRay(slab)
    params = presets.get("mini_0.2")

    columns, _, _ = _route_columns(ray, [_pt(np.array([0.0, 0.0, 25.0]))], params)
    assert columns[0].base_r == pytest.approx(params.pad_diameter * 0.5)


def test_fit_pads_is_a_no_op_without_model_landings():
    slab = trimesh.creation.box(extents=[20, 20, 2])
    slab.apply_translation([0, 0, 1])
    ray = DownRay(slab)
    assert _fit_pads(ray, [], presets.get("mini_0.2")) == 0


def _pt(position):
    from rsupport.types import SupportPoint

    return SupportPoint(position=np.asarray(position, dtype=float), normal=np.array([0.0, 0.0, -1.0]))
