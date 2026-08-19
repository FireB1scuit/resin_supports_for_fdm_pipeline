"""Stage 3 tests — support geometry.

Point lists are constructed directly here rather than imported from stage 2, so
this file exercises `supports.py` in isolation: that is also how the browser UI
calls it, with a hand-edited list.

The headline test is `test_self_printability_*`: every triangle of the generated
support geometry is walked and its overhang angle checked. If our own supports
would need supports, the generator is broken.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import shapely.geometry
import trimesh

from rsupport import mesh_io, presets
from rsupport.raycast import DownRay
from rsupport.supports import (
    Column,
    build_supports,
    make_brace,
    make_foot,
    make_pillar,
    make_tip,
    overhang_angles,
)
from rsupport.types import SupportParams, SupportPoint

PARAMS = presets.from_nozzle(0.2)


# --------------------------------------------------------------------------- #
# test models — every one a single closed shell, because DownRay.inside is a
# parity test and two overlapping shells would confuse it
# --------------------------------------------------------------------------- #


def extruded(profile_xz, depth: float) -> trimesh.Trimesh:
    """Extrude a 2D profile (given in world XZ) along Y, centred on y=0."""
    poly = shapely.geometry.Polygon(profile_xz)
    assert poly.is_valid
    mesh = trimesh.creation.extrude_polygon(poly, depth)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_translation([0.0, depth * 0.5, 0.0])
    return mesh


def table(bar_z: float = 10.0, half_width: float = 8.0, depth: float = 6.0) -> trimesh.Trimesh:
    """A narrow stem carrying a wide flat bar: two big overhanging ledges with
    clear air all the way to the plate."""
    return extruded(
        [
            (-1.5, 0.0),
            (1.5, 0.0),
            (1.5, bar_z),
            (half_width, bar_z),
            (half_width, bar_z + 2.0),
            (-half_width, bar_z + 2.0),
            (-half_width, bar_z),
            (-1.5, bar_z),
        ],
        depth,
    )


def bracket(depth: float = 6.0) -> trimesh.Trimesh:
    """A C-shape: an arm at z=12 with a slab top at z=4 directly beneath it, so
    supports under the arm must land on the model, not the plate."""
    return extruded(
        [
            (0.0, 0.0),
            (12.0, 0.0),
            (12.0, 16.0),
            (0.0, 16.0),
            (0.0, 12.0),
            (8.0, 12.0),
            (8.0, 4.0),
            (0.0, 4.0),
        ],
        depth,
    )


def solid_box(size: float = 10.0) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[size, size, size])
    return mesh_io.drop_to_bed(box)


def down_point(x, y, z, forced=False, source="overhang") -> SupportPoint:
    return SupportPoint(
        position=np.array([x, y, z], dtype=float),
        normal=np.array([0.0, 0.0, -1.0]),
        forced=forced,
        source=source,
    )


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


def printability_report(mesh, params: SupportParams, model=None, ray=None) -> dict:
    """Walk every triangle and classify its overhang.

    Two exclusions, both physical rather than convenient:

    * faces lying flat on the build plate at z=0 — they are printed on glass;
    * faces in contact with the model — the tip's cap sunk into the surface and
      the bottom of a pad that landed on the model. Material below a face means
      it is not an overhang at all.

    Everything else must be within ``params.printable_overhang_deg``.
    """
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return {"worst": 0.0, "violations": 0, "on_plate": 0, "on_model": 0, "total": 0}

    ang = overhang_angles(mesh)
    real = mesh.area_faces > 1e-12
    steep = real & (ang > params.printable_overhang_deg + 1e-6)

    tri = np.asarray(mesh.triangles)
    plate_tol = params.layer_height * 0.5
    on_plate = steep & (tri[:, :, 2].max(axis=1) <= plate_tol)

    on_model = np.zeros(len(ang), dtype=bool)
    idx = np.where(steep & ~on_plate)[0]
    if len(idx) and model is not None:
        ray = ray or DownRay(model)
        c = tri[idx].mean(axis=1)
        rest_tol = max(params.tip_penetration * 2.0, params.layer_height)
        inside = ray.inside(c)
        z_up, _, hit_up = ray.z_above(c)
        z_dn, _, hit_dn = ray.z_below(c)
        touching = (
            inside
            | (hit_up & ((z_up - c[:, 2]) <= rest_tol))
            | (hit_dn & ((c[:, 2] - z_dn) <= rest_tol))
        )
        on_model[idx[touching]] = True

    violations = steep & ~on_plate & ~on_model
    free = real & ~on_plate & ~on_model
    return {
        "worst": float(ang[free].max()) if free.any() else 0.0,
        "violations": int(violations.sum()),
        "violation_angles": ang[violations],
        "on_plate": int(on_plate.sum()),
        "on_model": int(on_model.sum()),
        "total": int(real.sum()),
    }


def assert_printable(mesh, params: SupportParams, model=None, ray=None):
    rep = printability_report(mesh, params, model, ray)
    assert rep["total"] > 0, "no geometry to check"
    assert rep["violations"] == 0, (
        f"{rep['violations']}/{rep['total']} support faces overhang more than "
        f"{params.printable_overhang_deg} deg (worst {rep['violation_angles'].max():.1f} deg)"
    )
    assert rep["worst"] <= params.printable_overhang_deg + 1e-6
    return rep


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def test_ring_mesh_normals_point_outward():
    pillar = make_pillar([0, 0, 0], [0, 0, 10], PARAMS)
    assert pillar.is_watertight
    c = pillar.triangles_center
    n = pillar.face_normals
    side = np.abs(n[:, 2]) < 0.5
    radial = np.einsum("ij,ij->i", n[side, :2], c[side, :2])
    assert (radial > 0).all(), "pillar side normals must face away from the axis"


@pytest.mark.parametrize("style", ["conical", "spherical"])
def test_tip_is_self_supporting(style):
    params = PARAMS.with_(tip_style=style)
    tip = make_tip([0, 0, 5], params)
    assert tip.is_watertight
    ang = overhang_angles(tip)
    # The only downward face is the bottom cap, where the pillar continues.
    bottom = tip.triangles[:, :, 2].max(axis=1) <= 5 - params.tip_length + 1e-9
    assert ang[~bottom].max() <= params.printable_overhang_deg + 1e-6


def test_foot_flares_downward():
    foot = make_foot([0, 0, 0], PARAMS)
    ang = overhang_angles(foot)
    on_plate = foot.triangles[:, :, 2].max(axis=1) <= 1e-9
    assert on_plate.any(), "the foot must have a face on the plate"
    assert ang[~on_plate].max() <= PARAMS.printable_overhang_deg + 1e-6
    lo, hi = foot.bounds
    assert (hi[0] - lo[0]) == pytest.approx(PARAMS.foot_diameter, rel=1e-6)


def test_pad_is_smaller_than_the_foot():
    foot = make_foot([0, 0, 0], PARAMS, on_model=False)
    pad = make_foot([0, 0, 0], PARAMS, on_model=True)
    assert np.ptp(pad.bounds[:, 0]) == pytest.approx(PARAMS.pad_diameter, rel=1e-6)
    assert np.ptp(pad.bounds[:, 0]) < np.ptp(foot.bounds[:, 0])


def test_brace_at_the_minimum_angle_is_printable():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 5.0])  # exactly 45 degrees
    brace = make_brace(a, b, PARAMS)
    assert brace.is_watertight
    assert overhang_angles(brace).max() <= PARAMS.printable_overhang_deg + 1e-6
    # Both walls and end caps sit at the strut angle; nothing is flat-down.
    assert overhang_angles(brace).max() == pytest.approx(45.0, abs=0.5)


def test_brace_too_shallow_would_not_print():
    """Sanity check on the metric itself: a 20 degree strut is not printable."""
    brace = make_brace([0, 0, 0], [10, 0, 3.6], PARAMS)
    assert overhang_angles(brace).max() > PARAMS.printable_overhang_deg


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


def test_empty_point_list():
    build = build_supports(solid_box(), [], PARAMS)
    assert build.n_points == 0
    assert len(build.mesh.faces) == 0


def test_ledge_pillar_lands_on_the_plate():
    model = table(bar_z=10.0)
    build = build_supports(model, [down_point(5.0, 0.0, 10.0)], PARAMS)

    assert build.n_points == 1
    assert not build.dropped
    lo, hi = build.mesh.bounds
    assert lo[2] == pytest.approx(0.0, abs=1e-9), "column must reach the build plate"
    assert hi[2] == pytest.approx(10.0 + PARAMS.tip_penetration, abs=1e-9)

    # A plate landing gets the wide adhesion foot.
    near_plate = build.mesh.vertices[build.mesh.vertices[:, 2] < 1e-9]
    width = 2 * np.linalg.norm(near_plate[:, :2] - np.array([5.0, 0.0]), axis=1).max()
    assert width == pytest.approx(PARAMS.foot_diameter, rel=1e-6)


def test_pillar_lands_on_the_model_not_through_it():
    model = bracket()
    # Under the arm at z=12; the slab below it tops out at z=4.
    build = build_supports(model, [down_point(4.0, 0.0, 12.0)], PARAMS)

    assert build.n_points == 1
    lo = build.mesh.bounds[0]
    assert lo[2] == pytest.approx(4.0 - PARAMS.tip_penetration, abs=1e-9), (
        "the pillar must stop on the slab at z=4, sunk only by tip_penetration"
    )
    assert lo[2] > 4.0 - 2 * PARAMS.tip_penetration

    # And it must be the small pad, not the 5 mm bed foot.
    base = build.mesh.vertices[build.mesh.vertices[:, 2] < lo[2] + 1e-9]
    width = 2 * np.linalg.norm(base[:, :2] - np.array([4.0, 0.0]), axis=1).max()
    assert width == pytest.approx(PARAMS.pad_diameter, rel=1e-6)


def test_point_over_solid_geometry_never_pierces_the_model():
    """A point sitting on top of a solid block cannot be reached from below.

    The only acceptable outcomes are a support that stops at the surface or no
    support at all — never a pillar driven up through the block.
    """
    model = solid_box(10.0)
    build = build_supports(model, [down_point(0.0, 0.0, 10.0)], PARAMS)

    assert build.dropped, "unreachable point must be reported, not silently lost"
    assert build.warnings
    if len(build.mesh.faces):
        inside = DownRay(model).inside(build.mesh.vertices)
        assert not inside.any(), "support geometry passed through the model"


def test_pillar_above_a_block_lands_on_the_block():
    model = solid_box(10.0)
    build = build_supports(model, [down_point(0.0, 0.0, 13.0)], PARAMS)

    assert build.n_points == 1
    lo = build.mesh.bounds[0]
    assert lo[2] == pytest.approx(10.0 - PARAMS.tip_penetration, abs=1e-9)
    # Nothing may be buried more than the deliberate penetration.
    assert build.mesh.vertices[:, 2].min() > 10.0 - 2 * PARAMS.tip_penetration


def test_blocked_pillar_leans_instead_of_dying():
    """A contact tucked beside the stem: the vertical column's flank clips the
    model, so the router should lean it clear rather than drop the point."""
    model = table(bar_z=10.0)
    build = build_supports(model, [down_point(1.9, 0.0, 10.0)], PARAMS)

    assert build.n_points == 1
    assert not build.dropped
    base = build.mesh.vertices[np.argmin(build.mesh.vertices[:, 2])]
    assert abs(base[0]) > 1.9, "the column should have leaned away from the stem"


def test_dropped_points_are_reported_not_lost():
    model = solid_box(10.0)
    pts = [down_point(0.0, 0.0, 10.0), down_point(0.0, 0.0, 13.0)]
    build = build_supports(model, pts, PARAMS)
    assert build.n_points + len(build.dropped) == len(pts)


# --------------------------------------------------------------------------- #
# tips
# --------------------------------------------------------------------------- #


def test_tip_contact_diameter_matches_the_parameter():
    model = table(bar_z=10.0)
    contact = np.array([5.0, 0.0, 10.0])
    build = build_supports(model, [down_point(*contact)], PARAMS)
    v = build.mesh.vertices

    ring = v[np.abs(v[:, 2] - contact[2]) < 1e-9]
    assert len(ring) == PARAMS.pillar_sections
    width = 2 * np.linalg.norm(ring[:, :2] - contact[:2], axis=1).max()
    assert width == pytest.approx(PARAMS.tip_diameter, rel=1e-6)

    # And it is genuinely thin approaching the contact, not thin at one ring.
    band = v[v[:, 2] > contact[2] - PARAMS.tip_length * 0.05]
    near = 2 * np.linalg.norm(band[:, :2] - contact[:2], axis=1).max()
    assert near <= PARAMS.tip_diameter * 1.2


def test_tip_diameter_follows_the_preset():
    fat = PARAMS.with_(tip_diameter=0.5)
    model = table(bar_z=10.0)
    contact = np.array([5.0, 0.0, 10.0])
    build = build_supports(model, [down_point(*contact)], fat)
    v = build.mesh.vertices
    ring = v[np.abs(v[:, 2] - contact[2]) < 1e-9]
    width = 2 * np.linalg.norm(ring[:, :2] - contact[:2], axis=1).max()
    assert width == pytest.approx(0.5, rel=1e-6)


def test_spherical_tip_is_a_ball_of_the_right_width():
    params = PARAMS.with_(tip_style="spherical")
    model = table(bar_z=10.0)
    contact = np.array([5.0, 0.0, 10.0])
    build = build_supports(model, [down_point(*contact)], params)
    v = build.mesh.vertices
    top = v[v[:, 2] > contact[2] - params.tip_diameter]
    width = 2 * np.linalg.norm(top[:, :2] - contact[:2], axis=1).max()
    assert width == pytest.approx(params.tip_diameter, rel=1e-3)
    assert_printable(build.mesh, params, model)


def test_tip_penetrates_the_model():
    model = table(bar_z=10.0)
    build = build_supports(model, [down_point(5.0, 0.0, 10.0)], PARAMS)
    assert build.mesh.bounds[1][2] == pytest.approx(10.0 + PARAMS.tip_penetration, abs=1e-9)


# --------------------------------------------------------------------------- #
# bracing
# --------------------------------------------------------------------------- #


def test_tall_isolated_pillar_gets_a_prop():
    model = table(bar_z=28.0)
    build = build_supports(model, [down_point(6.0, 0.0, 28.0)], PARAMS)
    free = 28.0 - PARAMS.tip_length - PARAMS.foot_height
    assert free > PARAMS.brace_slenderness * PARAMS.pillar_diameter
    assert build.n_braces >= 1, "a slender lone pillar must be propped"
    assert_printable(build.mesh, PARAMS, model)


def test_short_pillar_is_not_braced():
    model = table(bar_z=6.0)
    build = build_supports(model, [down_point(5.0, 0.0, 6.0)], PARAMS)
    free = 6.0 - PARAMS.tip_length - PARAMS.foot_height
    assert free < PARAMS.brace_slenderness * PARAMS.pillar_diameter
    assert build.n_braces == 0, "short pillars do not need bracing"


def test_neighbouring_tall_pillars_brace_each_other():
    model = table(bar_z=28.0)
    pts = [down_point(4.0, -2.0, 28.0), down_point(4.0, 2.0, 28.0)]
    build = build_supports(model, pts, PARAMS)
    assert build.n_points == 2
    assert build.n_braces == 1, "one strut should serve both pillars"

    # The strut sits between the two axes, not off at the plate.
    mid = np.array([4.0, 0.0])
    near = np.linalg.norm(build.mesh.vertices[:, :2] - mid, axis=1) < 1.0
    assert near.any()
    assert_printable(build.mesh, PARAMS, model)


def test_bracing_can_be_switched_off():
    model = table(bar_z=28.0)
    params = PARAMS.with_(brace_enabled=False)
    build = build_supports(model, [down_point(6.0, 0.0, 28.0)], params)
    assert build.n_braces == 0


def test_impossible_brace_band_disables_bracing_with_a_warning():
    model = table(bar_z=28.0)
    params = PARAMS.with_(printable_overhang_deg=40.0, brace_min_angle_deg=60.0)
    build = build_supports(model, [down_point(6.0, 0.0, 28.0)], params)
    assert build.n_braces == 0
    assert any("bracing disabled" in w for w in build.warnings)


# --------------------------------------------------------------------------- #
# the invariant, on realistic scenes
# --------------------------------------------------------------------------- #


def ledge_grid(bar_z: float, step: float = 1.5) -> list[SupportPoint]:
    pts = []
    for x in np.arange(-7.5, 7.6, step):
        if abs(x) < 2.5:
            continue
        for y in np.arange(-2.5, 2.6, step):
            pts.append(down_point(float(x), float(y), bar_z))
    return pts


def test_self_printability_on_a_plate_landing_scene():
    model = table(bar_z=10.0)
    pts = ledge_grid(10.0)
    assert len(pts) >= 20
    build = build_supports(model, pts, PARAMS)
    assert build.n_points >= len(pts) - 2
    rep = assert_printable(build.mesh, PARAMS, model)
    assert rep["total"] > 1000


def test_self_printability_on_a_model_landing_scene():
    model = bracket()
    pts = [
        down_point(float(x), float(y), 12.0)
        for x in np.arange(1.0, 7.1, 1.5)
        for y in np.arange(-2.0, 2.1, 2.0)
    ]
    build = build_supports(model, pts, PARAMS)
    assert build.n_points >= len(pts) - 2
    assert all(v > 4.0 - 2 * PARAMS.tip_penetration for v in [build.mesh.bounds[0][2]])
    assert_printable(build.mesh, PARAMS, model)


def test_self_printability_with_tall_braced_pillars():
    model = table(bar_z=28.0)
    build = build_supports(model, ledge_grid(28.0, step=3.0), PARAMS)
    assert build.n_braces > 0
    assert_printable(build.mesh, PARAMS, model)


@pytest.mark.parametrize("nozzle", [0.2, 0.25, 0.4])
def test_self_printability_across_presets(nozzle):
    params = presets.from_nozzle(nozzle)
    model = table(bar_z=14.0)
    build = build_supports(model, ledge_grid(14.0, step=3.0), params)
    assert_printable(build.mesh, params, model)


def test_self_printability_with_a_leaning_pillar():
    """Leaning eats overhang budget directly, so check the limit case."""
    params = PARAMS.with_(max_pillar_tilt_deg=PARAMS.printable_overhang_deg - 1.0)
    model = table(bar_z=10.0)
    build = build_supports(model, [down_point(1.9, 0.0, 10.0)], params)
    assert_printable(build.mesh, params, model)


# --------------------------------------------------------------------------- #
# stage-3 contract: cheap, stateless, re-runnable on an edited list
# --------------------------------------------------------------------------- #


def test_rerunning_on_an_edited_list_is_independent():
    model = table(bar_z=10.0)
    pts = ledge_grid(10.0)
    ray = DownRay(model)

    full = build_supports(model, pts, PARAMS, ray=ray)
    again = build_supports(model, pts, PARAMS, ray=ray)
    assert len(full.mesh.faces) == len(again.mesh.faces)

    trimmed = build_supports(model, pts[:-3], PARAMS, ray=ray)
    assert trimmed.n_points == full.n_points - 3
    assert len(trimmed.mesh.faces) < len(full.mesh.faces)


def test_supports_mesh_excludes_the_model():
    model = table(bar_z=10.0)
    build = build_supports(model, ledge_grid(10.0), PARAMS)
    assert len(build.mesh.faces) < len(model.faces) + 10_000
    # Nothing of the bar itself leaked into the supports mesh.
    assert build.mesh.bounds[1][2] <= 10.0 + PARAMS.tip_penetration + 1e-9


def test_smoke_a_few_hundred_points():
    model = table(bar_z=25.0, half_width=20.0, depth=15.0)
    pts = [
        down_point(float(x), float(y), 25.0)
        for x in np.arange(-19.0, 19.1, 1.25)
        for y in np.arange(-7.0, 7.1, 1.25)
        if abs(x) >= 2.5
    ]
    assert len(pts) >= 250

    t0 = time.perf_counter()
    build = build_supports(model, pts, PARAMS)
    elapsed = time.perf_counter() - t0

    assert build.n_points >= len(pts) - 5
    assert elapsed < 60.0
    assert_printable(build.mesh, PARAMS, model)
