"""Stage 2a: overhang faces and per-layer islands.

The meshes here are deliberately blunt - a T, a floating ball, a cone - because
the properties being asserted are ones you can work out on paper. If a test
fails, the expected answer is not in doubt.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from rsupport.overhang import (
    Island,
    find_islands,
    overhang_mask,
    overhang_severity,
    slice_polygons,
)

LAYER = 0.08


# ------------------------------------------------------------------ fixtures


def t_shape() -> trimesh.Trimesh:
    """A vertical post with a horizontal crossbar on top.

    Post 4x4 from z=0 to z=10.5; crossbar 20x4 from z=10 to z=12, so the two
    overlap rather than merely touching (which is what a real model would do).
    The crossbar's underside at z=10 is a 20x4 plane facing straight down.
    """
    post = trimesh.creation.box(extents=[4.0, 4.0, 10.5])
    post.apply_translation([0.0, 0.0, 10.5 * 0.5])
    bar = trimesh.creation.box(extents=[20.0, 4.0, 2.0])
    bar.apply_translation([0.0, 0.0, 11.0])
    return trimesh.util.concatenate([post, bar])


def floating_ball() -> trimesh.Trimesh:
    """A plate on the bed and a sphere hanging in the air above it.

    The sphere touches nothing: its lowest cross-section has no material
    anywhere beneath it, which is exactly the island condition.
    """
    plate = trimesh.creation.box(extents=[20.0, 20.0, 1.0])
    plate.apply_translation([0.0, 0.0, 0.5])
    ball = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    ball.apply_translation([0.0, 0.0, 10.0])  # spans z = 7 .. 13
    return trimesh.util.concatenate([plate, ball])


def standing_cone() -> trimesh.Trimesh:
    """A cone on its base: every side face leans inward, nothing overhangs."""
    cone = trimesh.creation.cone(radius=6.0, height=20.0, sections=64)
    cone.apply_translation([0.0, 0.0, -cone.bounds[0][2]])
    return cone


BALL_BOTTOM_Z = 7.0


# ------------------------------------------------------------------ severity


def test_severity_is_zero_up_one_down():
    box = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    sev = overhang_severity(box)
    normals = box.face_normals

    assert sev.shape == (len(box.faces),)
    assert sev.min() >= 0.0 and sev.max() <= 1.0
    assert np.allclose(sev[normals[:, 2] > 0.5], 0.0)  # top faces
    assert np.allclose(sev[normals[:, 2] < -0.5], 1.0)  # bottom faces
    assert np.allclose(sev[np.abs(normals[:, 2]) < 1e-6], 0.0)  # vertical walls


def test_severity_matches_cosine_on_a_known_slope():
    """A cone tipped upside down has faces at a known, uniform angle."""
    cone = trimesh.creation.cone(radius=10.0, height=10.0, sections=64)
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    sev = overhang_severity(cone)
    side = np.abs(cone.face_normals[:, 2]) > 1e-6
    side &= sev > 0.0
    # radius == height, so the flank normal is 45 deg off straight down. The
    # tolerance covers the faceting of a 64-sided cone.
    assert np.allclose(sev[side], np.cos(np.radians(45.0)), atol=1e-3)


# ---------------------------------------------------------------------- mask


def test_t_shape_flags_the_crossbar_underside_and_not_the_post():
    mesh = t_shape()
    mask = overhang_mask(mesh, 45.0)
    centers = mesh.triangles_center
    normals = mesh.face_normals

    underside = (normals[:, 2] < -0.9) & (centers[:, 2] > 5.0)
    assert underside.any(), "fixture is wrong: no crossbar underside"
    assert mask[underside].all(), "crossbar underside must need support"

    walls = np.abs(normals[:, 2]) < 1e-6
    assert walls.any()
    assert not mask[walls].any(), "vertical post walls must not need support"


def test_standing_cone_has_no_overhang_above_the_bed():
    mesh = standing_cone()
    mask = overhang_mask(mesh, 45.0)
    off_bed = mesh.triangles_center[:, 2] > 0.5
    assert not mask[off_bed].any()


def test_larger_angle_flags_more_faces():
    """The repo's `mini_0.2_dense` preset raises the angle to hold more, so a
    bigger angle must be a superset. See the convention note in overhang.py."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    tight = overhang_mask(mesh, 30.0)
    loose = overhang_mask(mesh, 60.0)
    assert loose.sum() > tight.sum()
    assert np.all(loose[tight]), "the larger angle must be a superset"


# -------------------------------------------------------------------- slicing


def test_slice_polygons_keeps_holes():
    tube = trimesh.creation.annulus(r_min=2.0, r_max=4.0, height=6.0, sections=64)
    layer = slice_polygons(tube, [0.0])[0]
    assert len(layer) == 1
    poly = layer[0]
    assert len(poly.interiors) == 1
    expected = np.pi * (4.0**2 - 2.0**2)
    assert poly.area == pytest.approx(expected, rel=0.02)


def test_slice_polygons_separates_disjoint_parts():
    a = trimesh.creation.box(extents=[2.0, 2.0, 10.0])
    a.apply_translation([-5.0, 0.0, 5.0])
    b = trimesh.creation.box(extents=[2.0, 2.0, 10.0])
    b.apply_translation([5.0, 0.0, 5.0])
    layer = slice_polygons(trimesh.util.concatenate([a, b]), [5.0])[0]
    assert len(layer) == 2
    assert sum(p.area for p in layer) == pytest.approx(8.0, rel=1e-6)


# -------------------------------------------------------------------- islands


def test_floating_ball_is_an_island_at_the_right_height():
    mesh = floating_ball()
    islands = find_islands(mesh, LAYER)

    assert len(islands) >= 1
    assert all(isinstance(i, Island) for i in islands)

    # The sphere's underside is a tangent point, so the first real cross-section
    # sits a fraction of a layer above z=7 and the polygon is tiny.
    hits = [i for i in islands if abs(i.z - BALL_BOTTOM_Z) < 0.5]
    assert hits, f"no island near z={BALL_BOTTOM_Z}; got {[round(i.z, 2) for i in islands]}"

    island = hits[0]
    assert island.centroid.shape == (3,)
    assert island.centroid[2] == pytest.approx(island.z)
    assert np.linalg.norm(island.centroid[:2]) < 1.0, "island should sit under the ball"
    assert island.area > 0.0
    assert island.polygon.contains(
        island.polygon.centroid
    ) or not island.polygon.is_empty


def test_coarse_step_finds_the_same_island_and_refines_back_to_layer_precision():
    mesh = floating_ball()
    fine = find_islands(mesh, LAYER)
    coarse = find_islands(mesh, LAYER, step=LAYER * 4)

    def nearest_z(islands):
        return min(i.z for i in islands if abs(i.z - BALL_BOTTOM_Z) < 1.0)

    assert nearest_z(coarse) == pytest.approx(nearest_z(fine), abs=LAYER * 2)

    unrefined = find_islands(mesh, LAYER, step=LAYER * 4, refine=False)
    assert nearest_z(unrefined) >= nearest_z(coarse) - 1e-9


def test_a_supported_ledge_is_not_an_island():
    """The T's crossbar is a big overhang but it grows out of the post, so its
    first layer overlaps material below. Overhang, yes; island, no."""
    islands = find_islands(t_shape(), LAYER)
    near_bar = [i for i in islands if 9.0 < i.z < 11.5]
    assert not near_bar, [round(i.z, 2) for i in near_bar]


def test_solid_cone_has_no_islands():
    assert find_islands(standing_cone(), LAYER) == []


def test_min_area_discards_slivers():
    mesh = floating_ball()
    assert find_islands(mesh, LAYER, min_area=0.0)
    # The ball's first cross-section is a sub-mm disc; a huge threshold must
    # throw it away rather than silently keeping it.
    assert find_islands(mesh, LAYER, min_area=1e6) == []


def test_bottom_layer_is_never_an_island():
    """A plain box rests on the plate; nothing about it is floating."""
    box = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    box.apply_translation([0.0, 0.0, 5.0])
    assert find_islands(box, LAYER) == []
