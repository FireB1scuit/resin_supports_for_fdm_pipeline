"""Tests for the per-face detail-density metric."""

from __future__ import annotations

import numpy as np
import trimesh

from conftest import bumpy_mask, random_rotation, smooth_mask, synthetic_mini
from rsupport.detail import detail_direction, face_detail


def test_shape_and_range(mini):
    d = face_detail(mini)
    assert d.shape == (len(mini.faces),)
    assert np.all(np.isfinite(d))
    assert d.min() >= 0.0
    assert d.max() <= 1.0 + 1e-9


def test_bumpy_side_scores_far_higher_than_flat_back(mini):
    d = face_detail(mini)
    bumpy = d[bumpy_mask(mini)]
    smooth = d[smooth_mask(mini)]
    assert len(bumpy) > 10 and len(smooth) > 10
    # The flat back is genuinely flat, so this is a large-margin comparison, not
    # a marginal one.
    assert smooth.mean() < 0.05
    assert bumpy.mean() > 0.3
    assert bumpy.mean() > smooth.mean() + 0.25


def test_flat_geometry_has_no_detail():
    box = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    d = face_detail(box)
    # Every face of a box is identical, so nothing stands out as detail. The
    # normalisation must not manufacture contrast that is not there.
    assert float(np.ptp(d)) < 1e-6


def test_scale_invariant(mini):
    small = mini.copy()
    small.apply_scale(0.1)
    assert np.allclose(face_detail(mini), face_detail(small), atol=1e-6)


def test_direction_points_at_the_bumpy_side(mini):
    direction = detail_direction(mini)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # The bumps are on +X; nothing else on the model is asymmetric.
    assert direction[0] > 0.9


def test_direction_rotates_with_the_mesh(mini):
    q = random_rotation(3)
    rotated = mini.copy()
    rotated.apply_transform(q)
    expected = q[:3, :3] @ detail_direction(mini)
    assert np.allclose(detail_direction(rotated), expected, atol=1e-3)


def test_symmetric_model_has_no_direction():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    direction = detail_direction(sphere)
    # A sphere's detail cancels in every direction; callers must cope with the
    # zero vector rather than get a made-up answer.
    assert np.allclose(direction, 0.0)


def test_empty_mesh_is_handled():
    empty = trimesh.Trimesh()
    assert face_detail(empty).shape == (0,)
    assert np.allclose(detail_direction(empty), 0.0)


def test_amplitude_increases_detail():
    quiet = face_detail(mesh := synthetic_mini(bump_amplitude=0.05))
    loud = face_detail(other := synthetic_mini(bump_amplitude=0.6))
    # Values are percentile-normalised, so compare the *share* of the model that
    # reads as detailed rather than the absolute numbers.
    assert loud[bumpy_mask(other)].mean() > quiet[bumpy_mask(mesh)].mean()
