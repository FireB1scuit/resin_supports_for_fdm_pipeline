"""Tests for candidate generation and orientation scoring."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from conftest import random_rotation
from rsupport import orient, presets
from rsupport.detail import detail_direction
from rsupport.types import Orientation

DOWN = np.array([0.0, 0.0, -1.0])


@pytest.fixture(scope="module")
def params():
    return presets.get()


def _angle_from_down(vector: np.ndarray) -> float:
    """Degrees between `vector` and the build plate normal."""
    v = np.asarray(vector, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(float(v @ DOWN), -1.0, 1.0))))


# --------------------------------------------------------------------- candidates


def test_candidates_are_labelled_deduped_and_capped(mini, params):
    candidates = orient.candidate_matrices(mini, params)
    assert 0 < len(candidates) <= orient.MAX_CANDIDATES

    labels = [label for _, label in candidates]
    assert "flat-base" in labels
    assert any(label.startswith("upright-lean-") for label in labels)
    assert any(label.startswith("base-lean-") for label in labels)

    downs = np.array([m[:3, :3].T @ DOWN for m, _ in candidates])
    dots = downs @ downs.T
    np.fill_diagonal(dots, -1.0)
    assert dots.max() < np.cos(np.radians(orient.DEDUPE_ANGLE_DEG)) + 1e-9


def test_candidate_matrices_are_rotations_dropped_to_the_bed(mini, params):
    for matrix, _ in orient.candidate_matrices(mini, params):
        rot = matrix[:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rot), 1.0)
        posed = orient.apply(mini, matrix)
        lo, hi = posed.bounds
        assert np.isclose(lo[2], 0.0, atol=1e-9)
        assert np.allclose((lo[:2] + hi[:2]) * 0.5, 0.0, atol=1e-9)


def test_lean_candidates_tip_the_detail_side_upward(mini, params):
    direction = detail_direction(mini)
    by_label = {label: m for m, label in orient.candidate_matrices(mini, params)}
    tilts = sorted(
        (int(label.rsplit("-", 1)[1]), m) for label, m in by_label.items()
        if label.startswith("base-lean-")
    )
    assert tilts, "expected lean variants of the flat-base pose"
    heights = [float((m[:3, :3] @ direction)[2]) for _, m in tilts]
    # Leaning must raise the detail-bearing side, monotonically, and never past
    # the configured limit.
    assert all(b > a for a, b in zip(heights, heights[1:]))
    assert tilts[-1][0] <= params.max_lean_deg
    assert heights[-1] == pytest.approx(np.sin(np.radians(tilts[-1][0])), abs=1e-6)


def test_lean_survives_a_model_with_no_detail_direction(params):
    """A symmetric model returns a zero detail direction; candidates still build."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=6.0)
    assert not detail_direction(sphere).any()
    candidates = orient.candidate_matrices(sphere, params)
    assert len(candidates) > 1


# ------------------------------------------------------------------------ scoring


def test_score_reports_every_weighted_term(mini, params):
    matrix, _ = orient.candidate_matrices(mini, params)[0]
    score, terms = orient.score_orientation(mini, matrix, params)
    assert set(orient.WEIGHTS) <= set(terms)
    assert np.isfinite(score)
    assert all(np.isfinite(v) for v in terms.values())


def test_upside_down_scores_worse_than_base_down(mini, params):
    upright = orient._pose_matrix(mini, DOWN)
    flipped = orient._pose_matrix(mini, -DOWN)
    up_score, up_terms = orient.score_orientation(mini, upright, params)
    down_score, down_terms = orient.score_orientation(mini, flipped, params)
    assert up_score < down_score
    # The reason should be the footprint, not an accident of some other term.
    assert up_terms["stability"] < down_terms["stability"]


def test_detail_penalty_responds_to_detail_facing_down(mini, params):
    """Rolling the bumpy side under the model must cost more than leaving it up."""
    # `_pose_matrix` takes the direction that ends up facing the plate, so +X
    # down is the bumpy side rolled underneath.
    front_down = orient._pose_matrix(mini, np.array([1.0, 0.0, 0.0]))
    front_up = orient._pose_matrix(mini, np.array([-1.0, 0.0, 0.0]))
    _, up_terms = orient.score_orientation(mini, front_up, params)
    _, down_terms = orient.score_orientation(mini, front_down, params)
    assert down_terms["detail_penalty"] > up_terms["detail_penalty"]


def test_score_is_scale_invariant(mini, params):
    small = mini.copy()
    small.apply_scale(0.25)
    a, terms_a = orient.score_orientation(mini, orient._pose_matrix(mini, DOWN), params)
    b, terms_b = orient.score_orientation(small, orient._pose_matrix(small, DOWN), params)
    # Not exact: `layer_height` is an absolute length, so the bed-contact band is
    # relatively thicker on a shrunken model.
    assert b == pytest.approx(a, abs=0.05)
    assert terms_b["height"] == pytest.approx(terms_a["height"], abs=1e-6)


# -------------------------------------------------------------------- the answer


def test_best_orientation_stands_the_mini_on_its_base(mini, params):
    best = orient.best_orientation(mini, params)
    assert isinstance(best, Orientation)
    # The disc base's outward normal is -Z in the model as built.
    settled = best.matrix[:3, :3] @ DOWN
    assert _angle_from_down(settled) <= params.max_lean_deg + 1e-6


def test_best_orientation_puts_the_widest_face_on_the_plate(mini, params):
    posed = orient.apply(mini, orient.best_orientation(mini, params))
    lo, hi = posed.bounds
    # The base disc is the widest thing on the model; standing correctly, the
    # model is taller than it is wide and the disc is at the bottom.
    assert hi[2] - lo[2] > (hi[0] - lo[0])
    bottom_slab = posed.vertices[posed.vertices[:, 2] < params.layer_height]
    assert len(bottom_slab) > 20
    assert np.linalg.norm(bottom_slab[:, :2], axis=1).max() > (hi[0] - lo[0]) * 0.45


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_result_is_invariant_to_the_input_pose(mini, params, seed):
    """The same physical face must end up on the plate however the STL arrived."""
    q = random_rotation(seed)
    rotated = mini.copy()
    rotated.apply_transform(q)

    best = orient.best_orientation(rotated, params)
    # Track the disc base's normal (-Z originally) through both transforms.
    settled = best.matrix[:3, :3] @ (q[:3, :3] @ DOWN)
    assert _angle_from_down(settled) <= params.max_lean_deg + 1e-6


def test_orientations_are_ranked_and_distinct(mini, params):
    ranked = orient.orientations(mini, params, top_k=3)
    assert len(ranked) == 3
    assert [o.score for o in ranked] == sorted(o.score for o in ranked)
    downs = np.array([o.matrix[:3, :3].T @ DOWN for o in ranked])
    dots = downs @ downs.T
    np.fill_diagonal(dots, -1.0)
    assert dots.max() < np.cos(np.radians(orient.DEDUPE_ANGLE_DEG))


def test_flat_base_bonus_is_recorded_and_only_on_the_base(mini, params):
    ranked = orient.orientations(mini, params, top_k=5)
    by_label = {o.label: o for o in ranked}
    assert "flat-base" in by_label
    assert by_label["flat-base"].terms["bonus"] == pytest.approx(-orient.FLAT_BASE_BONUS)
    for label, o in by_label.items():
        if label != "flat-base":
            assert o.terms["bonus"] == 0.0


def test_apply_matches_the_matrix(mini, params):
    best = orient.best_orientation(mini, params)
    posed = orient.apply(mini, best)
    expected = mini.vertices @ best.matrix[:3, :3].T + best.matrix[:3, 3]
    assert posed.vertices.shape == mini.vertices.shape
    assert np.allclose(np.sort(posed.vertices, axis=0), np.sort(expected, axis=0), atol=1e-9)
    assert mini.bounds[0][2] != posed.bounds[0][2] or np.isclose(mini.bounds[0][2], 0.0)


def test_a_thin_sheet_still_produces_an_answer(params):
    """Degenerate-ish input must not raise; scoring falls back gracefully."""
    sheet = trimesh.creation.box(extents=[20.0, 20.0, 0.2])
    best = orient.best_orientation(sheet, params)
    assert np.isfinite(best.score)
    posed = orient.apply(sheet, best)
    assert np.isclose(posed.bounds[0][2], 0.0)


def test_island_term_is_zero_without_the_overhang_module(mini, params):
    """`overhang.find_islands` is optional; its absence must not break scoring."""
    _, terms = orient.score_orientation(
        mini, orient._pose_matrix(mini, DOWN), params, islands=True
    )
    assert terms["island_count"] >= 0.0
    if orient.find_islands is None:
        assert terms["island_count"] == 0.0


def test_island_term_is_wired_to_find_islands(mini, params, monkeypatch):
    """Stand in for `overhang.find_islands` and check the term actually moves."""
    calls = []

    class FakeIsland:
        z = 1.0
        centroid = np.zeros(3)
        area = 1.0

    def fake_find_islands(mesh, layer_height, min_area=0.05):
        calls.append((layer_height, min_area))
        return [FakeIsland()] * 5

    monkeypatch.setattr(orient, "find_islands", fake_find_islands)
    _, terms = orient.score_orientation(mini, orient._pose_matrix(mini, DOWN), params)
    assert terms["island_count"] > 0.0
    assert calls and calls[0][0] >= params.support_layer_height
    assert calls[0][1] == params.island_min_area

    # And a module that blows up must not take orientation down with it.
    monkeypatch.setattr(
        orient, "find_islands", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("wip"))
    )
    _, terms = orient.score_orientation(mini, orient._pose_matrix(mini, DOWN), params)
    assert terms["island_count"] == 0.0
