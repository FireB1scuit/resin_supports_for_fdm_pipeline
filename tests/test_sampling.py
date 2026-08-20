"""Stage 2 tests — where supports touch the model.

Stage 2 outputs nothing but a list of SupportPoint, which is exactly what the
browser UI hands back after the user edits it. So these tests assert on the
list: that mandatory points survive, that spacing is honoured, that nothing is
placed where a pillar could not be built, and that a shape needing no support
gets none.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from rsupport import mesh_io, presets
from rsupport.sampling import SEVERITY_TIGHTEN, place_points, prune_points
from rsupport.types import SupportPoint

PARAMS = presets.from_nozzle(0.2)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def tee(bar_z: float = 15.0, span: float = 16.0) -> trimesh.Trimesh:
    """A stem carrying a wide bar: two big flat overhangs, clear air below."""
    stem = trimesh.creation.box(extents=[3.0, 3.0, bar_z])
    stem.apply_translation([0, 0, bar_z * 0.5])
    bar = trimesh.creation.box(extents=[span, 4.0, 2.0])
    bar.apply_translation([0, 0, bar_z + 1.0])
    return mesh_io.drop_to_bed(mesh_io.concat(stem, bar))


def cone_on_its_base(radius: float = 8.0, height: float = 20.0) -> trimesh.Trimesh:
    """Every surface leans inward. Nothing here needs holding up."""
    return mesh_io.drop_to_bed(trimesh.creation.cone(radius=radius, height=height, sections=48))


def floating_blob() -> trimesh.Trimesh:
    """A cube in mid-air beside a post — a genuine island, unreachable by
    growing upward from anything below it."""
    post = trimesh.creation.box(extents=[3, 3, 30])
    post.apply_translation([0, 0, 15])
    blob = trimesh.creation.box(extents=[4, 4, 4])
    blob.apply_translation([9, 0, 20])
    return mesh_io.drop_to_bed(mesh_io.concat(post, blob))


def positions(points) -> np.ndarray:
    if not points:
        return np.zeros((0, 3))
    return np.array([p.position for p in points])


# --------------------------------------------------------------------------- #
# what gets supported
# --------------------------------------------------------------------------- #


def test_a_flat_overhang_gets_points_underneath_it():
    model = tee()
    pts = place_points(model, PARAMS)
    assert pts, "a wide flat overhang must be supported"

    pos = positions(pts)
    # Everything should sit on the underside of the bar, not on the stem walls.
    assert np.all(pos[:, 2] < model.bounds[1][2])
    on_bar = np.abs(pos[:, 2] - 15.0) < 1.0
    assert on_bar.mean() > 0.8, "points should be on the bar underside"


def test_vertical_walls_are_not_supported():
    """A plain upright box has no overhang and must get nothing."""
    box = mesh_io.drop_to_bed(trimesh.creation.box(extents=[10, 10, 30]))
    assert place_points(box, PARAMS) == []


def test_a_cone_on_its_base_needs_no_support():
    assert place_points(cone_on_its_base(), PARAMS) == []


def test_normals_point_out_of_the_surface():
    pts = place_points(tee(), PARAMS)
    n = np.array([p.normal for p in pts])
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    # An overhang faces downward, so its outward normal has a negative z.
    assert (n[:, 2] < 0).mean() > 0.9


# --------------------------------------------------------------------------- #
# islands — the mandatory ones
# --------------------------------------------------------------------------- #


def test_an_island_gets_a_forced_point_at_the_right_height():
    model = floating_blob()
    pts = place_points(model, PARAMS)
    forced = [p for p in pts if p.forced]
    assert forced, "the floating cube starts in mid-air and must be forced"
    assert all(p.source == "island" for p in forced)

    zs = positions(forced)[:, 2]
    assert np.any(np.abs(zs - 18.0) < 1.0), f"expected a point at the cube's underside, got {zs}"


def test_forced_points_survive_pruning():
    model = floating_blob()
    pts = place_points(model, PARAMS)
    n_forced = sum(p.forced for p in pts)

    # A spacing far larger than the model would thin everything else away.
    kept = prune_points(pts, 1000.0)
    assert sum(p.forced for p in kept) == n_forced
    assert len(kept) <= len(pts)


def test_prune_respects_the_spacing_it_is_given():
    pts = [
        SupportPoint(position=np.array([x, 0.0, 10.0]), normal=np.array([0.0, 0.0, -1.0]))
        for x in np.linspace(0, 20, 41)  # 0.5 mm apart
    ]
    kept = prune_points(pts, 3.0)
    pos = positions(kept)
    d = cKDTree(pos).query(pos, k=2)[0][:, 1]
    assert d.min() >= 3.0 - 1e-6
    assert len(kept) < len(pts)


# --------------------------------------------------------------------------- #
# spacing and coverage
# --------------------------------------------------------------------------- #


def test_points_are_not_packed_tighter_than_the_spacing():
    """Blue-noise thinning, not a grid: no two ordinary points may crowd.

    Forced island points are exempt — they are placed regardless of who is
    nearby, because the alternative is a print that fails.
    """
    pts = [p for p in place_points(tee(), PARAMS) if not p.forced]
    pos = positions(pts)
    assert len(pos) > 4

    d = cKDTree(pos).query(pos, k=2)[0][:, 1]
    # Spacing tightens with overhang severity by design, bottoming out at
    # (1 - SEVERITY_TIGHTEN) for a face pointing straight down — which is
    # exactly what the bar's underside is. That floor is the real contract.
    floor = PARAMS.support_spacing * (1.0 - SEVERITY_TIGHTEN)
    assert d.min() >= floor - 1e-6


def test_nothing_is_left_further_than_the_unsupported_span():
    """A contact tip cannot bridge, so no part of an overhang may be stranded."""
    model = tee(span=24.0)
    pts = place_points(model, PARAMS)
    pos = positions(pts)

    # Sample the bar's underside and check each sample has a support near it.
    xs = np.linspace(-11.0, 11.0, 25)
    ys = np.linspace(-1.5, 1.5, 5)
    grid = np.array([[x, y, 15.0] for x in xs for y in ys if abs(x) > 2.0])
    d, _ = cKDTree(pos).query(grid)
    assert d.max() <= PARAMS.max_unsupported_span * 1.5, f"worst stranded point {d.max():.1f} mm"


def test_tighter_spacing_produces_more_points():
    model = tee()
    sparse = place_points(model, PARAMS.with_(support_spacing=6.0))
    dense = place_points(model, PARAMS.with_(support_spacing=2.0))
    assert len(dense) > len(sparse)


def test_a_shallower_overhang_threshold_supports_more():
    model = mesh_io.drop_to_bed(trimesh.creation.icosphere(subdivisions=3, radius=10.0))
    strict = place_points(model, PARAMS.with_(overhang_angle_deg=30.0))
    loose = place_points(model, PARAMS.with_(overhang_angle_deg=60.0))
    assert len(loose) >= len(strict)


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_points_round_trip_through_their_dict_form():
    """The UI ships this list to the browser and back as JSON."""
    pts = place_points(tee(), PARAMS)
    again = [SupportPoint.from_dict(p.as_dict()) for p in pts]
    assert len(again) == len(pts)
    assert np.allclose(positions(again), positions(pts))
    assert [p.forced for p in again] == [p.forced for p in pts]
    assert [p.source for p in again] == [p.source for p in pts]


def test_result_is_deterministic():
    model = tee()
    a = place_points(model, PARAMS)
    b = place_points(model, PARAMS)
    assert len(a) == len(b)
    assert np.allclose(positions(a), positions(b))


@pytest.mark.parametrize("preset", sorted(presets.PRESETS))
def test_every_preset_produces_a_usable_list(preset):
    params = presets.get(preset)
    pts = place_points(tee(), params)
    assert pts
    assert all(np.isfinite(p.position).all() for p in pts)
    assert all(np.isfinite(p.normal).all() for p in pts)


# --------------------------------------------------------------------------- #
# lifting the model off the plate
# --------------------------------------------------------------------------- #


def test_a_grounded_flat_bottom_is_not_an_overhang():
    """It is printed against glass, so it is the one downward face wanting
    nothing at all. Without this a cone standing on its base is the most heavily
    supported surface on the model."""
    box = mesh_io.drop_to_bed(trimesh.creation.box(extents=[10, 10, 20]))
    assert place_points(box, PARAMS.with_(lift_height=0.0)) == []


def test_a_lifted_flat_bottom_is_supported_like_any_other_overhang():
    """Once the model floats there is nothing under that face but air, and the
    plate is no longer an excuse to skip it."""
    lift = 5.0
    box = mesh_io.drop_to_bed(trimesh.creation.box(extents=[10, 10, 20]), lift=lift)
    points = place_points(box, PARAMS.with_(lift_height=lift))

    under = positions([p for p in points if p.position[2] < lift + 1e-6])
    assert len(under) > 8, "the whole underside wants holding, not a few corners"
    # And held across the footprint, not clustered at the rim.
    assert np.ptp(under[:, 0]) > 8.0
    assert np.ptp(under[:, 1]) > 8.0


def test_lifting_leaves_room_for_a_contact_tip():
    """`_supportable` measures clearance down to the floor beneath a point. The
    floor under a lifted model is the plate, not the model's own underside —
    reading it off the mesh rejects every point on the face that needs them."""
    lift = 3.0
    box = mesh_io.drop_to_bed(trimesh.creation.box(extents=[10, 10, 20]), lift=lift)
    assert lift > PARAMS.tip_length, "otherwise this proves nothing"
    assert place_points(box, PARAMS.with_(lift_height=lift))

