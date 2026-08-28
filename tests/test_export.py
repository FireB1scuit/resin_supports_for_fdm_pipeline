"""Export round-trips.

The 3MF path is hand-written XML, so it is worth proving a real slicer-side
library can read it back with both objects intact and correctly named.
"""

import zipfile

import pytest
import trimesh

from rsupport import export, presets


@pytest.fixture
def parts():
    model = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    model.apply_translation([0, 0, 10])
    sup = trimesh.creation.cylinder(radius=0.6, height=10.0)
    sup.apply_translation([0, 0, 5])
    return model, sup


def test_combined_welds_both(parts, tmp_path):
    model, sup = parts
    out = export.export_combined(model, sup, tmp_path / "a.stl")
    reloaded = trimesh.load(out, force="mesh")
    assert len(reloaded.faces) == len(model.faces) + len(sup.faces)


def test_separate_writes_two_files(parts, tmp_path):
    model, sup = parts
    written = export.export_separate(model, sup, tmp_path / "b.stl")
    assert [p.name for p in written] == ["b_model.stl", "b_supports.stl"]
    assert all(p.exists() for p in written)


def test_3mf_has_two_named_objects(parts, tmp_path):
    model, sup = parts
    params = presets.get("mini_0.2")
    out = export.export_3mf(model, sup, tmp_path / "c.3mf", params)

    scene = trimesh.load(out)
    counts = {name: len(geom.faces) for name, geom in scene.geometry.items()}
    assert counts == {"miniature": len(model.faces), "supports": len(sup.faces)}


def test_3mf_carries_per_object_layer_height(parts, tmp_path):
    """The whole reason for the 3MF path: supports slice coarser than the model."""
    model, sup = parts
    params = presets.get("mini_0.2")
    out = export.export_3mf(model, sup, tmp_path / "d.3mf", params)

    with zipfile.ZipFile(out) as z:
        config = z.read("Metadata/Slic3r_PE_model.config").decode()

    assert f'key="layer_height" value="{params.layer_height:g}"' in config
    assert f'key="layer_height" value="{params.support_layer_height:g}"' in config
    assert params.support_layer_height > params.layer_height


def test_export_dispatches_on_extension(parts, tmp_path):
    model, sup = parts
    params = presets.get("mini_0.2")
    assert export.export(model, sup, tmp_path / "e.3mf", params)[0].suffix == ".3mf"
    assert export.export(model, sup, tmp_path / "e.stl", params)[0].suffix == ".stl"


def test_plate_marker_cube_sits_on_the_plate_at_the_models_low_corner(parts):
    model, _ = parts
    cube = export.plate_marker_cube(model)

    assert cube.extents == pytest.approx([export.MARKER_CUBE_SIZE] * 3)
    lo, hi = cube.bounds
    model_lo, _ = model.bounds
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(export.MARKER_CUBE_SIZE)
    assert lo[0] == pytest.approx(model_lo[0])
    assert lo[1] == pytest.approx(model_lo[1])


def test_plate_marker_cube_follows_a_rotated_models_own_corner(parts):
    """Placed off the model's own bounding box rather than a fixed coordinate,
    so a model that arrived rotated still gets a marker at its actual corner."""
    model, _ = parts
    rotated = model.copy()
    rotated.apply_transform(
        trimesh.transformations.euler_matrix(0.4, 0.9, 1.7, axes="sxyz")
    )
    rotated.apply_translation([0, 0, -rotated.bounds[0][2]])  # back onto the plate

    cube = export.plate_marker_cube(rotated)
    lo, _ = cube.bounds
    model_lo, _ = rotated.bounds
    assert lo[0] == pytest.approx(model_lo[0])
    assert lo[1] == pytest.approx(model_lo[1])


def test_separate_export_adds_the_marker_cube_to_the_model_file_only(parts, tmp_path):
    model, sup = parts
    written = export.export_separate(model, sup, tmp_path / "g.stl", marker_cube=True)
    model_out, sup_out = (trimesh.load(p, force="mesh") for p in written)

    assert len(model_out.faces) == len(model.faces) + 12  # + one box-shaped cube
    assert len(sup_out.faces) == len(sup.faces)


def test_separate_export_without_the_flag_adds_no_cube(parts, tmp_path):
    model, sup = parts
    written = export.export_separate(model, sup, tmp_path / "h.stl")
    model_out = trimesh.load(written[0], force="mesh")
    assert len(model_out.faces) == len(model.faces)


def test_export_dispatch_threads_the_marker_cube_through_separate_mode(parts, tmp_path):
    model, sup = parts
    params = presets.get("mini_0.2")
    written = export.export(
        model, sup, tmp_path / "i.stl", params, mode="separate", marker_cube=True
    )
    model_out = trimesh.load(written[0], force="mesh")
    assert len(model_out.faces) == len(model.faces) + 12


def test_export_handles_empty_supports(parts, tmp_path):
    """A model that needs no supports must still export cleanly."""
    model, _ = parts
    empty = trimesh.Trimesh()
    params = presets.get("mini_0.2")

    combined = trimesh.load(export.export_combined(model, empty, tmp_path / "f.stl"), force="mesh")
    assert len(combined.faces) == len(model.faces)

    scene = trimesh.load(export.export_3mf(model, empty, tmp_path / "f.3mf", params))
    assert list(scene.geometry) == ["miniature"]
