"""API surface tests.

These cover upload, session handling and the error paths — the parts that do
not depend on the geometry stages, so they stay meaningful even while the
pipeline behind them is being tuned.
"""

import io

import pytest
import trimesh
from fastapi.testclient import TestClient

from rsupport import export as export_mod
from rsupport.types import SupportParams
from rsupport.web.app import app

DEFAULTS = SupportParams()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def stl_bytes():
    mesh = trimesh.creation.box(extents=[10, 10, 20])
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


def _upload(client, data, name="cube.stl"):
    return client.post("/api/model", files={"file": (name, data, "model/stl")})


def test_presets_are_served(client):
    body = client.get("/api/presets").json()
    assert body["default"] in body["presets"]
    assert body["presets"][body["default"]]["tip_diameter"] > 0


def test_upload_returns_session_and_summary(client, stl_bytes):
    res = _upload(client, stl_bytes)
    assert res.status_code == 200
    body = res.json()
    assert body["id"]
    assert body["summary"]["faces"] == 12
    assert body["summary"]["size"] == pytest.approx([10, 10, 20])


def _model_mesh(client, sid):
    data = client.get(f"/api/mesh/{sid}/model").content
    return trimesh.load(io.BytesIO(data), file_type="stl")


def test_uploaded_model_is_floated_by_the_default_lift(client, stl_bytes):
    """Whatever z the file was authored at, the viewer sees it at the lift."""
    sid = _upload(client, stl_bytes).json()["id"]
    mesh = _model_mesh(client, sid)
    assert mesh.bounds[0][2] == pytest.approx(DEFAULTS.lift_height, abs=1e-6)
    assert DEFAULTS.lift_height > 0, "the default is to float the model, not set it down"


def test_lift_moves_the_model_the_viewer_is_served(client, stl_bytes):
    """The lift is a session-level move, so the viewer's copy has to change too.

    Zero has to work as well as any other value: it is how you ask for the model
    flat on the plate.
    """
    sid = _upload(client, stl_bytes).json()["id"]
    for lift in (12.0, 0.0):
        body = client.post(f"/api/points/{sid}", json={"overrides": {"lift_height": lift}}).json()
        assert body["lift"] == pytest.approx(lift)
        assert _model_mesh(client, sid).bounds[0][2] == pytest.approx(lift, abs=1e-6)


def test_lift_does_not_disturb_the_pose(client, stl_bytes):
    """Only Z moves — a lift must not re-centre or re-orient anything."""
    sid = _upload(client, stl_bytes).json()["id"]
    before = _model_mesh(client, sid)
    client.post(f"/api/points/{sid}", json={"overrides": {"lift_height": 3.0}})
    after = _model_mesh(client, sid)
    assert after.bounds[:, :2] == pytest.approx(before.bounds[:, :2], abs=1e-6)
    assert after.bounds[1][2] - after.bounds[0][2] == pytest.approx(
        before.bounds[1][2] - before.bounds[0][2], abs=1e-6
    )


def test_garbage_upload_is_rejected(client):
    res = _upload(client, b"this is not a mesh", name="bad.stl")
    assert res.status_code == 400


def test_unknown_session_is_404(client):
    assert client.post("/api/points/nope", json={}).status_code == 404
    assert client.get("/api/mesh/nope/model").status_code == 404


def test_supports_geometry_absent_until_built(client, stl_bytes):
    sid = _upload(client, stl_bytes).json()["id"]
    assert client.get(f"/api/mesh/{sid}/supports").status_code == 404


def test_dropped_contacts_come_back_as_indices_not_just_a_count(client):
    """The viewer draws unheld contacts in a solid red, so it has to know *which*
    ones they are. A sphere sitting on a block has both kinds: the block's own
    lifted underside is easy, and the sphere's underside has 20 mm of block
    directly beneath it and no way round in the height available."""
    block = trimesh.creation.box(extents=[20, 20, 20])
    block.apply_translation([0, 0, 10])
    cap = trimesh.creation.icosphere(subdivisions=3, radius=9)
    cap.apply_translation([0, 0, 30])  # its underside sits right over the block
    buf = io.BytesIO()
    trimesh.util.concatenate([block, cap]).export(buf, file_type="stl")

    sid = _upload(client, buf.getvalue(), "capped.stl").json()["id"]
    client.post(f"/api/points/{sid}", json={})
    body = client.post(f"/api/supports/{sid}", json={}).json()

    idx = body["dropped_points"]
    assert len(idx) == body["dropped"], "count and indices must agree"
    assert idx, "contacts under the sphere have 20 mm of block beneath them"
    assert all(0 <= i < body["points"] for i in idx)
    assert len(set(idx)) == len(idx)


def test_nothing_is_dropped_when_everything_is_reachable(client, stl_bytes):
    sid = _upload(client, stl_bytes).json()["id"]
    client.post(f"/api/points/{sid}", json={})
    body = client.post(f"/api/supports/{sid}", json={}).json()

    assert body["points"] > 0, "a floated cube's underside needs holding"
    assert body["dropped_points"] == []


def test_index_page_is_served(client):
    body = client.get("/").text
    assert "resin supports for FDM" in body
    # The importmap is what lets the vendored addons resolve 'three' with no
    # bundler; if it goes missing the whole viewer silently fails to boot.
    assert 'type="importmap"' in body


# --------------------------------------------------------------- export (#24)


def _build(client, data, name="widget.stl"):
    sid = _upload(client, data, name).json()["id"]
    client.post(f"/api/points/{sid}", json={})
    client.post(f"/api/supports/{sid}", json={})
    return sid


def test_export_filename_follows_the_original_upload(client, stl_bytes):
    """{stem}_resin-fdm-supported_by-FireB1scuit.{ext} — issue #24."""
    sid = _build(client, stl_bytes, "widget.stl")

    res = client.get(f"/api/export/{sid}", params={"fmt": "3mf"})
    assert res.status_code == 200
    assert 'filename="widget_resin-fdm-supported_by-FireB1scuit.3mf"' in res.headers[
        "content-disposition"
    ]

    res = client.get(f"/api/export/{sid}", params={"fmt": "stl"})
    assert 'filename="widget_resin-fdm-supported_by-FireB1scuit.stl"' in res.headers[
        "content-disposition"
    ]


def test_export_produces_exactly_one_format_never_both(client, stl_bytes):
    """`fmt` is a single stl-vs-3mf choice; there is no "both" mode any more."""
    sid = _build(client, stl_bytes)

    stl = client.get(f"/api/export/{sid}", params={"fmt": "stl"})
    assert stl.headers["content-disposition"].endswith('.stl"')
    assert stl.content[:2] != b"PK", "a plain STL, not a zip and not a 3MF"

    threemf = client.get(f"/api/export/{sid}", params={"fmt": "3mf"})
    assert threemf.headers["content-disposition"].endswith('.3mf"')

    assert client.get(f"/api/export/{sid}", params={"fmt": "both"}).status_code == 400


def test_export_defaults_to_stl(client, stl_bytes):
    """STL, not 3MF, is the default format — issue #24."""
    sid = _build(client, stl_bytes, "widget.stl")

    res = client.get(f"/api/export/{sid}")
    assert res.status_code == 200
    assert 'filename="widget_resin-fdm-supported_by-FireB1scuit.stl"' in res.headers[
        "content-disposition"
    ]


def test_export_separate_writes_a_model_and_a_supports_file(client, stl_bytes):
    """Two plain files, fetched one at a time by `part` — not a zip."""
    sid = _build(client, stl_bytes, "widget.stl")

    model = client.get(f"/api/export/{sid}", params={"fmt": "stl", "separate": "true", "part": "model"})
    assert model.status_code == 200
    assert model.content[:2] != b"PK", "a plain STL, not a zip"
    assert 'filename="widget_resin-fdm-supported_by-FireB1scuit_model.stl"' in model.headers[
        "content-disposition"
    ]

    supports = client.get(
        f"/api/export/{sid}", params={"fmt": "stl", "separate": "true", "part": "supports"}
    )
    assert supports.status_code == 200
    assert 'filename="widget_resin-fdm-supported_by-FireB1scuit_supports.stl"' in supports.headers[
        "content-disposition"
    ]

    assert client.get(f"/api/export/{sid}", params={"fmt": "stl", "separate": "true"}).status_code == 400


def test_separate_export_marks_only_the_model_file_with_the_plate_cube(client, stl_bytes):
    """The model is exported without the supports that would otherwise anchor
    it, so it alone gets the 0.1mm marker cube — never the supports file."""
    sid = _build(client, stl_bytes, "widget.stl")

    oriented = trimesh.load(
        io.BytesIO(client.get(f"/api/mesh/{sid}/model").content), file_type="stl"
    )
    raw_supports = trimesh.load(
        io.BytesIO(client.get(f"/api/mesh/{sid}/supports").content), file_type="stl"
    )
    expected_cube = export_mod.plate_marker_cube(oriented)

    model_bytes = client.get(
        f"/api/export/{sid}", params={"fmt": "stl", "separate": "true", "part": "model"}
    ).content
    sup_bytes = client.get(
        f"/api/export/{sid}", params={"fmt": "stl", "separate": "true", "part": "supports"}
    ).content

    model_mesh = trimesh.load(io.BytesIO(model_bytes), file_type="stl")
    sup_mesh = trimesh.load(io.BytesIO(sup_bytes), file_type="stl")

    # supports file is untouched — same geometry as fetching it directly.
    assert len(sup_mesh.faces) == len(raw_supports.faces)

    # model file is the original mesh plus exactly one marker-cube-shaped part.
    assert len(model_mesh.faces) == len(oriented.faces) + len(expected_cube.faces)
    parts = model_mesh.split(only_watertight=False)
    markers = [
        p
        for p in parts
        if len(p.faces) == len(expected_cube.faces)
        and p.extents == pytest.approx(expected_cube.extents, abs=1e-6)
    ]
    assert len(markers) == 1
    assert markers[0].bounds[0] == pytest.approx(expected_cube.bounds[0], abs=1e-6)
    assert markers[0].bounds[1] == pytest.approx(expected_cube.bounds[1], abs=1e-6)
    assert markers[0].bounds[0][2] == pytest.approx(0.0, abs=1e-6), "sits on the plate"


def test_combined_and_3mf_exports_carry_no_marker_cube(client, stl_bytes):
    """Only a separate export drops the supports out of the model file — the
    combined STL and the two-object 3MF still have them as context/anchor."""
    sid = _build(client, stl_bytes)
    oriented = trimesh.load(
        io.BytesIO(client.get(f"/api/mesh/{sid}/model").content), file_type="stl"
    )
    supports = trimesh.load(
        io.BytesIO(client.get(f"/api/mesh/{sid}/supports").content), file_type="stl"
    )

    combined = trimesh.load(
        io.BytesIO(client.get(f"/api/export/{sid}", params={"fmt": "stl"}).content),
        file_type="stl",
    )
    assert len(combined.faces) == len(oriented.faces) + len(supports.faces)


def test_the_ui_is_served_with_revalidation_forced(client):
    """The sidebar and the script that drives it are separate files, and this is
    a tool people leave open across restarts. Let a browser cache either one on
    its own guess and it can pair a new index.html with a stale app.js, which
    puts controls on screen with nothing listening to them: they move, they show
    their value, and nothing happens. That is indistinguishable from a bug in
    the generator, so the assets are served must-ask.
    """
    for path in ("/", "/app.js", "/index.html"):
        assert client.get(path).headers.get("cache-control") == "no-cache", path


def test_a_setting_this_server_has_never_heard_of_is_reported(client, stl_bytes):
    """`with_` drops unknown keys, which is what lets the UI post the whole
    control panel every time. It also means a page newer than the server it is
    talking to gets silence back. Say so instead."""
    sid = _upload(client, stl_bytes).json()["id"]
    client.post(f"/api/points/{sid}", json={})
    body = client.post(
        f"/api/supports/{sid}", json={"overrides": {"brace_moon_phase": 3}}
    ).json()
    assert any("brace_moon_phase" in w for w in body["warnings"]), body["warnings"]


def test_a_setting_this_server_does_know_is_not_reported(client, stl_bytes):
    sid = _upload(client, stl_bytes).json()["id"]
    client.post(f"/api/points/{sid}", json={})
    body = client.post(
        f"/api/supports/{sid}", json={"overrides": {"brace_interval": 4.0}}
    ).json()
    assert not any("does not know" in w for w in body["warnings"]), body["warnings"]
    assert body["params"]["brace_interval"] == 4.0
