"""API surface tests.

These cover upload, session handling and the error paths — the parts that do
not depend on the geometry stages, so they stay meaningful even while the
pipeline behind them is being tuned.
"""

import io

import pytest
import trimesh
from fastapi.testclient import TestClient

from rsupport.web.app import app


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


def test_uploaded_model_is_dropped_to_the_bed(client, stl_bytes):
    """Whatever z the file was authored at, the viewer should see it on z=0."""
    sid = _upload(client, stl_bytes).json()["id"]
    data = client.get(f"/api/mesh/{sid}/model").content
    mesh = trimesh.load(io.BytesIO(data), file_type="stl")
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_garbage_upload_is_rejected(client):
    res = _upload(client, b"this is not a mesh", name="bad.stl")
    assert res.status_code == 400


def test_unknown_session_is_404(client):
    assert client.post("/api/points/nope", json={}).status_code == 404
    assert client.get("/api/mesh/nope/model").status_code == 404


def test_supports_geometry_absent_until_built(client, stl_bytes):
    sid = _upload(client, stl_bytes).json()["id"]
    assert client.get(f"/api/mesh/{sid}/supports").status_code == 404


def test_index_page_is_served(client):
    body = client.get("/").text
    assert "resin supports for FDM" in body
    # The importmap is what lets the vendored addons resolve 'three' with no
    # bundler; if it goes missing the whole viewer silently fails to boot.
    assert 'type="importmap"' in body
