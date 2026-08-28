"""The serverless front end, checked against the one it has to match.

`rsupport.web.browser` exists so the app can run with no host behind it. Its
whole value depends on it answering the same routes the same way as
`rsupport.web.app`, because `app.js` drives both and only swaps its transport.
So most of this file asserts the two agree rather than asserting the browser
router works in isolation — a router that works but disagrees is the failure
that would actually bite.
"""

from __future__ import annotations

import io

import pytest
import trimesh
from fastapi.testclient import TestClient

from rsupport.web.app import app
from rsupport.web.browser import Router


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def router():
    return Router()


@pytest.fixture
def stl_bytes():
    mesh = trimesh.creation.box(extents=[10, 10, 20])
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


def _upload(router, data, name="cube.stl"):
    r = router.route("POST", "/api/model", data=data, filename=name)
    assert r.ok, r.data
    return r.data["id"]


def test_presets_match_the_served_ones(client, router):
    """The control panel is built from this, so a difference here is a
    different UI depending on where the app is running."""
    served = client.get("/api/presets").json()
    local = router.route("GET", "/api/presets").data
    assert local == served


def test_upload_returns_a_session_and_summary(router, stl_bytes):
    r = router.route("POST", "/api/model", data=stl_bytes, filename="cube.stl")
    assert r.ok
    assert r.data["id"]
    assert r.data["summary"]["faces"] == 12


def test_the_pipeline_runs_end_to_end_with_no_server(router, stl_bytes):
    sid = _upload(router, stl_bytes)

    pts = router.route("POST", f"/api/points/{sid}", body={})
    assert pts.ok and pts.data["points"]

    built = router.route("POST", f"/api/supports/{sid}", body={})
    assert built.ok and built.data["faces"] > 0

    mesh = router.route("GET", f"/api/mesh/{sid}/supports")
    assert mesh.ok and mesh.body[:5] not in (b"", None)

    out = router.route("GET", f"/api/export/{sid}?mode=3mf")
    assert out.ok
    assert out.filename.endswith(".3mf")
    assert out.body[:2] == b"PK", "a 3MF is a zip"


def test_a_summary_matches_the_served_one(client, router, stl_bytes):
    """Same file, same numbers, whichever front end read it."""
    served = client.post(
        "/api/model", files={"file": ("cube.stl", stl_bytes, "model/stl")}
    ).json()["summary"]
    local = router.route("POST", "/api/model", data=stl_bytes, filename="cube.stl").data["summary"]
    assert local == served


def test_the_lift_moves_the_model_here_too(router, stl_bytes):
    """`lift_height` is applied by the shared core, not by either front end.
    If that ever stops being true, this is where it shows."""
    sid = _upload(router, stl_bytes)
    for lift in (0.0, 7.5):
        router.route("POST", f"/api/points/{sid}", body={"overrides": {"lift_height": lift}})
        stl = router.route("GET", f"/api/mesh/{sid}/model").body
        low = trimesh.load(io.BytesIO(stl), file_type="stl").bounds[0][2]
        assert low == pytest.approx(lift, abs=1e-4)


def test_garbage_upload_is_a_status_not_an_exception(router):
    """A browser build has no fastapi to turn a raise into a response, so the
    router has to do it — a traceback here would kill the worker."""
    r = router.route("POST", "/api/model", data=b"not an stl", filename="bad.stl")
    assert r.status == 400
    assert "could not read" in r.data["detail"]


def test_unknown_session_is_404(router):
    assert router.route("POST", "/api/points/nope", body={}).status == 404
    assert router.route("GET", "/api/mesh/nope/model").status == 404


def test_supports_geometry_absent_until_built(router, stl_bytes):
    sid = _upload(router, stl_bytes)
    assert router.route("GET", f"/api/mesh/{sid}/supports").status == 404


def test_an_unknown_route_is_404_not_a_crash(router):
    assert router.route("GET", "/api/nonsense").status == 404
    assert router.route("DELETE", "/api/presets").status == 404


def test_a_wrong_method_is_405(router, stl_bytes):
    sid = _upload(router, stl_bytes)
    assert router.route("GET", f"/api/points/{sid}").status == 405


def test_overhang_faces_match_the_served_ones(client, router, stl_bytes):
    """The overlay is read-only, but it is still one endpoint served two ways —
    same session-worth of geometry, same overhang mask either front end."""
    served_sid = client.post(
        "/api/model", files={"file": ("cube.stl", stl_bytes, "model/stl")}
    ).json()["id"]
    local_sid = _upload(router, stl_bytes)

    served = client.get(f"/api/overhang/{served_sid}").json()
    local = router.route("GET", f"/api/overhang/{local_sid}").data
    assert local == served


def test_overhang_faces_match_the_served_ones_at_an_explicit_angle(client, router, stl_bytes):
    served_sid = client.post(
        "/api/model", files={"file": ("cube.stl", stl_bytes, "model/stl")}
    ).json()["id"]
    local_sid = _upload(router, stl_bytes)

    served = client.get(f"/api/overhang/{served_sid}?angle_deg=10").json()
    local = router.route("GET", f"/api/overhang/{local_sid}?angle_deg=10").data
    assert local == served
    assert local["angle_deg"] == pytest.approx(10.0)


def test_overhang_of_unknown_session_is_404(router):
    assert router.route("GET", "/api/overhang/nope").status == 404


def test_a_setting_this_build_has_never_heard_of_is_reported(router, stl_bytes):
    sid = _upload(router, stl_bytes)
    router.route("POST", f"/api/points/{sid}", body={})
    body = router.route(
        "POST", f"/api/supports/{sid}", body={"overrides": {"not_a_real_setting": 3}}
    ).data
    assert any("not_a_real_setting" in w for w in body["warnings"])
