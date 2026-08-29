"""Multi-model bookkeeping: a workspace groups the sessions one tab has
uploaded and remembers which one is active.

Every pipeline route still takes a model's own sid directly and touches
nothing else — a workspace only tracks which sids belong together and which
one a click in the viewer most recently chose. These tests pin that: multiple
models can live in one workspace, the active one can be switched, and running
a stage on one model's sid never disturbs another model's session.
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


def _box_bytes(extents):
    mesh = trimesh.creation.box(extents=extents)
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


@pytest.fixture
def cube_bytes():
    return _box_bytes([10, 10, 20])


@pytest.fixture
def slab_bytes():
    return _box_bytes([30, 6, 6])


# --------------------------------------------------------------- http (app.py)


def _create_ws(client):
    r = client.post("/api/workspace")
    assert r.status_code == 200
    return r.json()["id"]


def _add_model(client, wid, data, name):
    r = client.post(f"/api/workspace/{wid}/model", files={"file": (name, data, "model/stl")})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_workspace_holds_more_than_one_model(client, cube_bytes, slab_bytes):
    wid = _create_ws(client)
    body = _add_model(client, wid, cube_bytes, "cube.stl")
    assert len(body["models"]) == 1
    body = _add_model(client, wid, slab_bytes, "slab.stl")
    assert len(body["models"]) == 2
    names = {m["name"] for m in body["models"]}
    assert names == {"cube.stl", "slab.stl"}


def test_the_most_recent_upload_is_active(client, cube_bytes, slab_bytes):
    wid = _create_ws(client)
    cube = _add_model(client, wid, cube_bytes, "cube.stl")
    assert cube["active"] == cube["models"][0]["id"]
    slab = _add_model(client, wid, slab_bytes, "slab.stl")
    slab_id = slab["models"][1]["id"]
    assert slab["active"] == slab_id


def test_switching_the_active_model(client, cube_bytes, slab_bytes):
    wid = _create_ws(client)
    cube = _add_model(client, wid, cube_bytes, "cube.stl")
    cube_id = cube["models"][0]["id"]
    slab = _add_model(client, wid, slab_bytes, "slab.stl")
    assert slab["active"] != cube_id

    r = client.post(f"/api/workspace/{wid}/active", json={"sid": cube_id})
    assert r.status_code == 200
    assert r.json()["active"] == cube_id

    # And it sticks — a plain GET reports the same thing back.
    assert client.get(f"/api/workspace/{wid}").json()["active"] == cube_id


def test_switching_to_a_model_not_in_the_workspace_is_404(client, cube_bytes):
    wid = _create_ws(client)
    _add_model(client, wid, cube_bytes, "cube.stl")
    r = client.post(f"/api/workspace/{wid}/active", json={"sid": "not-a-real-sid"})
    assert r.status_code == 404


def test_unknown_workspace_is_404(client):
    assert client.get("/api/workspace/nope").status_code == 404
    assert client.post("/api/workspace/nope/active", json={"sid": "x"}).status_code == 404


def test_rotating_one_model_does_not_touch_another(client, cube_bytes, slab_bytes):
    """The whole point of a workspace is that each model is still its own,
    fully independent Session — switching which one is "active" is UI
    bookkeeping, not a gate the pipeline passes through."""
    wid = _create_ws(client)
    cube = _add_model(client, wid, cube_bytes, "cube.stl")
    cube_id = cube["models"][0]["id"]
    slab = _add_model(client, wid, slab_bytes, "slab.stl")
    slab_id = slab["models"][1]["id"]

    before_cube = client.get(f"/api/mesh/{cube_id}/model").content
    before_slab = client.get(f"/api/mesh/{slab_id}/model").content

    r = client.post(f"/api/rotate/{slab_id}", json={"rx": 0, "ry": 90, "rz": 0})
    assert r.status_code == 200

    # The rotated model actually changed …
    after_slab = client.get(f"/api/mesh/{slab_id}/model").content
    assert after_slab != before_slab
    # … and the other model in the same workspace is untouched.
    after_cube = client.get(f"/api/mesh/{cube_id}/model").content
    assert after_cube == before_cube


def test_build_operates_only_on_the_named_model(client, cube_bytes, slab_bytes):
    wid = _create_ws(client)
    cube = _add_model(client, wid, cube_bytes, "cube.stl")
    cube_id = cube["models"][0]["id"]
    slab = _add_model(client, wid, slab_bytes, "slab.stl")
    slab_id = slab["models"][1]["id"]

    client.post(f"/api/points/{cube_id}", json={})
    client.post(f"/api/supports/{cube_id}", json={})

    # The slab's own session never had a stage run on it at all.
    assert client.get(f"/api/mesh/{slab_id}/supports").status_code == 404
    assert client.get(f"/api/mesh/{cube_id}/supports").status_code == 200


# ------------------------------------------------------------ router (browser.py)


def _create_ws_r(router):
    r = router.route("POST", "/api/workspace")
    assert r.ok, r.data
    return r.data["id"]


def _add_model_r(router, wid, data, name):
    r = router.route("POST", f"/api/workspace/{wid}/model", data=data, filename=name)
    assert r.ok, r.data
    return r.data


def test_workspace_payload_matches_between_front_ends(client, router, cube_bytes, slab_bytes):
    wid_http = _create_ws(client)
    served = _add_model(client, wid_http, cube_bytes, "cube.stl")

    wid_local = _create_ws_r(router)
    local = _add_model_r(router, wid_local, cube_bytes, "cube.stl")

    # ids are per-run UUIDs, so compare shapes rather than exact values.
    assert set(served) == set(local)
    assert served["models"][0]["name"] == local["models"][0]["name"]
    assert served["models"][0]["summary"] == local["models"][0]["summary"]


def test_the_router_agrees_switching_is_possible_with_no_server(router, cube_bytes, slab_bytes):
    wid = _create_ws_r(router)
    cube = _add_model_r(router, wid, cube_bytes, "cube.stl")
    cube_id = cube["models"][0]["id"]
    _add_model_r(router, wid, slab_bytes, "slab.stl")

    r = router.route("POST", f"/api/workspace/{wid}/active", body={"sid": cube_id})
    assert r.ok
    assert r.data["active"] == cube_id


def test_router_rejects_unknown_workspace_as_a_status_not_a_crash(router):
    assert router.route("GET", "/api/workspace/nope").status == 404
    r = router.route("POST", "/api/workspace/nope/model", data=b"whatever", filename="x.stl")
    assert r.status == 404


def test_router_rejects_wrong_method_on_workspace_routes(router, cube_bytes):
    wid = _create_ws_r(router)
    model = _add_model_r(router, wid, cube_bytes, "cube.stl")
    sid = model["models"][0]["id"]
    assert router.route("GET", f"/api/workspace/{wid}/model").status == 405
    assert router.route("GET", f"/api/workspace/{wid}/active").status == 405
    assert router.route("POST", f"/api/workspace/{wid}", body={"sid": sid}).status == 405
