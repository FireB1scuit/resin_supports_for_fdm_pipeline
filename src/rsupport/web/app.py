"""The local web app: HTTP in front of :mod:`rsupport.web.core`.

Geometry stays in Python; the browser is a viewer and a control panel. Sessions
live in memory for the life of the process — this is a single-user tool you run
on your own machine, so there is no database and no persistence, and closing it
throws the work away.

Every endpoint here is a translation layer and nothing more: unpack the request,
call the matching function in ``core``, turn a :class:`~rsupport.web.core.StageError`
into an ``HTTPException``. The logic itself is shared with the browser build
(:mod:`rsupport.web.browser`), which runs the same stages with no server at all.
Put behaviour in ``core``, not here, or the two front ends will drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core
from .core import SessionStore, StageError

STATIC = Path(__file__).parent / "static"

#: Kept module-level and named as it was: `tests/test_web.py` builds a fresh
#: TestClient per test against this one app instance.
_store = SessionStore()

MAX_SESSIONS = core.MAX_SESSIONS


def _get(sid: str):
    try:
        return _store.get(sid)
    except StageError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


# ----------------------------------------------------------------- payloads


class ParamPatch(BaseModel):
    preset: str | None = None
    nozzle: float | None = None
    overrides: dict[str, Any] = {}

    def patch(self) -> dict:
        return {"preset": self.preset, "nozzle": self.nozzle, "overrides": self.overrides}


class OrientRequest(ParamPatch):
    pick: int = 0
    skip: bool = False  # keep the model's existing pose


class RotateRequest(ParamPatch):
    #: Degrees, always applied fresh from the file's own pose — not
    #: accumulated — so the sliders stay an absolute, reversible readout.
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


class PointsRequest(ParamPatch):
    pass


class SupportsRequest(ParamPatch):
    points: list[dict] | None = None  # None = use whatever stage 2 produced


# --------------------------------------------------------------------- app

app = FastAPI(title="resin supports for FDM", docs_url="/api/docs")


def _staged(fn, *args, **kwargs):
    """Run a core stage, turning its failure into fastapi's."""
    try:
        return fn(*args, **kwargs)
    except StageError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@app.get("/api/presets")
def api_presets() -> dict:
    return core.presets_payload()


@app.post("/api/model")
def api_model(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    session = _staged(core.load_model, data, file.filename)
    _store.put(session)
    return core.model_payload(session, file.filename)


@app.post("/api/orient/{sid}")
def api_orient(sid: str, req: OrientRequest) -> dict:
    return _staged(core.run_orient, _get(sid), pick=req.pick, skip=req.skip, **req.patch())


@app.post("/api/rotate/{sid}")
def api_rotate(sid: str, req: RotateRequest) -> dict:
    return _staged(core.run_rotate, _get(sid), rx=req.rx, ry=req.ry, rz=req.rz, **req.patch())


@app.post("/api/points/{sid}")
def api_points(sid: str, req: PointsRequest) -> dict:
    return _staged(core.run_points, _get(sid), **req.patch())


@app.post("/api/supports/{sid}")
def api_supports(sid: str, req: SupportsRequest) -> dict:
    return _staged(core.run_supports, _get(sid), points=req.points, **req.patch())


@app.get("/api/overhang/{sid}")
def api_overhang(sid: str, angle_deg: float | None = None) -> dict:
    return _staged(core.overhang_faces, _get(sid), angle_deg)


@app.get("/api/mesh/{sid}/{which}")
def api_mesh(sid: str, which: str) -> Response:
    data = _staged(core.mesh_stl, _get(sid), which)
    return Response(data, media_type="model/stl")


@app.get("/api/export/{sid}")
def api_export(sid: str, fmt: str = "stl", separate: bool = False, part: str | None = None) -> Response:
    name, data = _staged(core.export_bytes, _get(sid), fmt=fmt, separate=separate, part=part)
    return Response(
        data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


class _RevalidatingStatic(StaticFiles):
    """The UI, served so a browser always asks whether it is still current.

    This is a local tool people leave open across restarts, and the sidebar and
    the script that drives it are two separate files. Without a cache header a
    browser may apply its own freshness guess to each of them independently, so
    it is entitled to take the new index.html and keep the old app.js — which
    puts controls on screen with no listeners attached to them. They move, they
    show their value, and nothing happens, which looks exactly like a bug in the
    generator and is impossible to guess at from the outside.

    ``no-cache`` does not stop the file being cached, only being *used* without
    asking. Every response still carries an etag, so the usual answer is a 304
    and the cost is one conditional request per file per load.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", _RevalidatingStatic(directory=STATIC, html=True), name="static")
