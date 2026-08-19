"""The local web app.

Geometry stays in Python; the browser is a viewer and a control panel. Sessions
live in memory for the life of the process — this is a single-user tool you run
on your own machine, so there is no database and no persistence, and closing it
throws the work away.

The three pipeline stages are separate endpoints on purpose. Changing a tip
diameter only needs stage 3 re-run, which takes a moment; changing the overhang
angle needs stages 2 and 3; changing the orientation needs all of it. The UI
knows which is which and calls accordingly.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import export as export_mod
from .. import mesh_io, presets
from ..types import SupportParams, SupportPoint

STATIC = Path(__file__).parent / "static"

# Sessions are dropped oldest-first past this, so a long-running server does not
# accumulate a gigabyte of abandoned uploads.
MAX_SESSIONS = 8


@dataclass
class Session:
    id: str
    original: trimesh.Trimesh
    params: SupportParams
    oriented: trimesh.Trimesh | None = None
    orientations: list = field(default_factory=list)
    applied: int = 0
    points: list[SupportPoint] = field(default_factory=list)
    supports: trimesh.Trimesh | None = None
    warnings: list[str] = field(default_factory=list)
    touched: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def _get(sid: str) -> Session:
    with _lock:
        s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "session not found — reload the page and drop the file again")
    s.touched = time.time()
    return s


def _put(session: Session) -> None:
    with _lock:
        _sessions[session.id] = session
        if len(_sessions) > MAX_SESSIONS:
            oldest = sorted(_sessions.values(), key=lambda s: s.touched)[: -MAX_SESSIONS]
            for s in oldest:
                _sessions.pop(s.id, None)


# ----------------------------------------------------------------- payloads


class ParamPatch(BaseModel):
    preset: str | None = None
    nozzle: float | None = None
    overrides: dict[str, Any] = {}


class OrientRequest(ParamPatch):
    pick: int = 0
    skip: bool = False  # keep the model's existing pose


class PointsRequest(ParamPatch):
    pass


class SupportsRequest(ParamPatch):
    points: list[dict] | None = None  # None = use whatever stage 2 produced


def _resolve_params(session: Session, patch: ParamPatch) -> SupportParams:
    if patch.nozzle is not None:
        params = presets.from_nozzle(patch.nozzle)
    elif patch.preset:
        params = presets.get(patch.preset)
    else:
        params = session.params
    return params.with_(**patch.overrides)


# --------------------------------------------------------------------- app

app = FastAPI(title="resin supports for FDM", docs_url="/api/docs")


@app.get("/api/presets")
def api_presets() -> dict:
    return {
        "default": presets.DEFAULT_PRESET,
        "presets": {name: vars(p) for name, p in presets.PRESETS.items()},
    }


@app.post("/api/model")
def api_model(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    suffix = Path(file.filename or "upload.stl").suffix or ".stl"
    try:
        mesh = mesh_io.mesh_from_bytes(data, suffix)
    except Exception as exc:
        raise HTTPException(400, f"could not read {file.filename!r}: {exc}") from exc

    session = Session(id=uuid.uuid4().hex[:12], original=mesh, params=presets.get())
    session.oriented = mesh_io.drop_to_bed(mesh)
    _put(session)
    return {"id": session.id, "name": file.filename, "summary": mesh_io.summary(mesh)}


@app.post("/api/orient/{sid}")
def api_orient(sid: str, req: OrientRequest) -> dict:
    session = _get(sid)
    from .. import orient as orient_mod

    session.params = _resolve_params(session, req)

    t0 = time.perf_counter()
    if req.skip:
        session.orientations = []
        session.applied = 0
        session.oriented = mesh_io.drop_to_bed(session.original)
    else:
        session.orientations = orient_mod.orientations(session.original, session.params, top_k=3)
        session.applied = min(req.pick, len(session.orientations) - 1)
        session.oriented = orient_mod.apply(
            session.original, session.orientations[session.applied]
        )

    # An orientation change invalidates everything downstream.
    session.points = []
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "applied": session.applied,
        "orientations": [o.as_dict() for o in session.orientations],
        "summary": mesh_io.summary(session.oriented),
    }


@app.post("/api/points/{sid}")
def api_points(sid: str, req: PointsRequest) -> dict:
    session = _get(sid)
    from .. import sampling

    session.params = _resolve_params(session, req)
    if session.oriented is None:
        raise HTTPException(409, "orient the model first")

    t0 = time.perf_counter()
    session.points = sampling.place_points(session.oriented, session.params)
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "points": [p.as_dict() for p in session.points],
    }


@app.post("/api/supports/{sid}")
def api_supports(sid: str, req: SupportsRequest) -> dict:
    session = _get(sid)
    from .. import supports as supports_mod

    session.params = _resolve_params(session, req)
    if session.oriented is None:
        raise HTTPException(409, "orient the model first")

    if req.points is not None:
        session.points = [SupportPoint.from_dict(d) for d in req.points]

    t0 = time.perf_counter()
    build = supports_mod.build_supports(session.oriented, session.points, session.params)
    session.supports = build.mesh
    session.warnings = list(build.warnings)
    return {
        "elapsed": time.perf_counter() - t0,
        "points": len(session.points),
        "braces": build.n_braces,
        "dropped": len(build.dropped),
        "faces": int(len(build.mesh.faces)) if build.mesh is not None else 0,
        "warnings": session.warnings[:20],
        "params": vars(session.params),
    }


@app.get("/api/mesh/{sid}/{which}")
def api_mesh(sid: str, which: str) -> Response:
    """Serve geometry to the viewer as binary STL.

    Not the most compact wire format, but it needs no encoder here and no
    decoder there, and this only ever travels over localhost.
    """
    session = _get(sid)
    mesh = {"model": session.oriented, "supports": session.supports}.get(which)
    if mesh is None or len(mesh.faces) == 0:
        raise HTTPException(404, f"no {which} geometry yet")

    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return Response(buf.getvalue(), media_type="model/stl")


@app.get("/api/export/{sid}")
def api_export(sid: str, mode: str = "3mf") -> FileResponse:
    session = _get(sid)
    if session.oriented is None:
        raise HTTPException(409, "nothing to export yet")

    import tempfile

    suffix = ".3mf" if mode == "3mf" else ".stl"
    out_dir = Path(tempfile.mkdtemp(prefix="rsupport_"))
    supports = session.supports if session.supports is not None else trimesh.Trimesh()
    written = export_mod.export(
        session.oriented, supports, out_dir / f"supported{suffix}", session.params, mode=mode
    )
    path = written[0]
    if mode == "separate":
        # Two files, so hand back a zip rather than picking one.
        import zipfile

        zip_path = out_dir / "supported.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in written:
                z.write(p, p.name)
        path = zip_path

    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
