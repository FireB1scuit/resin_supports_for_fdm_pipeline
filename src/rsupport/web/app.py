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

import numpy as np
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
    #: The chosen pose, sitting flat on z=0. `oriented` is this same mesh
    #: floated up by ``params.lift_height`` — kept apart so moving the lift
    #: slider is a translation rather than another orientation search.
    grounded: trimesh.Trimesh | None = None
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


def _resolve_params(session: Session, patch: ParamPatch) -> SupportParams:
    if patch.nozzle is not None:
        params = presets.from_nozzle(patch.nozzle)
    elif patch.preset:
        params = presets.get(patch.preset)
    else:
        params = session.params
    return params.with_(**patch.overrides)


def _set_params(session: Session, patch: ParamPatch) -> SupportParams:
    """Resolve a patch onto the session and re-float the model if it moved.

    The lift is the one parameter that changes the *model*, not the supports, so
    it is applied here rather than in a geometry stage. Everything downstream
    reads absolute Z, so a moved model invalidates the point list — the caller
    is stage 2 or a stage-3 run whose points came back from the client with the
    model in its old place, and only re-running stage 2 fixes that. The UI
    re-runs both.
    """
    session.params = _resolve_params(session, patch)
    if session.grounded is not None:
        lift = float(session.params.lift_height)
        current = float(session.oriented.bounds[0][2]) if session.oriented is not None else None
        if current is None or abs(current - lift) > 1e-9:
            session.oriented = mesh_io.set_lift(session.grounded, lift)
            session.points = []
            session.supports = None
    return session.params


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
    session.grounded = mesh_io.drop_to_bed(mesh)
    session.oriented = mesh_io.set_lift(session.grounded, session.params.lift_height)
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
        session.grounded = mesh_io.drop_to_bed(session.original)
    else:
        session.orientations = orient_mod.orientations(session.original, session.params, top_k=3)
        session.applied = min(req.pick, len(session.orientations) - 1)
        session.grounded = orient_mod.apply(
            session.original, session.orientations[session.applied]
        )
    session.oriented = mesh_io.set_lift(session.grounded, session.params.lift_height)

    # An orientation change invalidates everything downstream.
    session.points = []
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "applied": session.applied,
        "orientations": [o.as_dict() for o in session.orientations],
        "summary": mesh_io.summary(session.oriented),
    }


@app.post("/api/rotate/{sid}")
def api_rotate(sid: str, req: RotateRequest) -> dict:
    """Manual rotation about the model's own X, Y, then Z axes, in degrees.

    Always applied from ``session.original`` rather than composed onto the
    current pose, so the three sliders stay an absolute readout — dialling
    them back to 0/0/0 reproduces the file's pose exactly, the same as
    ``skip`` does for the (opt-in, CLI-only from the UI's perspective) search
    in :func:`api_orient`.
    """
    session = _get(sid)
    session.params = _resolve_params(session, req)

    t0 = time.perf_counter()
    rot = trimesh.transformations.euler_matrix(
        np.radians(req.rx), np.radians(req.ry), np.radians(req.rz), axes="sxyz"
    )
    rotated = session.original.copy()
    rotated.apply_transform(rot)

    session.orientations = []
    session.applied = 0
    session.grounded = mesh_io.drop_to_bed(rotated)
    session.oriented = mesh_io.set_lift(session.grounded, session.params.lift_height)

    # A rotation invalidates everything downstream, same as a reorient.
    session.points = []
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "summary": mesh_io.summary(session.oriented),
    }


@app.post("/api/points/{sid}")
def api_points(sid: str, req: PointsRequest) -> dict:
    session = _get(sid)
    from .. import sampling

    _set_params(session, req)
    if session.oriented is None:
        raise HTTPException(409, "orient the model first")

    t0 = time.perf_counter()
    session.points = sampling.place_points(session.oriented, session.params)
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "lift": float(session.params.lift_height),
        "points": [p.as_dict() for p in session.points],
    }


@app.post("/api/supports/{sid}")
def api_supports(sid: str, req: SupportsRequest) -> dict:
    session = _get(sid)
    from .. import supports as supports_mod

    _set_params(session, req)
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
