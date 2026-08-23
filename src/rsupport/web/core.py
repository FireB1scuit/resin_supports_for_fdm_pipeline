"""The app's logic, with no transport attached.

Everything the UI can ask for lives here as a plain function over a
:class:`Session`: load a model, orient it, rotate it, place points, build the
scaffold, hand back geometry. Nothing in this module knows about HTTP, and
nothing in it imports fastapi.

That is the whole point. There are two front ends and they must not drift:

* :mod:`rsupport.web.app` — FastAPI, for ``python -m rsupport.web`` and the
  container;
* :mod:`rsupport.web.browser` — a dispatch table for the Pyodide worker, where
  there is no server at all and the "request" is a ``postMessage``.

The stages are separate for the same reason the endpoints were: changing a tip
diameter only needs stage 3 re-run, changing the overhang angle needs 2 and 3,
changing the orientation needs all of it. The UI knows which is which.

Failures raise :class:`StageError`, which carries an HTTP-ish status code
because one of the two callers needs one and the other can ignore it.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .. import export as export_mod
from .. import mesh_io, presets
from ..types import SupportParams, SupportPoint

#: Sessions are dropped oldest-first past this, so a long-running server does
#: not accumulate a gigabyte of abandoned uploads. A browser build only ever has
#: one, but the cap costs nothing there.
MAX_SESSIONS = 8


class StageError(Exception):
    """Something the caller asked for cannot be done, and why.

    ``status`` is an HTTP code so :mod:`rsupport.web.app` can hand it straight
    to fastapi; the browser front end reports ``detail`` and ignores the number.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


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


class SessionStore:
    """In-memory sessions, oldest evicted first.

    No database and no persistence: this is a single-user tool, whether it is
    running on your own machine or entirely inside your own tab, and closing it
    throws the work away.
    """

    def __init__(self, limit: int = MAX_SESSIONS):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._limit = limit

    def get(self, sid: str) -> Session:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            raise StageError(404, "session not found — reload the page and drop the file again")
        session.touched = time.time()
        return session

    def put(self, session: Session) -> Session:
        with self._lock:
            self._sessions[session.id] = session
            if len(self._sessions) > self._limit:
                oldest = sorted(self._sessions.values(), key=lambda s: s.touched)
                for stale in oldest[: -self._limit]:
                    self._sessions.pop(stale.id, None)
        return session


# ------------------------------------------------------------------ params


def unknown_overrides(overrides: dict[str, Any]) -> list[str]:
    """Names in a patch that this build of SupportParams has never heard of.

    ``with_`` ignores them, which is what makes the UI able to post the whole
    control panel every time. It also means a page talking to an older build
    gets silence: the control moves, the request succeeds, and the value is
    dropped without a word. Report them so the log says so.
    """
    return sorted(set(overrides) - set(SupportParams.__dataclass_fields__))


def resolve_params(
    session: Session,
    preset: str | None = None,
    nozzle: float | None = None,
    overrides: dict[str, Any] | None = None,
) -> SupportParams:
    if nozzle is not None:
        params = presets.from_nozzle(nozzle)
    elif preset:
        params = presets.get(preset)
    else:
        params = session.params
    return params.with_(**(overrides or {}))


def set_params(session: Session, **patch: Any) -> SupportParams:
    """Resolve a patch onto the session and re-float the model if it moved.

    The lift is the one parameter that changes the *model*, not the supports, so
    it is applied here rather than in a geometry stage. Everything downstream
    reads absolute Z, so a moved model invalidates the point list — the caller
    is stage 2, or a stage-3 run whose points came back from the client with the
    model in its old place, and only re-running stage 2 fixes that. The UI
    re-runs both.
    """
    session.params = resolve_params(session, **patch)
    if session.grounded is not None:
        lift = float(session.params.lift_height)
        current = float(session.oriented.bounds[0][2]) if session.oriented is not None else None
        if current is None or abs(current - lift) > 1e-9:
            session.oriented = mesh_io.set_lift(session.grounded, lift)
            session.points = []
            session.supports = None
    return session.params


def params_payload(params: SupportParams) -> dict:
    """A parameter set as the UI needs to see it.

    ``brace_angle_deg`` is None until somebody sets it, meaning "the shallowest
    angle that prints". A slider cannot show None, and the alternative — the
    client working the angle out for itself — would put the printability rule in
    two places. So report the angle actually in force, and let a moved slider
    come back as an explicit value.
    """
    from .. import resin  # geometry imports stay lazy, as elsewhere in here

    out = dict(vars(params))
    out["brace_angle_deg"] = resin.link_angle(params)
    return out


def presets_payload() -> dict:
    return {
        "default": presets.DEFAULT_PRESET,
        "presets": {name: params_payload(p) for name, p in presets.PRESETS.items()},
    }


# ------------------------------------------------------------------ stages


def load_model(data: bytes, filename: str | None = None) -> Session:
    suffix = Path(filename or "upload.stl").suffix or ".stl"
    try:
        mesh = mesh_io.mesh_from_bytes(data, suffix)
    except Exception as exc:
        raise StageError(400, f"could not read {filename!r}: {exc}") from exc

    session = Session(id=uuid.uuid4().hex[:12], original=mesh, params=presets.get())
    session.grounded = mesh_io.drop_to_bed(mesh)
    session.oriented = mesh_io.set_lift(session.grounded, session.params.lift_height)
    return session


def model_payload(session: Session, filename: str | None = None) -> dict:
    return {
        "id": session.id,
        "name": filename,
        "summary": mesh_io.summary(session.original),
    }


def run_orient(session: Session, pick: int = 0, skip: bool = False, **patch: Any) -> dict:
    from .. import orient as orient_mod

    session.params = resolve_params(session, **patch)

    t0 = time.perf_counter()
    if skip:
        session.orientations = []
        session.applied = 0
        session.grounded = mesh_io.drop_to_bed(session.original)
    else:
        session.orientations = orient_mod.orientations(session.original, session.params, top_k=3)
        session.applied = min(pick, len(session.orientations) - 1)
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


def run_rotate(
    session: Session, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, **patch: Any
) -> dict:
    """Manual rotation about the model's own X, Y, then Z axes, in degrees.

    Always applied from ``session.original`` rather than composed onto the
    current pose, so the three sliders stay an absolute readout — dialling them
    back to 0/0/0 reproduces the file's pose exactly, the same as ``skip`` does
    for the (opt-in) search in :func:`run_orient`.
    """
    session.params = resolve_params(session, **patch)

    t0 = time.perf_counter()
    rot = trimesh.transformations.euler_matrix(
        np.radians(rx), np.radians(ry), np.radians(rz), axes="sxyz"
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


def run_points(session: Session, **patch: Any) -> dict:
    from .. import sampling

    set_params(session, **patch)
    if session.oriented is None:
        raise StageError(409, "orient the model first")

    t0 = time.perf_counter()
    session.points = sampling.place_points(session.oriented, session.params)
    session.supports = None
    return {
        "elapsed": time.perf_counter() - t0,
        "lift": float(session.params.lift_height),
        "points": [p.as_dict() for p in session.points],
    }


def run_supports(session: Session, points: list[dict] | None = None, **patch: Any) -> dict:
    from .. import supports as supports_mod

    overrides = patch.get("overrides") or {}
    set_params(session, **patch)
    if session.oriented is None:
        raise StageError(409, "orient the model first")

    if points is not None:
        session.points = [SupportPoint.from_dict(d) for d in points]

    t0 = time.perf_counter()
    build = supports_mod.build_supports(session.oriented, session.points, session.params)
    session.supports = build.mesh
    session.warnings = list(build.warnings)
    if stale := unknown_overrides(overrides):
        session.warnings.insert(
            0,
            f"this server does not know the setting(s) {', '.join(stale)} — it is "
            "older than the page talking to it, so those controls did nothing. "
            "Restart it.",
        )
    return {
        "elapsed": time.perf_counter() - t0,
        "points": len(session.points),
        "braces": build.n_braces,
        "dropped": len(build.dropped),
        # *Which* ones, so the viewer can mark them. Stage 3 hands back the very
        # objects it was given, so identity is the mapping — matching on
        # position would have to survive a float round-trip through JSON.
        "dropped_points": dropped_indices(session.points, build.dropped),
        "faces": int(len(build.mesh.faces)) if build.mesh is not None else 0,
        "warnings": session.warnings[:20],
        "params": params_payload(session.params),
    }


def dropped_indices(points: list[SupportPoint], dropped: list[SupportPoint]) -> list[int]:
    """Positions in `points` of the contacts stage 3 could not support.

    The viewer draws every contact in a washed-out red and these in a solid one,
    so an overhang left unheld is obvious rather than buried in a log line. It
    keeps the point list itself intact — a dropped contact is still a contact,
    still clickable, and still there to be dragged somewhere supportable.
    """
    at = {id(p): i for i, p in enumerate(points)}
    return sorted(at[id(p)] for p in dropped if id(p) in at)


def mesh_stl(session: Session, which: str) -> bytes:
    """Geometry for the viewer, as binary STL.

    Not the most compact wire format, but it needs no encoder here and no
    decoder there — and in a browser build it never travels at all.
    """
    mesh = {"model": session.oriented, "supports": session.supports}.get(which)
    if mesh is None or len(mesh.faces) == 0:
        raise StageError(404, f"no {which} geometry yet")
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


def export_bytes(session: Session, mode: str = "3mf") -> tuple[str, bytes]:
    """The finished thing, as ``(filename, payload)``.

    Bytes rather than a path, because one caller streams a file off disk and the
    other hands the array to the page to save. ``separate`` writes two meshes,
    so it comes back zipped rather than by picking one.
    """
    if session.oriented is None:
        raise StageError(409, "nothing to export yet")

    import tempfile

    suffix = ".3mf" if mode == "3mf" else ".stl"
    out_dir = Path(tempfile.mkdtemp(prefix="rsupport_"))
    supports = session.supports if session.supports is not None else trimesh.Trimesh()
    written = export_mod.export(
        session.oriented, supports, out_dir / f"supported{suffix}", session.params, mode=mode
    )
    path = written[0]
    if mode == "separate":
        zip_path = out_dir / "supported.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in written:
                z.write(p, p.name)
        path = zip_path
    return path.name, path.read_bytes()
