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
    #: The filename the model arrived under, kept only to build the exported
    #: file's name (see `export_stem`) — never re-parsed for anything else.
    upload_name: str | None = None
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


@dataclass
class Workspace:
    """Which uploaded models belong together, and which of them is active.

    A tab can hold more than one model now, but nothing about running a stage
    changes: every function above still takes one model's own :class:`Session`
    directly, addressed by its own ``sid``, and each session is exactly as
    independent as it always was. A ``Workspace`` is bookkeeping the UI needs
    — which sids came from drops into the same tab, and which one a click in
    the viewer most recently chose — not a new place the pipeline runs.
    Switching the active model touches nothing about any ``Session``: rotate,
    lift and the build stages all still act only on the sid they are given.
    """

    id: str
    model_ids: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    active: str | None = None
    touched: float = field(default_factory=time.time)


class WorkspaceStore:
    """Same shape and the same eviction policy as :class:`SessionStore`, kept
    separate because a workspace and the models in it are evicted
    independently — losing one on its own schedule must not break the other.
    """

    def __init__(self, limit: int = MAX_SESSIONS):
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()
        self._limit = limit

    def get(self, wid: str) -> Workspace:
        with self._lock:
            workspace = self._workspaces.get(wid)
        if workspace is None:
            raise StageError(404, "workspace not found — reload the page and start again")
        workspace.touched = time.time()
        return workspace

    def put(self, workspace: Workspace) -> Workspace:
        with self._lock:
            self._workspaces[workspace.id] = workspace
            if len(self._workspaces) > self._limit:
                oldest = sorted(self._workspaces.values(), key=lambda w: w.touched)
                for stale in oldest[: -self._limit]:
                    self._workspaces.pop(stale.id, None)
        return workspace


def create_workspace() -> Workspace:
    return Workspace(id=uuid.uuid4().hex[:12])


def add_model(workspace: Workspace, session: Session, filename: str | None = None) -> None:
    """Add a freshly uploaded model to a workspace and make it the active one.

    Always the active one, not just the first: dropping a file has always put
    it on screen immediately, and holding onto earlier models in the
    background is not a reason to change what a drop does.
    """
    workspace.model_ids.append(session.id)
    workspace.names[session.id] = filename or "model"
    workspace.active = session.id
    workspace.touched = time.time()


def set_active(workspace: Workspace, sid: str) -> None:
    """Point the workspace at a different one of its own models.

    Every other route still takes an explicit sid, so this changes nothing
    about the pipeline — it only records which sid the UI means by "the
    active model" the next time it does not name one itself.
    """
    if sid not in workspace.model_ids:
        raise StageError(404, f"{sid!r} is not a model in this workspace")
    workspace.active = sid
    workspace.touched = time.time()


def workspace_payload(workspace: Workspace, store: SessionStore) -> dict:
    """Every model in the workspace whose session is still alive, and which is
    active. The two stores share a cap but not a clock, so a model can be
    evicted out from under a workspace that still names it — left out here
    rather than reported and then 404ing the moment it is clicked.
    """
    models = []
    for mid in workspace.model_ids:
        try:
            session = store.get(mid)
        except StageError:
            continue
        models.append({
            "id": mid,
            "name": workspace.names.get(mid),
            "summary": mesh_io.summary(session.original),
        })
    live_ids = {m["id"] for m in models}
    active = workspace.active if workspace.active in live_ids else None
    return {"id": workspace.id, "active": active, "models": models}


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

    session = Session(
        id=uuid.uuid4().hex[:12], original=mesh, params=presets.get(), upload_name=filename
    )
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


def overhang_area(mesh, angle_deg: float) -> float:
    """mm^2 of downward face flagged at ``angle_deg``.

    The one number that says whether the pose is worth keeping: it is the area
    the scaffold has to reach, and re-posing the model is the only thing that
    meaningfully shrinks it. Cheap enough to hand back on every placement run —
    a mask and a dot product over face areas the mesh has already computed.
    """
    from .. import overhang as overhang_mod

    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return 0.0
    mask = overhang_mod.overhang_mask(mesh, angle_deg)
    return float(np.asarray(mesh.area_faces, dtype=np.float64)[mask].sum())


def run_points(session: Session, **patch: Any) -> dict:
    from .. import sampling

    set_params(session, **patch)
    if session.oriented is None:
        raise StageError(409, "orient the model first")

    t0 = time.perf_counter()
    session.points = sampling.place_points(session.oriented, session.params)
    session.supports = None
    lo, hi = session.oriented.bounds
    return {
        "elapsed": time.perf_counter() - t0,
        "lift": float(session.params.lift_height),
        "points": [p.as_dict() for p in session.points],
        "size": [float(v) for v in (hi - lo)],
        "overhang_area": overhang_area(session.oriented, session.params.overhang_angle_deg),
    }


def overhang_faces(session: Session, angle_deg: float | None = None) -> dict:
    """Which faces of the current pose are overhang, for the viewer's toggle.

    This is a read-only viewer aid, not a build stage: it reflects whatever
    pose happens to be loaded right now (``session.oriented``) and is
    recomputed only when the caller asks — it has no scope in the UI's
    points/geometry staleness table and must never be given one.

    Args:
        angle_deg: the overhang angle to flag at. Defaults to the session's
            current ``overhang_angle_deg`` so a caller that does not know
            better still gets a sensible answer; the viewer passes the live
            value of its own slider so the overlay answers to whatever angle
            is dialled in at the moment it is toggled on.
    """
    from .. import overhang as overhang_mod

    if session.oriented is None:
        raise StageError(409, "orient the model first")

    mesh = session.oriented
    angle = float(angle_deg) if angle_deg is not None else float(session.params.overhang_angle_deg)
    n_faces = int(len(mesh.faces))
    if n_faces == 0:
        return {"faces": [], "total_faces": 0, "angle_deg": angle, "area": 0.0}

    mask = overhang_mod.overhang_mask(mesh, angle)
    area = float(np.asarray(mesh.area_faces, dtype=np.float64)[mask].sum())
    return {
        "faces": np.nonzero(mask)[0].tolist(),
        "total_faces": n_faces,
        "angle_deg": angle,
        "area": area,
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
        # An upper bound, not a measurement. The scaffold is a soup of closed
        # shells that merely overlap at every junction (see resin._joint_radius),
        # so the signed volume counts each overlap twice. Good enough to answer
        # "roughly how much filament is this costing me"; not good enough to
        # present as a fact, which is why the UI says "under".
        "volume": support_volume(build.mesh),
        "warnings": session.warnings[:20],
        "params": params_payload(session.params),
    }


def support_volume(mesh) -> float:
    """mm^3 enclosed by the support mesh, over-counting overlaps at joints."""
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return 0.0
    try:
        return max(0.0, float(mesh.volume))
    except Exception:  # degenerate soup — a number here is not worth a 500
        return 0.0


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


def export_stem(session: Session) -> str:
    """The exported file's basename, without an extension.

    Named for the file that was uploaded rather than a generic "supported",
    and signed, per GitHub issue #24 — so the file a person saves says both
    what it is and where it came from once it is sitting in a downloads
    folder next to a dozen others.
    """
    name = session.upload_name or "model"
    stem = Path(name).stem or "model"
    return f"{stem}_resin-fdm-supported_by-FireB1scuit"


def export_bytes(
    session: Session, fmt: str = "stl", separate: bool = False, part: str | None = None
) -> tuple[str, bytes]:
    """The finished thing, as ``(filename, payload)``.

    Bytes rather than a path, because one caller streams a file off disk and the
    other hands the array to the page to save. ``fmt`` is the file type — "stl"
    (the default) or "3mf", never both, per issue #24 — and ``separate`` picks
    combined vs. two-file output within that format. Two files come back one at
    a time rather than zipped: ``part`` ("model" or "supports") picks which,
    since a slicer wants two plain files, not an archive it has to unpack
    first, and a browser download is one file per call anyway.
    """
    if session.oriented is None:
        raise StageError(409, "nothing to export yet")
    if fmt not in ("stl", "3mf"):
        raise StageError(400, f"unknown export format {fmt!r} — must be 'stl' or '3mf'")
    if separate and part not in ("model", "supports"):
        raise StageError(400, "separate export needs part='model' or part='supports'")

    import tempfile

    suffix = ".3mf" if fmt == "3mf" else ".stl"
    mode = "separate" if separate else ("3mf" if fmt == "3mf" else "combined")
    stem = export_stem(session)
    out_dir = Path(tempfile.mkdtemp(prefix="rsupport_"))
    supports = session.supports if session.supports is not None else trimesh.Trimesh()
    written = export_mod.export(
        session.oriented,
        supports,
        out_dir / f"{stem}{suffix}",
        session.params,
        mode=mode,
        # A separate export drops the supports out of the model file, so the
        # marker cube is what keeps a floating model's pose visually honest —
        # see `export_mod.plate_marker_cube`.
        marker_cube=separate,
    )
    if mode == "separate" and part == "supports":
        if len(written) < 2:
            raise StageError(404, "no supports to export")
        path = written[1]
    else:
        path = written[0]
    return path.name, path.read_bytes()
