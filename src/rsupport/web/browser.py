"""The app with the server taken out: same routes, no network.

In a browser build the whole pipeline runs in the tab, under Pyodide, inside a
Web Worker. There is no host to talk to after the initial download — which is
the entire point, since it makes the thing free to host and means a model never
leaves the machine it was opened on.

This module is what the worker calls. It deliberately presents the **same routes
as** :mod:`rsupport.web.app`, dispatched here instead of over HTTP::

    route("POST", f"/api/points/{sid}", body={...})

so ``app.js`` keeps every call site it already had and only its transport
changes — one ``fetch`` swapped for one ``postMessage``. Two front ends that
agree on paths and payloads are far harder to drift apart than two that each
invent their own, and the logic under both is the single copy in
:mod:`rsupport.web.core`.

Responses mirror what fastapi would have sent: a status, and either decoded
JSON or bytes. Nothing here raises for an expected failure — a bad upload or a
missing session comes back as a status, exactly as it would over the wire.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import core
from .core import SessionStore, StageError

__all__ = ["Router", "route"]

_MESH = re.compile(r"^/api/mesh/([^/]+)/([^/]+)$")
_EXPORT = re.compile(r"^/api/export/([^/]+)$")
_STAGE = re.compile(r"^/api/(orient|rotate|points|supports)/([^/]+)$")
_WORKSPACE_MODEL = re.compile(r"^/api/workspace/([^/]+)/model$")
_WORKSPACE_ACTIVE = re.compile(r"^/api/workspace/([^/]+)/active$")
_WORKSPACE_GET = re.compile(r"^/api/workspace/([^/]+)$")


class Response:
    """What a route produced: a status, and JSON or bytes.

    ``filename`` is set only by the export route, where the page needs a name to
    save the download under — a browser has no Content-Disposition to read when
    nothing was ever served.
    """

    __slots__ = ("status", "data", "body", "filename")

    def __init__(self, status=200, data=None, body=None, filename=None):
        self.status = status
        self.data = data
        self.body = body
        self.filename = filename

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def to_json(self) -> str:
        """The structured half, for a worker that hands bytes back separately."""
        return json.dumps(
            {
                "status": self.status,
                "ok": self.ok,
                "data": self.data,
                "filename": self.filename,
                "binary": self.body is not None,
            }
        )


class Router:
    """One session store and the routes over it.

    A tab only ever holds one model, but the store is the same one the server
    uses — sharing it keeps the eviction and not-found behaviour identical
    rather than approximately so.
    """

    def __init__(self):
        self.store = SessionStore()
        #: Same relationship as in `rsupport.web.app`: a separate store because
        #: a workspace and the models it names are evicted independently.
        self.workspaces = core.WorkspaceStore()

    # -- entry point ---------------------------------------------------- #

    def route(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        data: bytes | None = None,
        filename: str | None = None,
    ) -> Response:
        """Dispatch one request. Never raises for an expected failure."""
        method = method.upper()
        split = urlsplit(path)
        query = parse_qs(split.query)
        try:
            return self._dispatch(method, split.path, body or {}, data, filename, query)
        except StageError as exc:
            return Response(exc.status, {"detail": exc.detail})
        except Exception as exc:  # a genuine bug, reported rather than swallowed
            return Response(500, {"detail": f"{type(exc).__name__}: {exc}"})

    def _dispatch(self, method, path, body, data, filename, query) -> Response:
        if path == "/api/presets" and method == "GET":
            return Response(200, core.presets_payload())

        if path == "/api/model" and method == "POST":
            if data is None:
                raise StageError(400, "no file data")
            session = core.load_model(bytes(data), filename)
            self.store.put(session)
            return Response(200, core.model_payload(session, filename))

        if path == "/api/workspace" and method == "POST":
            workspace = core.create_workspace()
            self.workspaces.put(workspace)
            return Response(200, core.workspace_payload(workspace, self.store))

        if m := _WORKSPACE_MODEL.match(path):
            if method != "POST":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            workspace = self.workspaces.get(m.group(1))
            if data is None:
                raise StageError(400, "no file data")
            session = core.load_model(bytes(data), filename)
            self.store.put(session)
            core.add_model(workspace, session, filename)
            return Response(200, core.workspace_payload(workspace, self.store))

        if m := _WORKSPACE_ACTIVE.match(path):
            if method != "POST":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            workspace = self.workspaces.get(m.group(1))
            core.set_active(workspace, str(body.get("sid", "")))
            return Response(200, core.workspace_payload(workspace, self.store))

        if m := _WORKSPACE_GET.match(path):
            if method != "GET":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            workspace = self.workspaces.get(m.group(1))
            return Response(200, core.workspace_payload(workspace, self.store))

        if m := _STAGE.match(path):
            if method != "POST":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            return Response(200, self._stage(m.group(1), self.store.get(m.group(2)), body))

        if m := _MESH.match(path):
            if method != "GET":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            session = self.store.get(m.group(1))
            return Response(200, body=core.mesh_stl(session, m.group(2)))

        if m := _EXPORT.match(path):
            if method != "GET":
                return Response(405, {"detail": f"{method} not allowed on {path}"})
            session = self.store.get(m.group(1))
            mode = (query.get("mode") or ["3mf"])[0]
            name, payload = core.export_bytes(session, mode)
            return Response(200, body=payload, filename=name)

        return Response(404, {"detail": f"no route for {method} {path}"})

    @staticmethod
    def _stage(name: str, session, body: dict) -> dict:
        """Run one pipeline stage, splitting a request body the way the
        fastapi models do — the param patch from the stage's own arguments."""
        patch = {
            "preset": body.get("preset"),
            "nozzle": body.get("nozzle"),
            "overrides": body.get("overrides") or {},
        }
        if name == "orient":
            return core.run_orient(
                session, pick=int(body.get("pick", 0)), skip=bool(body.get("skip", False)), **patch
            )
        if name == "rotate":
            return core.run_rotate(
                session,
                rx=float(body.get("rx", 0.0)),
                ry=float(body.get("ry", 0.0)),
                rz=float(body.get("rz", 0.0)),
                **patch,
            )
        if name == "points":
            return core.run_points(session, **patch)
        return core.run_supports(session, points=body.get("points"), **patch)


#: The router a worker talks to. One per interpreter, which is one per tab.
_router = Router()


def route(method, path, body=None, data=None, filename=None) -> Response:
    return _router.route(method, path, body=body, data=data, filename=filename)
