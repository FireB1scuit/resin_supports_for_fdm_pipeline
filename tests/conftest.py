"""Test setup: make `src/` importable, and build the synthetic miniature.

The synthetic mini stands in for a real sculpt in every stage-1 test. It has the
three features orientation has to reason about:

    * a wide thin disc base   -- the obvious thing to set flat on the plate
    * a tall body             -- so "lying down" is a genuinely different pose
    * a bumpy front, flat back -- so detail has a direction to point in

Dimensions here are test fixture dimensions, not geometry-code constants; the
rule about deriving everything from the nozzle applies to `src/`, not to a
stand-in for an STL someone would drop on the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_RADIUS = 8.0
BASE_HEIGHT = 1.0
BODY_WIDTH = 6.0
BODY_HEIGHT = 20.0


def synthetic_mini(bump_amplitude: float = 0.35, subdivisions: int = 3) -> trimesh.Trimesh:
    """A box body on a disc base, with a sculpted-looking patch on its +X side.

    The displacement tapers to zero at the patch boundary, so the body stays a
    closed shell and the -X side stays perfectly flat.
    """
    body = trimesh.creation.box(extents=[BODY_WIDTH, BODY_WIDTH, BODY_HEIGHT])
    body.apply_translation([0.0, 0.0, BASE_HEIGHT + BODY_HEIGHT * 0.5])
    for _ in range(subdivisions):
        verts, faces = trimesh.remesh.subdivide(body.vertices, body.faces)
        body = trimesh.Trimesh(verts, faces, process=True)

    v = body.vertices.copy()
    front = np.isclose(v[:, 0], v[:, 0].max(), atol=1e-6)
    y = v[front, 1]
    z = v[front, 2] - BASE_HEIGHT
    # Window that vanishes on every edge of the face, so the shell stays closed.
    window = np.cos(np.pi * y / BODY_WIDTH) * np.sin(np.pi * z / BODY_HEIGHT)
    ripple = np.sin(4.0 * np.pi * y / BODY_WIDTH) * np.sin(6.0 * np.pi * z / BODY_HEIGHT)
    v[front, 0] += bump_amplitude * window * ripple
    body = trimesh.Trimesh(v, body.faces, process=False)

    base = trimesh.creation.cylinder(radius=BASE_RADIUS, height=BASE_HEIGHT, sections=64)
    base.apply_translation([0.0, 0.0, BASE_HEIGHT * 0.5])

    return trimesh.util.concatenate([body, base])


def _body_patch_mask(mesh: trimesh.Trimesh, sign: float) -> np.ndarray:
    """Faces well inside the +X (sign=1) or -X (sign=-1) side of the body.

    Kept clear of the box's own edges, which are legitimately high curvature and
    would swamp the comparison we actually care about.
    """
    c = np.asarray(mesh.triangles_center)
    return (
        (sign * c[:, 0] > BODY_WIDTH * 0.4)
        & (np.abs(c[:, 1]) < BODY_WIDTH * 0.33)
        & (c[:, 2] > BASE_HEIGHT + BODY_HEIGHT * 0.2)
        & (c[:, 2] < BASE_HEIGHT + BODY_HEIGHT * 0.8)
    )


def bumpy_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    return _body_patch_mask(mesh, 1.0)


def smooth_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    return _body_patch_mask(mesh, -1.0)


def random_rotation(seed: int = 0) -> np.ndarray:
    """An arbitrary but reproducible 4x4 rotation, for pose-invariance tests."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return trimesh.transformations.rotation_matrix(rng.uniform(0.5, 2.5), axis)


@pytest.fixture(scope="session")
def mini() -> trimesh.Trimesh:
    return synthetic_mini()
