"""Stage 3 — turn support points into printable support geometry.

This module owns two things: the shell primitives every part of a support is
lofted from, and :func:`build_supports`, the stage-3 entry point the CLI, the
web app and the tests all call. The scaffold itself is planned and assembled in
:mod:`rsupport.resin`.

The one rule everything here bends to
-------------------------------------
**Generated support geometry may never contain an overhang steeper than
``params.printable_overhang_deg``.** If our supports need supports, the
generator is broken. That single constraint decides most of the design below,
so it is worth spelling out the maths.

Take a surface of revolution whose horizontal cross-sections are circles of
radius ``r(z)`` whose centres drift sideways at rate ``m`` (mm of horizontal
travel per mm of height — a leaning strut). Its outward normal at azimuth ``u``
has a vertical component proportional to ``-(m·u + r'(z))``, so the steepest
overhang anywhere on that surface is::

    angle_below_horizontal = atan(|m| + max(r', 0))

which gives three design rules used throughout:

* A profile that **narrows going up** (``r' <= 0``) is always self-supporting.
  Hence the tip cone is wide at the bottom and thin at the top, the base is a
  disc that only ever narrows on its way up, and the "spherical" tip is a dome
  rather than a ball (a free-floating sphere's underside is a 90 degree
  overhang).
* Leaning a strut by ``t`` costs exactly ``t`` degrees of overhang budget, so
  every lean is clamped by :func:`rsupport.avoidance.strut_lean`.
* A **flat downward face is a 90 degree overhang**, always. So a shaft is built
  as one continuous stack of rings — base, join cone, shaft, all in a single
  shell — rather than as separate capped primitives stacked on top of each
  other. Where two parts genuinely do meet, the buried cap is suppressed with
  ``cap_bottom=False`` rather than left face to face with its neighbour.

Cross-links are the exception that proves the rule: a plain strut at angle ``a``
above horizontal has side walls overhanging by ``90 - a`` and end caps
overhanging by ``a``. Both are printable only in the band
``[90 - printable, printable]`` — with the defaults, roughly 40 to 50 degrees.
That is why links are placed at a *chosen* angle by picking the two attachment
heights, instead of just connecting two convenient points. See
``resin._link_angle``.

Nothing here casts a general ray: ``rsupport.raycast.DownRay`` only does
straight down, which is all this stage needs, and a leaning strut's axis is
still a straight segment.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from .raycast import DownRay
from .types import SupportBuild, SupportParams

__all__ = [
    "build_supports",
    "make_tip",
    "make_foot",
    "make_strut",
    "mesh_rings",
    "rings_on_axis",
    "foot_profile",
    "tip_profile",
    "LineXY",
    "overhang_angles",
]

_EPS = 1e-9

#: Most of the flare the base can afford, as a fraction of its height. The rest
#: is straight-walled disc — see :func:`foot_profile`.
FOOT_FLARE_SHARE = 0.5


# --------------------------------------------------------------------------- #
# primitive construction — rings in, mesh out
# --------------------------------------------------------------------------- #


def mesh_rings(
    rings: np.ndarray,
    sections: int,
    cap_bottom: bool = True,
    cap_top: bool = True,
) -> trimesh.Trimesh:
    """Loft a stack of horizontal circles into a closed shell.

    Args:
        rings: (M, 4) array of ``(x, y, z, radius)``, ordered bottom to top.
            The centres may drift in XY — that is how a leaning strut is built.
        sections: polygon sides per ring.
        cap_bottom / cap_top: close the ends with a triangle fan.
    """
    rings = np.asarray(rings, dtype=np.float64).reshape(-1, 4)
    rings = _dedupe_rings(rings)
    m = len(rings)
    if m < 2:
        return trimesh.Trimesh()

    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    verts = np.empty((m * sections, 3), dtype=np.float64)
    for i, (cx, cy, z, r) in enumerate(rings):
        s = i * sections
        verts[s : s + sections, 0] = cx + r * ca
        verts[s : s + sections, 1] = cy + r * sa
        verts[s : s + sections, 2] = z

    k = np.arange(sections, dtype=np.int64)
    k1 = (k + 1) % sections

    faces = []
    for i in range(m - 1):
        lo = i * sections
        up = (i + 1) * sections
        faces.append(np.column_stack([lo + k, lo + k1, up + k1]))
        faces.append(np.column_stack([lo + k, up + k1, up + k]))

    extra = []
    n_v = len(verts)
    if cap_bottom:
        c = n_v + len(extra)
        extra.append([rings[0, 0], rings[0, 1], rings[0, 2]])
        # Reversed winding so the fan's normal points down.
        faces.append(np.column_stack([np.full(sections, c), k1, k]))
    if cap_top:
        c = n_v + len(extra)
        extra.append([rings[-1, 0], rings[-1, 1], rings[-1, 2]])
        base = (m - 1) * sections
        faces.append(np.column_stack([np.full(sections, c), base + k, base + k1]))

    if extra:
        verts = np.vstack([verts, np.asarray(extra, dtype=np.float64)])

    return trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=False)


def _dedupe_rings(rings: np.ndarray) -> np.ndarray:
    """Drop rings that repeat the previous height, and force monotone z."""
    if len(rings) == 0:
        return rings
    keep = [rings[0]]
    for r in rings[1:]:
        if r[2] - keep[-1][2] <= 1e-9:
            # Same height: keep the wider one so nothing pinches shut.
            if r[3] > keep[-1][3]:
                keep[-1] = r
            continue
        keep.append(r)
    return np.asarray(keep, dtype=np.float64)


def rings_on_axis(axis_xy, profile) -> np.ndarray:
    """Attach ``(z, radius)`` pairs to an axis, giving ``(x, y, z, r)`` rings."""
    out = np.empty((len(profile), 4), dtype=np.float64)
    for i, (z, r) in enumerate(profile):
        xy = axis_xy(z)
        out[i] = (xy[0], xy[1], z, r)
    return out


class LineXY:
    """Callable ``z -> (x, y)`` along the straight line p0 -> p1."""

    __slots__ = ("p0", "p1")

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)

    def __call__(self, z: float) -> np.ndarray:
        dz = self.p1[2] - self.p0[2]
        t = 0.0 if abs(dz) <= _EPS else (z - self.p0[2]) / dz
        return self.p0[:2] + (self.p1[:2] - self.p0[:2]) * t


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #


def foot_profile(z0: float, height: float, r_bottom: float, r_top: float):
    """The base: a straight-walled disc on the plate, then a flare to the shaft.

    The disc is the part that does the sticking. A base that starts at
    ``r_bottom`` and immediately tapers away has its full width for one layer
    only, so what actually grips the glass is a ring of extrusion; holding the
    width for a real distance gives a solid puck instead. It also makes the
    footprints of neighbouring supports *overlap* — with a lifted model there is
    a shaft roughly every ``support_spacing``, and their bases merge into a
    continuous raft that is far harder to knock off the plate than any number of
    separate feet.

    The flare above it narrows going up, so like every profile here it is
    self-supporting (``r' <= 0`` — see the module docstring). Degenerate cases
    collapse gracefully: a base no wider than the shaft is a stub of shaft, and
    a zero-height base is nothing at all.
    """
    r_top = float(r_top)
    r_bottom = max(float(r_bottom), r_top)
    height = float(height)
    if height <= _EPS:
        return [(z0, r_top)]
    # 45 degrees is the resin convention for the flare, but it does not always
    # have the room, and the disc has first claim on the height: it is the half
    # that earns the base its place.
    flare = min(r_bottom - r_top, height * FOOT_FLARE_SHARE)
    if flare <= _EPS:
        return [(z0, r_bottom), (z0 + height, r_top)]
    return [(z0, r_bottom), (z0 + height - flare, r_bottom), (z0 + height, r_top)]


def tip_profile(contact_z: float, tip_len: float, params: SupportParams, base_r: float):
    """The contact taper, narrowing as it rises, sunk into the model at the top."""
    tip_r = params.tip_diameter * 0.5
    pen = params.tip_penetration
    z0 = contact_z - tip_len
    profile = [(z0, base_r)]

    if params.tip_style == "spherical":
        # A ball tip, but only its upper half is emitted as geometry: the lower
        # hemisphere of a real sphere overhangs at 90 degrees, so it is replaced
        # by the taper that runs up to the ball's equator. What prints is a
        # ball-nose of diameter tip_diameter that touches at a point.
        r = tip_r
        z_eq = contact_z + pen - r
        if z_eq <= z0 + tip_len * 0.25:
            z_eq = z0 + tip_len * 0.5
        profile.append((z_eq, r))
        for deg in (30.0, 55.0, 75.0, 85.0):
            phi = math.radians(deg)
            profile.append((z_eq + r * math.sin(phi), r * math.cos(phi)))
    else:
        profile.append((contact_z, tip_r))
        profile.append((contact_z + pen, tip_r))
    return profile


# --------------------------------------------------------------------------- #
# public builders
# --------------------------------------------------------------------------- #


def make_tip(
    contact,
    params: SupportParams,
    base=None,
    base_diameter: float | None = None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """The bit that touches the model: a taper from the arm it grows out of up
    to ``params.tip_diameter``, sunk ``params.tip_penetration`` into the surface.

    Args:
        contact: (3,) point on the model surface.
        base: (3,) bottom of the taper. Defaults to ``params.tip_length``
            straight below the contact. A different XY makes the tip lean.
        base_diameter: width where the taper starts. Defaults to the shaft.
    """
    contact = np.asarray(contact, dtype=np.float64)
    sections = sections or params.ring_sections
    if base is None:
        base = contact - np.array([0.0, 0.0, params.tip_length])
    base = np.asarray(base, dtype=np.float64)

    tip_len = float(contact[2] - base[2])
    if tip_len <= _EPS:
        return trimesh.Trimesh()
    if base_diameter is None:
        base_diameter = params.shaft_lower_diameter
    axis = LineXY(base, contact + np.array([0.0, 0.0, params.tip_penetration]))
    profile = tip_profile(float(contact[2]), tip_len, params, float(base_diameter) * 0.5)
    return mesh_rings(rings_on_axis(axis, profile), sections)


def make_foot(
    base,
    params: SupportParams,
    top=None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """Bed adhesion. A bare shaft will not stay stuck to the plate, so a plate
    landing gets a ``params.foot_diameter`` disc, ``params.foot_height`` tall,
    flaring up to the shaft. See :func:`foot_profile`.
    """
    base = np.asarray(base, dtype=np.float64)
    sections = sections or params.ring_sections
    r_bottom = params.foot_diameter * 0.5
    r_top = params.shaft_lower_diameter * 0.5
    if top is None:
        top = base + np.array([0.0, 0.0, params.foot_height])
    top = np.asarray(top, dtype=np.float64)
    height = float(top[2] - base[2])
    if height <= _EPS:
        return trimesh.Trimesh()
    axis = LineXY(base, top)
    profile = foot_profile(float(base[2]), height, r_bottom, r_top)
    return mesh_rings(rings_on_axis(axis, profile), sections)


def make_strut(
    a,
    b,
    params: SupportParams,
    radius: float | None = None,
    cap_bottom: bool = True,
    cap_top: bool = True,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """A straight strut with horizontal cross-sections — arm, shaft or link.

    Sheared rather than rotated, so the end caps stay flat and horizontal. A
    rotated cap would be an overhang of ``90 - tilt``. Suppress a cap where the
    strut is buried in what it joins, rather than leaving two caps face to face:
    a flat downward face is a 90 degree overhang however deeply it is buried.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo, hi = (a, b) if a[2] <= b[2] else (b, a)
    if hi[2] - lo[2] <= _EPS:
        return trimesh.Trimesh()
    r = params.arm_diameter * 0.5 if radius is None else float(radius)
    axis = LineXY(lo, hi)
    profile = [(float(lo[2]), r), (float(hi[2]), r)]
    return mesh_rings(
        rings_on_axis(axis, profile), sections or params.ring_sections, cap_bottom, cap_top
    )


# --------------------------------------------------------------------------- #
# stage-3 entry point
# --------------------------------------------------------------------------- #


def build_supports(
    mesh,
    points,
    params: SupportParams | None = None,
    ray: DownRay | None = None,
) -> SupportBuild:
    """Turn a list of SupportPoint into printable support geometry.

    Returns SupportBuild (see types.py) with the supports-only mesh — the model
    is never touched, so the caller can export the two separately or concatenate
    them. Nothing about the input point list is cached, which is the point: the
    browser UI deletes and adds points and re-runs only this stage.

    Args:
        mesh: the oriented model, already sitting on z=0.
        points: list[SupportPoint] from stage 2, or an edited version of it.
        params: SupportParams; defaults to the built-in set.
        ray: a DownRay already built over ``mesh``. Pass it in when re-running
            this stage repeatedly — building the grid is the expensive part.
    """
    params = params or SupportParams()
    points = list(points)
    if not points:
        return SupportBuild(mesh=trimesh.Trimesh(), n_points=0)

    if ray is None:
        ray = DownRay(mesh)

    # Imported here rather than at module scope: resin.py lofts its geometry
    # from the primitives above, so a top-level import would be circular.
    from .resin import build_resin

    return build_resin(mesh, points, params, ray=ray)


# --------------------------------------------------------------------------- #
# invariant helper — shared by the tests and the CLI
# --------------------------------------------------------------------------- #


def overhang_angles(mesh) -> np.ndarray:
    """Per-face angle below horizontal, in degrees.

    0 for a vertical wall, 90 for a flat downward face, negative for anything
    facing upward. This is the quantity ``printable_overhang_deg`` caps.
    """
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return np.zeros(0)
    nz = np.clip(np.asarray(mesh.face_normals)[:, 2], -1.0, 1.0)
    return np.degrees(np.arcsin(-nz))
