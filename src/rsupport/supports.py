"""Stage 3 — turn support points into printable support geometry.

This is the module that actually makes the plastic. It takes the point list
stage 2 produced (or the one the user edited in the browser) and builds the
tips, pillars, feet and cross-braces that hold the model up.

The one rule everything here bends to
-------------------------------------
**Generated support geometry may never contain an overhang steeper than
``params.printable_overhang_deg``.** If our supports need supports, the
generator is broken. That single constraint decides most of the design below,
so it is worth spelling out the maths.

Take a surface of revolution whose horizontal cross-sections are circles of
radius ``r(z)`` whose centres drift sideways at rate ``m`` (mm of horizontal
travel per mm of height — a tilted pillar). Its outward normal at azimuth ``u``
has a vertical component proportional to ``-(m·u + r'(z))``, so the steepest
overhang anywhere on that surface is::

    angle_below_horizontal = atan(|m| + max(r', 0))

which gives three design rules used throughout:

* A profile that **narrows going up** (``r' <= 0``) is always self-supporting.
  Hence the tip cone is wide at the bottom and thin at the top, the foot is a
  cone flaring *downward*, and the "spherical" tip is a dome rather than a ball
  (a free-floating sphere's underside is a 90 degree overhang).
* Tilting a pillar by ``t`` costs exactly ``t`` degrees of overhang budget, so
  ``max_pillar_tilt_deg`` must stay under ``printable_overhang_deg``.
* A **flat downward face is a 90 degree overhang**, always. So a support column
  is built as one continuous stack of rings — foot, pillar, tip, all in a
  single shell — rather than as separate capped primitives stacked on top of
  each other. There are exactly two flat caps per column: the bottom one, which
  rests on the build plate or is sunk into the model, and the top one, which
  faces up. No interior caps exist to violate the invariant.

Braces are the exception that proves the rule: a plain tilted cylinder at angle
``a`` above horizontal has side walls overhanging by ``90 - a`` and end caps
overhanging by ``a``. Both are printable only in the band
``[90 - printable, printable]`` — with the defaults, roughly 40 to 50 degrees.
That is why braces are placed at a *chosen* angle (``brace_min_angle_deg``,
clamped into that band) by picking the two attachment heights, instead of just
connecting two convenient points.

Nothing here casts a general ray: ``rsupport.raycast.DownRay`` only does
straight down, which is all this stage needs, and a tilted column's axis is
still a straight segment so ``segment_blocked`` handles it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh_io import concat
from .raycast import DownRay
from .types import SupportBuild, SupportParams, SupportPoint

__all__ = [
    "build_supports",
    "make_tip",
    "make_pillar",
    "make_foot",
    "make_brace",
    "Column",
    "overhang_angles",
]

_EPS = 1e-9

# How many azimuths / tilt steps the collision search tries before giving up.
_TILT_STEPS = 3
_TILT_AZIMUTHS = 8
# Rays fired around a pillar's circumference to catch the model clipping its
# side rather than its axis.
_PERIMETER_PROBES = 6
# Azimuths tried when dropping an angled prop to the plate.
_PROP_AZIMUTHS = 12
# A tall pillar gets at most this many struts; more is filament and misery.
_MAX_BRACES_PER_COLUMN = 2

# Pad fitting: how many radii to try, and how many probes around each ring.
_PAD_STEPS = 7
_PAD_RING = 12
_PAD_SINK_STEPS = 4
_PAD_NUDGE_STEPS = 4
_PAD_NUDGE_FRACTION = 0.7


# --------------------------------------------------------------------------- #
# column bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class Column:
    """One routed support column: where it touches, where it lands, how it leans.

    The axis is the straight line from ``base`` to ``contact``. Cross-sections
    stay horizontal circles centred on that line, so a leaning column is a
    sheared cylinder rather than a rotated one — which keeps its end caps flat
    and horizontal instead of tipping them into a steep overhang.
    """

    point: SupportPoint
    contact: np.ndarray  # (3,) contact point on the model surface
    base: np.ndarray  # (3,) centre of the bottom cap (already sunk if on model)
    land_z: float  # surface the column rests on
    on_model: bool  # True: landed on the model, False: on the build plate
    foot_h: float  # height of the flared foot / pad section
    tip_len: float  # length of the contact taper
    tilt_deg: float = 0.0
    base_r: float = 0.0  # bottom cap radius; 0 means "use the parameter default"

    @property
    def top_z(self) -> float:
        return float(self.contact[2])

    @property
    def z_att_lo(self) -> float:
        """Lowest height a brace may attach at."""
        return float(self.base[2]) + self.foot_h

    @property
    def z_att_hi(self) -> float:
        """Highest height a brace may attach at — below the tip taper."""
        return self.top_z - self.tip_len

    @property
    def free_height(self) -> float:
        """Length of the bare pillar: what actually buckles."""
        return max(0.0, self.z_att_hi - self.z_att_lo)

    def xy_at(self, z: float) -> np.ndarray:
        z0, z1 = float(self.base[2]), self.top_z
        t = 0.0 if z1 - z0 <= _EPS else (z - z0) / (z1 - z0)
        return self.base[:2] + (self.contact[:2] - self.base[:2]) * t

    def point_at(self, z: float) -> np.ndarray:
        xy = self.xy_at(z)
        return np.array([xy[0], xy[1], z], dtype=np.float64)


# --------------------------------------------------------------------------- #
# primitive construction — rings in, mesh out
# --------------------------------------------------------------------------- #


def _mesh_rings(
    rings: np.ndarray,
    sections: int,
    cap_bottom: bool = True,
    cap_top: bool = True,
) -> trimesh.Trimesh:
    """Loft a stack of horizontal circles into a closed shell.

    Args:
        rings: (M, 4) array of ``(x, y, z, radius)``, ordered bottom to top.
            The centres may drift in XY — that is how a leaning column is built.
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


def _rings_on_axis(axis_xy, profile) -> np.ndarray:
    """Attach ``(z, radius)`` pairs to an axis, giving ``(x, y, z, r)`` rings."""
    out = np.empty((len(profile), 4), dtype=np.float64)
    for i, (z, r) in enumerate(profile):
        xy = axis_xy(z)
        out[i] = (xy[0], xy[1], z, r)
    return out


# --- profiles ------------------------------------------------------------- #


def _foot_profile(z0: float, height: float, r_bottom: float, r_top: float):
    """A cone flaring *downward*: every layer is smaller than the one below."""
    return [(z0, max(r_bottom, r_top)), (z0 + height, r_top)]


def _tip_profile(contact_z: float, tip_len: float, params: SupportParams, base_r: float):
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


# --- public builders ------------------------------------------------------ #


def make_tip(
    contact,
    params: SupportParams,
    base=None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """The bit that touches the model: a taper from pillar width up to
    ``params.tip_diameter``, sunk ``params.tip_penetration`` into the surface.

    Args:
        contact: (3,) point on the model surface.
        base: (3,) bottom of the taper. Defaults to ``params.tip_length``
            straight below the contact. A different XY makes the tip lean.
    """
    contact = np.asarray(contact, dtype=np.float64)
    sections = sections or params.pillar_sections
    if base is None:
        base = contact - np.array([0.0, 0.0, params.tip_length])
    base = np.asarray(base, dtype=np.float64)

    tip_len = float(contact[2] - base[2])
    if tip_len <= _EPS:
        return trimesh.Trimesh()
    axis = _LineXY(base, contact + np.array([0.0, 0.0, params.tip_penetration]))
    profile = _tip_profile(float(contact[2]), tip_len, params, params.pillar_diameter * 0.5)
    return _mesh_rings(_rings_on_axis(axis, profile), sections)


def make_pillar(
    base,
    top,
    params: SupportParams,
    radius: float | None = None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """A straight column between two points, with horizontal cross-sections.

    Leaning it shears the cylinder rather than rotating it, so the end caps stay
    flat — a rotated cap would be an overhang of ``90 - tilt`` degrees.
    """
    base = np.asarray(base, dtype=np.float64)
    top = np.asarray(top, dtype=np.float64)
    if top[2] - base[2] <= _EPS:
        return trimesh.Trimesh()
    r = params.pillar_diameter * 0.5 if radius is None else float(radius)
    axis = _LineXY(base, top)
    profile = [(float(base[2]), r), (float(top[2]), r)]
    return _mesh_rings(_rings_on_axis(axis, profile), sections or params.pillar_sections)


def make_foot(
    base,
    params: SupportParams,
    on_model: bool = False,
    top=None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """Bed adhesion. A bare 1.2 mm pillar will not stay stuck to the plate, so
    plate landings get a ``params.foot_diameter`` cone; model landings get the
    smaller ``params.pad_diameter`` pad, which only has to spread the load.
    """
    base = np.asarray(base, dtype=np.float64)
    sections = sections or params.pillar_sections
    height = params.pad_height if on_model else params.foot_height
    r_bottom = (params.pad_diameter if on_model else params.foot_diameter) * 0.5
    r_top = params.pillar_diameter * 0.5
    if top is None:
        top = base + np.array([0.0, 0.0, height])
    top = np.asarray(top, dtype=np.float64)
    height = float(top[2] - base[2])
    if height <= _EPS:
        return trimesh.Trimesh()
    axis = _LineXY(base, top)
    profile = _foot_profile(float(base[2]), height, r_bottom, r_top)
    return _mesh_rings(_rings_on_axis(axis, profile), sections)


def make_brace(
    p0,
    p1,
    params: SupportParams,
    radius: float | None = None,
    sections: int | None = None,
) -> trimesh.Trimesh:
    """A diagonal strut between two points.

    Unlike the columns this *is* a rotated cylinder, because its end caps are
    buried in the pillars it joins and its axis angle is chosen to keep both the
    side walls (``90 - angle``) and the caps (``angle``) inside the printable
    limit. See ``_brace_angle``.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if np.linalg.norm(p1 - p0) <= _EPS:
        return trimesh.Trimesh()
    r = params.brace_diameter * 0.5 if radius is None else float(radius)
    return trimesh.creation.cylinder(
        radius=r, segment=[p0, p1], sections=sections or params.pillar_sections
    )


class _LineXY:
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
# routing: where does each column actually go?
# --------------------------------------------------------------------------- #


def _segment_flags(
    ray: DownRay,
    a: np.ndarray,
    b: np.ndarray,
    trim: float,
    offsets: np.ndarray | None = None,
    samples: int = 12,
) -> np.ndarray:
    """Does the (optionally offset) segment a->b pass through the model?

    ``trim`` shortens both ends, because both ends touch the model by design:
    the top is on the surface, the bottom rests on it.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    d = b - a
    length = np.linalg.norm(d, axis=1)
    good = length > 1e-6
    t = np.zeros(len(a))
    t[good] = np.clip(trim / length[good], 0.0, 0.45)
    a2 = a + d * t[:, None]
    b2 = b - d * t[:, None]
    if offsets is not None:
        a2 = a2 + offsets
        b2 = b2 + offsets
    flags = ray.segment_blocked(a2, b2, samples=samples)
    return flags & good


def _perimeter_blocked(
    ray: DownRay, a: np.ndarray, b: np.ndarray, radius: float, trim: float
) -> np.ndarray:
    """Ring of probes at the pillar's surface — catches the model clipping the
    side of a column whose axis happens to be clear."""
    n = len(a)
    hit = np.zeros(n, dtype=bool)
    for i in range(_PERIMETER_PROBES):
        phi = 2.0 * np.pi * i / _PERIMETER_PROBES
        off = np.zeros((n, 3))
        off[:, 0] = radius * np.cos(phi)
        off[:, 1] = radius * np.sin(phi)
        hit |= _segment_flags(ray, a, b, trim, offsets=off, samples=10)
    return hit


def _landing(
    ray: DownRay, xy: np.ndarray, from_z: np.ndarray, plate_eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Highest *exposed* model surface below each (xy, from_z), else the plate.

    Not every surface is somewhere a pillar can stand. Miniatures are routinely
    shipped as interpenetrating shells — an arm pushed into a torso, a cape
    overlapping a back — and the highest surface below a contact is often the
    part of one shell that lies buried inside another. A pillar landing there
    starts inside solid sculpt: it prints, but it is fused into the model and
    can never be snapped off.

    So the whole column of crossings is walked from the top. Counting crossings
    above a height gives its parity, and a landing is only real when the space
    just above it is outside the model — even index in the descending list.
    """
    probe = np.column_stack([xy, from_z])
    n = len(probe)
    out_z = np.zeros(n)
    out_on = np.zeros(n, dtype=bool)

    pidx, _, z = ray.column_hits(probe)
    if len(pidx) == 0:
        return out_z, out_on

    order = np.lexsort((-z, pidx))  # grouped by point, height descending
    pidx, z = pidx[order], z[order]

    # Merge crossings at the same height: a column running along a shared edge
    # hits both triangles and would otherwise flip the parity.
    tol = max(float(np.ptp(ray.bounds[:, 2])), 1.0) * 1e-9
    first = np.empty(len(z), dtype=bool)
    first[0] = True
    first[1:] = (pidx[1:] != pidx[:-1]) | (np.abs(np.diff(z)) > tol)
    pidx, z = pidx[first], z[first]

    starts = np.searchsorted(pidx, np.arange(n), side="left")
    depth = np.arange(len(z)) - starts[pidx]

    exposed = (depth % 2 == 0) & (z < probe[pidx, 2] - 1e-4)
    if exposed.any():
        best = np.full(n, -np.inf)
        np.maximum.at(best, pidx[exposed], z[exposed])
        found = np.isfinite(best)
        out_on = found & (best > plate_eps)
        out_z = np.where(out_on, np.where(found, best, 0.0), 0.0)
    return out_z, out_on


def _section_lengths(avail: float, params: SupportParams, on_model: bool) -> tuple[float, float]:
    """Split a column's height into foot / tip, compressing short columns."""
    foot_target = params.pad_height if on_model else params.foot_height
    foot_h = min(foot_target, avail * 0.35)
    tip_len = min(params.tip_length, avail * 0.6)
    return foot_h, tip_len


def _make_column(
    pt: SupportPoint,
    contact: np.ndarray,
    base_xy: np.ndarray,
    land_z: float,
    on_model: bool,
    params: SupportParams,
    tilt_deg: float,
) -> Column:
    sink = params.tip_penetration if on_model else 0.0
    z_bot = land_z - sink
    base = np.array([base_xy[0], base_xy[1], z_bot], dtype=np.float64)
    avail = float(contact[2]) - z_bot
    foot_h, tip_len = _section_lengths(avail, params, on_model)
    return Column(
        point=pt,
        contact=contact,
        base=base,
        land_z=land_z,
        on_model=on_model,
        foot_h=foot_h,
        tip_len=tip_len,
        tilt_deg=tilt_deg,
    )


def _fit_pads(ray: DownRay, columns: list[Column], params: SupportParams) -> int:
    """Shrink each model-landing pad to the material actually underneath it.

    A pad wider than the feature it lands on leaves its rim hanging: the centre
    is anchored in the surface, the edge is a 90 degree overhang over nothing.
    On a miniature that is the common case, not the exception — pillars land on
    arms, shoulders and cape folds barely wider than the pillar itself — so the
    pad is measured rather than assumed.

    A ring of probes is dropped at each candidate radius, largest first. Since
    the pad is sunk ``tip_penetration`` into the surface, a probe just above the
    pad's underside is inside the model wherever the pad is genuinely resting.
    """
    idx = [i for i, c in enumerate(columns) if c.on_model]
    if not idx:
        return 0

    pillar_r = params.pillar_diameter * 0.5
    r_hi = max(params.pad_diameter * 0.5, pillar_r * 1.05)
    # The pad is allowed *narrower* than the pillar. Flaring outward on the way
    # up is an overhang, but a gentle one, and _column_mesh lengthens the foot
    # to keep that flare inside the printable slope. A narrow base that rests on
    # a thin blade beats a wide one whose rim hangs off both edges.
    r_lo = max(params.nozzle_diameter * 0.75, pillar_r * 0.35)
    radii = np.linspace(r_hi, r_lo, _PAD_STEPS) if r_hi > r_lo else np.array([r_lo])

    phi = np.arange(_PAD_RING) * (2.0 * np.pi / _PAD_RING)
    ring = np.column_stack([np.cos(phi), np.sin(phi)])
    lift = params.tip_penetration * 0.25

    probes = np.empty((len(idx), len(radii), _PAD_RING, 3))
    for k, i in enumerate(idx):
        b = columns[i].base
        probes[k, :, :, 0] = b[0] + ring[None, :, 0] * radii[:, None]
        probes[k, :, :, 1] = b[1] + ring[None, :, 1] * radii[:, None]
        probes[k, :, :, 2] = b[2] + lift

    supported = ray.inside(probes.reshape(-1, 3)).reshape(len(idx), len(radii), _PAD_RING)
    whole_ring = supported.all(axis=2)

    stubborn: list[int] = []
    for k, i in enumerate(idx):
        # radii run largest to smallest, so the first True is the widest that fits.
        if whole_ring[k].any():
            columns[i].base_r = float(radii[int(np.argmax(whole_ring[k]))])
        else:
            columns[i].base_r = float(radii[-1])
            stubborn.append(i)

    if stubborn:
        return _settle_stubborn_pads(ray, columns, stubborn, params, ring, lift)
    return 0


def _settle_stubborn_pads(
    ray: DownRay,
    columns: list[Column],
    stubborn: list[int],
    params: SupportParams,
    ring: np.ndarray,
    lift: float,
) -> int:
    """Rescue pads that will not rest flat even at the narrowest radius.

    Two moves, in order:

    1. **Slide inboard.** A contact near the edge of a feature lands its pad
       half off. Shifting the base away from the unsupported side puts it back
       on the material, and over a pillar tens of millimetres tall that shift is
       a fraction of a degree of lean — far cheaper than losing the support.
    2. **Sink deeper.** A cap buried in solid material is not an overhang at
       all, and a curved feature widens below its crown. Bounded to one pillar
       diameter; past that we are skewering the sculpt rather than standing on
       it.

    Returns how many are still unresolved.
    """
    depths = np.linspace(0.0, params.pillar_diameter, _PAD_SINK_STEPS + 1)[1:]
    unresolved = 0

    for i in stubborn:
        col = columns[i]

        def ring_flags(base: np.ndarray, dz: float = 0.0) -> np.ndarray:
            probe = np.column_stack(
                [
                    base[0] + ring[:, 0] * col.base_r,
                    base[1] + ring[:, 1] * col.base_r,
                    np.full(len(ring), base[2] + lift - dz),
                ]
            )
            return ray.inside(probe)

        base = col.base.copy()
        settled = False
        for _ in range(_PAD_NUDGE_STEPS):
            flags = ring_flags(base)
            if flags.all():
                settled = True
                break
            # Step away from the unsupported arc, towards the material.
            away = ring[~flags].mean(axis=0)
            norm = float(np.linalg.norm(away))
            if norm < _EPS:
                break
            base[:2] -= (away / norm) * col.base_r * _PAD_NUDGE_FRACTION

        if not settled:
            for extra in depths:
                if ring_flags(base, extra).all():
                    base[2] -= extra
                    settled = True
                    break

        if settled:
            col.base = base
        else:
            unresolved += 1

    return unresolved


def _route_columns(
    ray: DownRay,
    points: list[SupportPoint],
    params: SupportParams,
) -> tuple[list[Column], list[SupportPoint], list[str]]:
    """Decide where every column lands, resolving collisions.

    Order of attack per point, as specified in PLAN.md stage 3:

    1. straight down onto whatever ``z_below`` finds (model or plate);
    2. if that is blocked, lean the pillar up to ``max_pillar_tilt_deg``,
       trying a few azimuths (starting in the direction the surface faces,
       which is the direction with the most free air);
    3. each leaned candidate independently re-checks whether it should land on
       the model instead of the plate;
    4. if the *axis* is clear and only the pillar's flank clips the model,
       keep the vertical column and warn — a slightly fused support is far
       better than a missing one;
    5. otherwise drop the point and record it.
    """
    contacts = np.array([np.asarray(p.position, dtype=np.float64) for p in points])
    n = len(contacts)
    plate_eps = max(1e-6, params.layer_height * 0.5)
    pillar_r = params.pillar_diameter * 0.5
    axis_trim = max(params.layer_height * 2.0, 1e-4)
    flank_trim = max(params.pillar_diameter, params.tip_length)
    min_height = max(params.tip_penetration * 2.0, params.layer_height * 2.0)

    land_z, on_model = _landing(ray, contacts[:, :2], contacts[:, 2], plate_eps)

    warnings: list[str] = []
    columns: list[Column] = []
    dropped: list[SupportPoint] = []

    too_short = (contacts[:, 2] - land_z) < min_height

    base0 = np.column_stack([contacts[:, 0], contacts[:, 1], land_z])
    axis_bad = _segment_flags(ray, base0, contacts, axis_trim)
    flank_bad = _perimeter_blocked(ray, base0, contacts, pillar_r * 0.95, flank_trim)

    need_reroute = np.where(~too_short & (axis_bad | flank_bad))[0]
    solved: dict[int, tuple[np.ndarray, float, bool, float]] = {}
    if len(need_reroute):
        solved = _tilt_search(ray, contacts, points, need_reroute, params, plate_eps)

    n_flank_only = 0
    for i, pt in enumerate(points):
        if too_short[i]:
            dropped.append(pt)
            continue
        if i in solved:
            base_xy, lz, om, tilt = solved[i]
            columns.append(_make_column(pt, contacts[i], base_xy, lz, om, params, tilt))
            continue
        if axis_bad[i]:
            dropped.append(pt)
            continue
        if flank_bad[i]:
            n_flank_only += 1
        columns.append(
            _make_column(pt, contacts[i], contacts[i, :2], float(land_z[i]), bool(on_model[i]),
                         params, 0.0)
        )

    n_short = int(too_short.sum())
    if n_short:
        warnings.append(
            f"{n_short} point(s) dropped: less than {min_height:.3f} mm of clearance "
            "below the contact, nothing to build a support out of"
        )
    n_hard = len(dropped) - n_short
    if n_hard > 0:
        warnings.append(
            f"{n_hard} point(s) dropped: the pillar would pass through the model and "
            f"no lean up to {params.max_pillar_tilt_deg:.0f} deg found a clear line"
        )
    if n_flank_only:
        warnings.append(
            f"{n_flank_only} pillar(s) graze the model along their flank; kept vertical "
            "(they will fuse to the model and need cutting rather than snapping)"
        )
    n_tilted = sum(1 for c in columns if c.tilt_deg > 0.0)
    if n_tilted:
        warnings.append(f"{n_tilted} pillar(s) leaned to clear the model")

    stuck = _fit_pads(ray, columns, params)
    if stuck:
        warnings.append(
            f"{stuck} pillar(s) land on a feature too small to rest a pad on; their base "
            "will partly overhang (a short bridge, not a floating island)"
        )

    return columns, dropped, warnings


def _tilt_search(
    ray: DownRay,
    contacts: np.ndarray,
    points: list[SupportPoint],
    idx: np.ndarray,
    params: SupportParams,
    plate_eps: float,
) -> dict[int, tuple[np.ndarray, float, bool, float]]:
    """Batched lean search. Returns ``{point_index: (base_xy, land_z, on_model, tilt)}``.

    Candidates are ordered by increasing tilt (the gentlest fix wins) and, within
    a tilt, starting from the direction the surface normal points, since that is
    where the free air is.
    """
    pillar_r = params.pillar_diameter * 0.5
    axis_trim = max(params.layer_height * 2.0, 1e-4)
    flank_trim = max(params.pillar_diameter, params.tip_length)
    max_tilt = max(0.0, float(params.max_pillar_tilt_deg))
    if max_tilt <= _EPS:
        return {}

    tilts = [max_tilt * (k + 1) / _TILT_STEPS for k in range(_TILT_STEPS)]

    # Preferred azimuth per point: the horizontal part of the surface normal.
    prefer = np.zeros(len(contacts))
    for i in idx:
        nrm = np.asarray(points[i].normal, dtype=np.float64)
        if np.linalg.norm(nrm[:2]) > 1e-6:
            prefer[i] = math.atan2(nrm[1], nrm[0])

    cand_pt: list[int] = []
    cand_tilt: list[float] = []
    cand_dir: list[np.ndarray] = []
    for tilt in tilts:
        m = math.tan(math.radians(tilt))
        for a in range(_TILT_AZIMUTHS):
            dphi = 2.0 * np.pi * a / _TILT_AZIMUTHS
            for i in idx:
                phi = prefer[i] + dphi
                cand_pt.append(int(i))
                cand_tilt.append(tilt)
                cand_dir.append(np.array([math.cos(phi) * m, math.sin(phi) * m]))

    if not cand_pt:
        return {}

    cp = np.asarray(cand_pt)
    cd = np.asarray(cand_dir)
    top = contacts[cp]

    # First pass assumes a plate landing, then re-solves the base once the real
    # landing height is known (the lean's horizontal run scales with height).
    base_xy = top[:, :2] + cd * top[:, 2:3]
    lz, om = _landing(ray, base_xy, top[:, 2], plate_eps)
    base_xy = top[:, :2] + cd * (top[:, 2] - lz)[:, None]
    lz, om = _landing(ray, base_xy, top[:, 2], plate_eps)
    base = np.column_stack([base_xy, lz])

    ok = (top[:, 2] - lz) > max(params.tip_penetration * 2.0, params.layer_height * 2.0)
    ok &= ~_segment_flags(ray, base, top, axis_trim)
    ok &= ~_perimeter_blocked(ray, base, top, pillar_r * 0.95, flank_trim)

    out: dict[int, tuple[np.ndarray, float, bool, float]] = {}
    for j in np.where(ok)[0]:
        i = int(cp[j])
        if i in out:
            continue  # candidates are already in preference order
        out[i] = (base_xy[j], float(lz[j]), bool(om[j]), float(cand_tilt[j]))
    return out


# --------------------------------------------------------------------------- #
# cross-braces
# --------------------------------------------------------------------------- #


def _brace_angle(params: SupportParams) -> float | None:
    """The angle above horizontal every brace is built at, or None if the
    parameters leave no printable band.

    A cylindrical strut at angle ``a`` has side walls overhanging ``90 - a`` and
    end caps overhanging ``a``, so both ends of the band are hard limits.
    """
    limit = float(params.printable_overhang_deg)
    lo = max(float(params.brace_min_angle_deg), 90.0 - limit + 2.0)
    hi = limit - 2.0
    if lo > hi:
        return None
    return lo


def _build_braces(
    ray: DownRay,
    columns: list[Column],
    params: SupportParams,
) -> tuple[list[trimesh.Trimesh], int, list[str]]:
    """Strut the slender pillars.

    Only columns whose bare length exceeds ``brace_slenderness * pillar_diameter``
    start a brace — bracing everything wastes filament and makes the supports
    miserable to remove. Each such column reaches for its nearest neighbours
    within ``brace_max_span``; the attachment heights are chosen so the strut
    lands on the fixed printable angle. A column with nobody in reach gets an
    angled prop down to the plate instead.
    """
    parts: list[trimesh.Trimesh] = []
    warnings: list[str] = []
    if not params.brace_enabled or len(columns) == 0:
        return parts, 0, warnings

    alpha = _brace_angle(params)
    if alpha is None:
        warnings.append(
            f"bracing disabled: brace_min_angle_deg={params.brace_min_angle_deg:.0f} leaves no "
            f"printable band under printable_overhang_deg={params.printable_overhang_deg:.0f}"
        )
        return parts, 0, warnings
    tan_a = math.tan(math.radians(alpha))

    threshold = params.brace_slenderness * params.pillar_diameter
    tall = [i for i, c in enumerate(columns) if c.free_height > threshold]
    if not tall:
        return parts, 0, warnings

    anchors = np.array([c.xy_at(0.5 * (c.z_att_lo + c.z_att_hi)) for c in columns])
    tree = cKDTree(anchors)

    made: set[frozenset] = set()
    count = np.zeros(len(columns), dtype=int)
    n_props = 0

    # Tallest first: they are the ones most in need of an anchor.
    for i in sorted(tall, key=lambda k: -columns[k].free_height):
        if count[i] >= _MAX_BRACES_PER_COLUMN:
            continue
        neigh = tree.query_ball_point(anchors[i], params.brace_max_span)
        order = sorted(
            (j for j in neigh if j != i),
            key=lambda j: float(np.linalg.norm(anchors[j] - anchors[i])),
        )
        linked = False
        for j in order:
            if frozenset((i, j)) in made:
                # A strut to this neighbour already exists, and a strut braces
                # both of its ends. This column is served — without this the
                # second pillar of a braced pair fell through to _fit_prop and
                # got a redundant prop to the plate.
                linked = True
                break
            if count[j] >= _MAX_BRACES_PER_COLUMN + 1:
                continue
            seg = _fit_brace(ray, columns[i], columns[j], tan_a, params)
            if seg is None:
                continue
            parts.append(make_brace(seg[0], seg[1], params))
            made.add(frozenset((i, j)))
            count[i] += 1
            count[j] += 1
            linked = True
            break
        if not linked:
            prop = _fit_prop(ray, columns[i], tan_a, params)
            if prop is not None:
                parts.extend(prop)
                count[i] += 1
                n_props += 1

    n_braces = len(made) + n_props
    if tall and n_braces == 0:
        warnings.append(
            f"{len(tall)} slender pillar(s) could not be braced (no neighbour within "
            f"{params.brace_max_span:.1f} mm and no clear line to the plate)"
        )
    return parts, n_braces, warnings


def _fit_brace(
    ray: DownRay,
    a: Column,
    b: Column,
    tan_a: float,
    params: SupportParams,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick attachment heights on two columns giving a strut at the target angle.

    Endpoints sit exactly on the pillar axes, so each end cap (radius
    ``brace_diameter/2``) is fully buried inside the pillar it joins.
    """
    margin = params.brace_diameter
    limit = float(params.printable_overhang_deg)

    for hi_col, lo_col in ((a, b), (b, a)):
        hi_lo, hi_hi = hi_col.z_att_lo + margin, hi_col.z_att_hi - margin
        lo_lo, lo_hi = lo_col.z_att_lo + margin, lo_col.z_att_hi - margin
        if hi_hi <= hi_lo or lo_hi <= lo_lo:
            continue

        z_hi = hi_hi
        for _ in range(3):  # a couple of passes settle sheared (leaning) columns
            p_hi_xy = hi_col.xy_at(z_hi)
            span = float(np.linalg.norm(p_hi_xy - lo_col.xy_at(z_hi)))
            if span < params.pillar_diameter * 1.5 or span > params.brace_max_span:
                z_hi = None
                break
            z_lo = z_hi - span * tan_a
            if z_lo < lo_lo:
                drop = lo_lo - z_lo
                z_hi -= drop
                if z_hi < hi_lo:
                    z_hi = None
                    break
                z_lo = lo_lo
            if z_lo > lo_hi:
                shift = z_lo - lo_hi
                z_hi -= shift
                z_lo -= shift
                if z_hi < hi_lo or z_lo < lo_lo:
                    z_hi = None
                    break
            break
        if z_hi is None:
            continue

        p_hi = hi_col.point_at(z_hi)
        p_lo = lo_col.point_at(z_lo)
        d = p_hi - p_lo
        run = float(np.linalg.norm(d[:2]))
        if run < _EPS or d[2] <= _EPS:
            continue
        ang = math.degrees(math.atan2(d[2], run))
        # Hard gate: outside this band the strut itself would need support.
        if ang < 90.0 - limit + 0.5 or ang > limit - 0.5:
            continue
        if float(np.linalg.norm(d)) > params.brace_max_span * 1.5:
            continue
        if _segment_flags(ray, p_lo[None, :], p_hi[None, :], 0.0)[0]:
            continue
        return p_lo, p_hi
    return None


def _fit_prop(
    ray: DownRay,
    col: Column,
    tan_a: float,
    params: SupportParams,
) -> list[trimesh.Trimesh] | None:
    """An angled strut from a lonely tall pillar down to the build plate."""
    margin = params.brace_diameter
    pad_h = max(params.pad_height, params.brace_diameter)
    z_hi = col.z_att_hi - margin
    if z_hi <= col.z_att_lo + margin:
        return None
    # Keep the horizontal run inside brace_max_span, starting lower if need be.
    run = (z_hi - pad_h) / tan_a
    if run > params.brace_max_span:
        run = params.brace_max_span
        z_hi = pad_h + run * tan_a
        if z_hi > col.z_att_hi - margin or z_hi <= col.z_att_lo + margin:
            return None
    if run <= _EPS:
        return None

    top = col.point_at(z_hi)
    for k in range(_PROP_AZIMUTHS):
        phi = 2.0 * np.pi * k / _PROP_AZIMUTHS
        foot_xy = top[:2] + np.array([math.cos(phi), math.sin(phi)]) * run
        base = np.array([foot_xy[0], foot_xy[1], pad_h])
        if ray.inside(base[None, :])[0]:
            continue
        # The plate under the prop's foot must be free of model.
        under = np.array([[foot_xy[0], foot_xy[1], pad_h * 0.5]])
        if ray.inside(under)[0]:
            continue
        if _segment_flags(ray, base[None, :], top[None, :], 0.0)[0]:
            continue
        strut = make_brace(base, top, params)
        pad_axis = _LineXY(
            np.array([foot_xy[0], foot_xy[1], 0.0]), np.array([foot_xy[0], foot_xy[1], pad_h])
        )
        pad = _mesh_rings(
            _rings_on_axis(
                pad_axis,
                _foot_profile(0.0, pad_h, params.pad_diameter * 0.5, params.brace_diameter * 0.5),
            ),
            params.pillar_sections,
        )
        return [strut, pad]
    return None


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def _column_mesh(col: Column, params: SupportParams) -> trimesh.Trimesh:
    """Foot, pillar and tip as one continuous shell.

    Deliberately *not* three stacked primitives: a stack leaves interior flat
    caps, and a flat downward face is a 90 degree overhang no matter how deeply
    it is buried. One ring stack has exactly two caps, both legitimate.
    """
    pillar_r = params.pillar_diameter * 0.5
    z_bot = float(col.base[2])
    z_top = col.top_z
    foot_h = col.foot_h
    if col.base_r > 0.0:
        r_base = col.base_r  # measured against what is actually under the pad
        if r_base < pillar_r:
            # Flaring out to full pillar width is an overhang of atan(dr/dh);
            # give it enough height to stay inside the printable limit.
            slope = math.tan(math.radians(min(params.printable_overhang_deg, 89.0)))
            needed = (pillar_r - r_base) / max(slope, _EPS)
            foot_h = min(max(foot_h, needed), max(z_top - z_bot - col.tip_len, 0.0))
    else:
        r_base = (params.pad_diameter if col.on_model else params.foot_diameter) * 0.5
        r_base = max(r_base, pillar_r * 1.05)

    profile = _foot_profile(z_bot, foot_h, r_base, pillar_r)
    profile.append((z_top - col.tip_len, pillar_r))
    profile.extend(_tip_profile(z_top, col.tip_len, params, pillar_r)[1:])

    axis = _LineXY(col.base, col.contact + np.array([0.0, 0.0, params.tip_penetration]))
    return _mesh_rings(_rings_on_axis(axis, profile), params.pillar_sections)


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

    if params.support_style == "tree":
        # Imported here rather than at module scope: tree.py reuses the ring
        # and profile machinery below, so a top-level import would be circular.
        from .tree import build_tree

        return build_tree(mesh, points, params, ray=ray)

    columns, dropped, warnings = _route_columns(ray, points, params)
    parts = [_column_mesh(c, params) for c in columns]

    brace_parts, n_braces, brace_warnings = _build_braces(ray, columns, params)
    parts.extend(brace_parts)
    warnings.extend(brace_warnings)

    return SupportBuild(
        mesh=concat(*parts),
        n_points=len(columns),
        n_braces=n_braces,
        dropped=dropped,
        warnings=warnings,
    )


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
