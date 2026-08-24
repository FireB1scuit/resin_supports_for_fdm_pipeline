"""Shared data contracts for the pipeline.

The pipeline runs in three separable stages, and these types are the handoffs
between them:

    mesh --orient--> Orientation --sampling--> list[SupportPoint] --supports--> Trimesh

Stage 3 must stay cheap to re-run on an edited point list, which is why a
SupportPoint carries no geometry, only where a support should touch the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

TipStyle = Literal["conical", "spherical"]


@dataclass(frozen=True)
class SupportParams:
    """Every dimension the generator needs, in millimetres.

    Do not construct this directly with magic numbers — use
    :func:`rsupport.presets.from_nozzle`, which derives the whole set from the
    nozzle diameter. See CLAUDE.md.
    """

    # --- printer ---
    nozzle_diameter: float = 0.2
    layer_height: float = 0.08
    support_layer_height: float = 0.16

    # --- contact tip: the part that touches the model ---
    tip_diameter: float = 0.3
    tip_length: float = 1.5
    tip_penetration: float = 0.1
    tip_style: TipStyle = "conical"

    # --- scaffold ---
    # A support is base -> join cone -> vertical shaft -> angled arm -> tip.
    # The shaft stays thin the whole way, tapering slightly upward; it does not
    # thicken into a trunk.
    shaft_upper_diameter: float = 0.7
    shaft_lower_diameter: float = 1.2
    arm_diameter: float = 0.6
    # How far an arm may lean off vertical on its way up to a contact.
    arm_angle_deg: float = 45.0
    # 0 = one shaft per contact. 1 = arms reach far to share a shaft, so there
    # are far fewer shafts standing on the plate.
    parenting: float = 0.5
    # Height from one cross-link to the next up the *same* pair of shafts. Two
    # neighbouring pillars are tied at every multiple of this that fits between
    # them, so a tall pair gets a ladder and a short one gets a single rung. The
    # floor under it is physical: links closer together than their own combined
    # thickness are one lump, not two links.
    brace_interval: float = 8.0
    # Polygon sides per lofted ring.
    ring_sections: int = 12

    # --- routing and collision ---
    # How far any strut may lean off vertical. This is the whole printability
    # budget for anything that does not run straight up: a strut leaning by `a`
    # overhangs by exactly `a`, so it must stay under printable_overhang_deg.
    # It also sets how far a shaft can travel sideways per layer, and therefore
    # how well the scaffold routes around the model.
    strut_lean_deg: float = 40.0
    # Fattest strut the avoidance field samples a collision radius for.
    max_strut_diameter: float = 3.0
    # Gap kept between a strut and the model surface.
    xy_clearance: float = 0.4
    # Vertical sampling step for collision and reachability. Finer follows the
    # model more closely and costs more; this is Cura's "collision resolution".
    collision_pitch: float = 0.6
    # Supports may only ever land on the build plate. A contact the plate cannot
    # be routed to is left unsupported rather than propped off the sculpt.
    # Turning this off is what will one day allow a shaft to start on the model;
    # that half is not built yet, so off currently buys only the crude
    # stand-where-you-are fallback in `resin._drop_shaft`.
    plate_only: bool = True

    # --- cross-links ---
    brace_enabled: bool = True
    brace_diameter: float = 0.8
    # Furthest apart two shafts may be and still be linked.
    brace_max_span: float = 12.0
    # Height above the plate below which no cross-link is laid. 0 means "as low
    # as the geometry allows", which is half a base height up — the links start
    # inside the feet, where they cost nothing. Raise it to keep the bottom of
    # the scaffold open: that is where the base discs have already merged into a
    # raft and where a cutter has to get in first.
    brace_start_height: float = 0.0
    # Clear air kept under the model: no link's top end comes within this of a
    # shaft's top, which is where that shaft's arms leave for their contacts. A
    # link crowded up against the arm fan is the one there is no room to get a
    # blade to, and it is right where the model is most delicate.
    #
    # Deliberately not derived from the nozzle, and deliberately 0 by default.
    # It is spent out of a pair's window of linkable height, and that window is
    # set by how tall the shafts are — a property of the model. A coarse nozzle
    # makes shafts fatter, not taller, so scaling this with it takes the same
    # millimetres out of a shorter window: measured on the sample mini, a
    # headroom of one link diameter costs nothing at a 0.2 nozzle and a third
    # of the lattice at 0.4. It is a judgement about how much room you want to
    # get a cutter into, so it is left to whoever is holding the cutters.
    brace_headroom: float = 0.0
    # The angle a link is laid at, above horizontal. None takes the shallowest
    # angle the printable band allows, which is the most horizontal span per
    # millimetre of vertical run — and vertical run is what short shafts have
    # least of. A set value is clamped into that band: outside it a link
    # overhangs more than the printer can lay down.
    brace_angle_deg: float | None = None

    # --- base ---
    # Bed adhesion, and the one part of the scaffold deliberately *not* derived
    # from the nozzle: what holds onto a plate is a footprint measured in mm,
    # not a multiple of an extrusion width. A disc this wide and this tall sits
    # under every shaft that reaches the plate, and because a lifted model needs
    # a shaft roughly every `support_spacing`, neighbouring discs overlap into
    # something very close to a raft.
    foot_diameter: float = 10.0
    foot_height: float = 0.4

    # --- join cone ---
    # The transition piece between the base disc and the shaft — "base -> join
    # cone -> shaft" above. It used to be computed as a fraction of the disc's
    # own width and height, so widening the base for more bed adhesion also
    # puffed the cone out, and there was no way to size one without the other.
    # They are independent sliders now: the disc is bed adhesion, the cone is
    # what actually carries the shaft down onto it, and it is present at its
    # own size whatever the disc is doing, including when the disc is turned
    # off (`foot_height=0`). Clamped no wider than the disc it sits on — a cone
    # wider than what it stands on would flare outward on its way up, which is
    # an overhang.
    join_cone_diameter: float = 2.5
    join_cone_height: float = 1.0

    # --- placement ---
    # How far the whole model floats above the plate. This is what a resin
    # printer does — the sculpt hangs off the scaffold rather than being printed
    # against the plate — and on FDM it buys the same things: no elephant's foot
    # on the model, no first-layer squish across its underside, and a bottom
    # face that is supported like any other overhang instead of being ignored as
    # "on the bed". 0 sets the model down flat on the plate.
    lift_height: float = 5.0
    overhang_angle_deg: float = 45.0
    support_spacing: float = 3.0
    max_unsupported_span: float = 5.0
    island_min_area: float = 0.05

    # --- orientation ---
    max_lean_deg: float = 35.0
    printable_overhang_deg: float = 50.0

    def with_(self, **kwargs) -> SupportParams:
        """Return a copy with fields overridden. Ignores unknown keys."""
        known = {k: v for k, v in kwargs.items() if k in self.__dataclass_fields__}
        return replace(self, **known)

    @property
    def base_height(self) -> float:
        """Disc plus join cone: the full height of what a shaft stands on."""
        return max(0.0, self.foot_height) + max(0.0, self.join_cone_height)


@dataclass
class SupportPoint:
    """A single place on the model that wants holding up.

    Attributes:
        position: contact point on the model surface, in oriented model space.
        normal: outward surface normal at that point.
        forced: True for island points, which must never be thinned out.
        source: where it came from — "island", "overhang", "manual", "span".
    """

    position: np.ndarray
    normal: np.ndarray
    forced: bool = False
    source: str = "overhang"

    def as_dict(self) -> dict:
        return {
            "position": [float(v) for v in self.position],
            "normal": [float(v) for v in self.normal],
            "forced": bool(self.forced),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SupportPoint:
        return cls(
            position=np.asarray(d["position"], dtype=np.float64),
            normal=np.asarray(d["normal"], dtype=np.float64),
            forced=bool(d.get("forced", False)),
            source=str(d.get("source", "overhang")),
        )


@dataclass
class Orientation:
    """A candidate print pose and why it scored the way it did.

    `matrix` is a 4x4 homogeneous transform taking the *original* mesh into the
    oriented pose, already dropped so the lowest point sits at z=0 and centred
    on the origin in XY.
    """

    matrix: np.ndarray
    score: float
    terms: dict[str, float] = field(default_factory=dict)
    label: str = ""

    def as_dict(self) -> dict:
        return {
            "matrix": [float(v) for v in np.asarray(self.matrix).ravel()],
            "score": float(self.score),
            "terms": {k: float(v) for k, v in self.terms.items()},
            "label": self.label,
        }


@dataclass
class SupportBuild:
    """Result of stage 3."""

    mesh: object  # trimesh.Trimesh — untyped to keep this module import-light
    n_points: int = 0
    #: Cross-link struts built. A pair of shafts tied at four heights is four.
    n_braces: int = 0
    dropped: list[SupportPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Junction balls, as ``(x, y, z, radius)`` rows — every point where a strut
    #: meets a pillar, with the ball dropped on it to keep the corner solid in
    #: the exported soup of shells. See :func:`rsupport.resin._joint_radius`.
    #: They are inscribed in the pillar, so their undersides are the one place
    #: in the output where a steep face is not an overhang.
    joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def strut_lean(params: SupportParams) -> float:
    """Lean allowance off vertical, clamped below the printable limit.

    A strut leaning by `a` degrees overhangs by exactly `a`, so this is the
    entire printability budget for anything that does not run straight up.

    It lives here rather than in ``avoidance`` because both collision backends
    need it and neither may import the other. ``rsupport.avoidance`` re-exports
    it, which is where callers should still reach for it.
    """
    return float(np.clip(params.strut_lean_deg, 1.0, params.printable_overhang_deg - 2.0))
