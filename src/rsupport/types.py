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

    # --- pillar ---
    pillar_diameter: float = 1.2
    pillar_sections: int = 12
    max_pillar_tilt_deg: float = 30.0

    # --- bracing ---
    brace_enabled: bool = True
    brace_diameter: float = 0.8
    brace_slenderness: float = 15.0
    brace_min_angle_deg: float = 45.0
    brace_max_span: float = 12.0

    # --- foot ---
    foot_diameter: float = 5.0
    foot_height: float = 0.6
    pad_diameter: float = 2.0
    pad_height: float = 0.4

    # --- placement ---
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
    n_braces: int = 0
    dropped: list[SupportPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
