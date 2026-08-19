"""Parameter presets.

Every support dimension is derived from the nozzle diameter, so switching
nozzles rescales the whole support system coherently instead of leaving a
0.3 mm tip spec on a printer that cannot extrude below 0.4 mm.

The anchor values come from the Resin2FDM documentation, which is the closest
thing to a published spec for "resin-style supports that survive on FDM":

    pillar    1.2 - 2.0 mm   (start at 1.5 if they snap)
    tip       0.3 - 0.6 mm   (as small as the printer can reliably print)
    braces    diagonal, so isolated pillars hold each other up
    supports sliced at ~2x the model's layer height
"""

from __future__ import annotations

from .types import SupportParams

# Multipliers relating each feature to the nozzle diameter. Chosen so a 0.2 mm
# nozzle lands on the documented defaults (tip 0.3, pillar 1.2).
_TIP_RATIO = 1.5  # tip must be at least ~1.5 extrusion widths to be printable
_PILLAR_RATIO = 6.0
_BRACE_RATIO = 4.0

# Below this the tip stops being a tip and starts being a scar.
_TIP_MIN = 0.25
_TIP_MAX = 0.6
_PILLAR_MIN = 1.0
_PILLAR_MAX = 2.0


def from_nozzle(
    nozzle_diameter: float = 0.2,
    layer_height: float | None = None,
    **overrides,
) -> SupportParams:
    """Build a full parameter set for a given nozzle.

    Args:
        nozzle_diameter: mm. 0.2 and 0.25 are the good ones for miniatures;
            0.4 works but is less forgiving.
        layer_height: mm for the model itself. Defaults to 40% of the nozzle,
            clamped to the 0.08-0.12 range the docs recommend for minis.
        **overrides: any SupportParams field, applied last.
    """
    n = float(nozzle_diameter)
    if n <= 0:
        raise ValueError("nozzle_diameter must be positive")

    if layer_height is None:
        layer_height = min(0.12, max(0.08, round(n * 0.4, 3)))

    tip = _clamp(n * _TIP_RATIO, _TIP_MIN, _TIP_MAX)
    pillar = _clamp(n * _PILLAR_RATIO, _PILLAR_MIN, _PILLAR_MAX)
    brace = _clamp(n * _BRACE_RATIO, n * 2.0, pillar * 0.8)

    params = SupportParams(
        nozzle_diameter=n,
        layer_height=layer_height,
        support_layer_height=round(layer_height * 2, 3),
        tip_diameter=round(tip, 3),
        tip_length=round(max(1.0, pillar), 3),
        tip_penetration=round(max(0.05, layer_height * 1.25), 3),
        pillar_diameter=round(pillar, 3),
        brace_diameter=round(brace, 3),
        foot_diameter=round(max(4.0, pillar * 4.0), 3),
        # A tip cannot bridge, so points must sit closer together than the
        # printer's reliable bridging distance.
        max_unsupported_span=round(max(4.0, pillar * 4.0), 3),
        # --- tree ---
        # Clearance has to clear the printer's own imprecision, so it scales
        # with the nozzle rather than being a fixed gap.
        xy_clearance=round(max(0.3, n * 2.0), 3),
        # Sampling finer than a couple of support layers buys nothing: the
        # slicer cannot follow the model that closely anyway.
        tree_layer_pitch=round(max(0.4, layer_height * 6.0), 3),
        # A trunk carrying a dozen branches needs real section, but past this
        # it stops being removable by hand.
        max_branch_diameter=round(pillar * 2.5, 3),
        # --- resin scaffold ---
        # Resin shafts run roughly 0.6 mm at the top to 1.2 mm at the bottom.
        # The ratio carries over; the absolute sizes come from the nozzle.
        shaft_upper_diameter=round(max(n * 3.0, pillar * 0.55), 3),
        shaft_lower_diameter=round(pillar, 3),
        arm_diameter=round(max(n * 2.5, pillar * 0.5), 3),
    )
    return params.with_(**overrides)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


#: Named presets surfaced in the UI.
PRESETS: dict[str, SupportParams] = {
    "mini_0.2": from_nozzle(0.2),
    "mini_0.25": from_nozzle(0.25),
    "mini_0.4": from_nozzle(0.4),
    # Fatter and further apart: fewer scars to clean, less reliable on fine detail.
    "mini_0.2_sparse": from_nozzle(0.2, support_spacing=4.5, tip_diameter=0.4),
    # Everything held, for models that keep failing.
    "mini_0.2_dense": from_nozzle(0.2, support_spacing=2.0, overhang_angle_deg=55.0),
}

DEFAULT_PRESET = "mini_0.2"


def get(name: str = DEFAULT_PRESET) -> SupportParams:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None
