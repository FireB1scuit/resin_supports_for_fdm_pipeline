"""Headless entry point.

The browser app is the point of the project, but every stage is reachable from
here too — which is what makes the pipeline testable, scriptable over a folder
of models, and debuggable without a UI in the way.

    rsupport info      mini.stl
    rsupport orient    mini.stl -o oriented.stl
    rsupport supports  mini.stl -o ready.3mf --nozzle 0.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import export as export_mod
from . import mesh_io, presets
from .types import SupportParams


def _add_param_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("support parameters", "override the preset")
    g.add_argument("--preset", default=presets.DEFAULT_PRESET, choices=sorted(presets.PRESETS))
    g.add_argument("--nozzle", type=float, help="nozzle diameter in mm; rederives every dimension")
    g.add_argument("--layer-height", type=float, dest="layer_height")
    g.add_argument("--tip", type=float, dest="tip_diameter", help="contact tip diameter (mm)")
    g.add_argument("--shaft", type=float, dest="shaft_lower_diameter", help="shaft diameter (mm)")
    g.add_argument("--spacing", type=float, dest="support_spacing", help="support spacing (mm)")
    g.add_argument("--overhang", type=float, dest="overhang_angle_deg", help="overhang angle (deg)")
    g.add_argument("--tip-style", dest="tip_style", choices=["conical", "spherical"])
    g.add_argument(
        "--lift",
        type=float,
        dest="lift_height",
        help="how far the model floats above the plate (mm); 0 sets it down flat",
    )
    g.add_argument(
        "--base-width", type=float, dest="foot_diameter", help="adhesion base diameter (mm)"
    )
    g.add_argument(
        "--base-height", type=float, dest="foot_height", help="adhesion base height (mm)"
    )
    g.add_argument(
        "--lean",
        type=float,
        dest="strut_lean_deg",
        help="how far a strut may lean off vertical (deg)",
    )
    g.add_argument(
        "--parenting",
        type=float,
        help="0 = one shaft per contact, 1 = many tips share a shaft",
    )
    g.add_argument("--no-braces", action="store_true", help="disable diagonal cross-links")
    g.add_argument(
        "--allow-model-landings",
        action="store_true",
        help="let a shaft stand on the model where the plate cannot be routed to "
        "(partial: the shaft stops where it is rather than starting from the model)",
    )


def _params_from_args(args) -> SupportParams:
    if args.nozzle is not None:
        params = presets.from_nozzle(args.nozzle)
    else:
        params = presets.get(args.preset)

    overrides = {
        k: getattr(args, k, None)
        for k in (
            "layer_height",
            "tip_diameter",
            "shaft_lower_diameter",
            "support_spacing",
            "overhang_angle_deg",
            "tip_style",
            "strut_lean_deg",
            "parenting",
            "lift_height",
            "foot_diameter",
            "foot_height",
        )
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if getattr(args, "no_braces", False):
        overrides["brace_enabled"] = False
    if getattr(args, "allow_model_landings", False):
        overrides["plate_only"] = False
    return params.with_(**overrides)


def cmd_info(args) -> int:
    mesh = mesh_io.load(args.input)
    info = mesh_io.summary(mesh)
    print(f"{Path(args.input).name}")
    print(f"  faces      {info['faces']:,}")
    print(f"  vertices   {info['vertices']:,}")
    print("  size       {:.1f} x {:.1f} x {:.1f} mm".format(*info["size"]))
    print(f"  watertight {info['watertight']}")
    return 0


def cmd_orient(args) -> int:
    from . import orient as orient_mod

    params = _params_from_args(args)
    mesh = mesh_io.load(args.input)

    t0 = time.perf_counter()
    results = orient_mod.orientations(mesh, params, top_k=3)
    dt = time.perf_counter() - t0

    print(f"evaluated in {dt:.2f}s — top {len(results)}:")
    for i, o in enumerate(results):
        terms = "  ".join(f"{k}={v:.3f}" for k, v in sorted(o.terms.items()))
        print(f"  [{i}] {o.label:<14} score={o.score:.4f}   {terms}")

    chosen = results[args.pick]
    out = orient_mod.apply(mesh, chosen)
    if args.output:
        mesh_io.save(out, args.output)
        print(f"wrote {args.output}")
    return 0


def cmd_supports(args) -> int:
    from . import sampling, supports

    params = _params_from_args(args)
    mesh = mesh_io.load(args.input)
    print(f"loaded {len(mesh.faces):,} faces")

    # The model arrives already posed the way it should print. All we do is set
    # it on the bed. Auto-orientation is available behind --auto-orient, but it
    # is a taste judgement and the file's own pose is the better default.
    if args.auto_orient:
        from . import orient as orient_mod

        t0 = time.perf_counter()
        results = orient_mod.orientations(mesh, params, top_k=3)
        chosen = results[min(args.pick, len(results) - 1)]
        mesh = orient_mod.apply(mesh, chosen)
        print(f"oriented '{chosen.label}' (score {chosen.score:.4f}) in {time.perf_counter()-t0:.2f}s")
        mesh = mesh_io.set_lift(mesh, params.lift_height)
    else:
        mesh = mesh_io.drop_to_bed(mesh, lift=params.lift_height)

    if params.lift_height > 0:
        print(f"floating the model {params.lift_height:g} mm above the plate")

    t0 = time.perf_counter()
    points = sampling.place_points(mesh, params)
    n_forced = sum(1 for p in points if p.forced)
    print(f"placed {len(points)} support points ({n_forced} forced) in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    build = supports.build_supports(mesh, points, params)
    print(
        f"built {len(build.mesh.faces):,} support faces, {build.n_braces} links, "
        f"{len(build.dropped)} dropped in {time.perf_counter()-t0:.2f}s"
    )
    for w in build.warnings[:10]:
        print(f"  ! {w}")

    written = export_mod.export(mesh, build.mesh, args.output, params, mode=args.mode)
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_serve(args) -> int:
    from .web import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsupport",
        description="Auto-orient an STL and give it resin-style supports that print on FDM.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="print mesh facts")
    pi.add_argument("input")
    pi.set_defaults(func=cmd_info)

    po = sub.add_parser("orient", help="find and apply the best print orientation")
    po.add_argument("input")
    po.add_argument("-o", "--output")
    po.add_argument("--pick", type=int, default=0, help="use the Nth best orientation")
    _add_param_args(po)
    po.set_defaults(func=cmd_orient)

    ps = sub.add_parser("supports", help="place supports on a pre-posed model and export")
    ps.add_argument("input")
    ps.add_argument("-o", "--output", required=True)
    ps.add_argument("--mode", default="auto", choices=["auto", "combined", "separate", "3mf"])
    ps.add_argument(
        "--auto-orient",
        action="store_true",
        help="re-pose the model first (off by default: the file's own pose is used)",
    )
    ps.add_argument("--pick", type=int, default=0, help="with --auto-orient, use the Nth candidate")
    _add_param_args(ps)
    ps.set_defaults(func=cmd_supports)

    pw = sub.add_parser("serve", help="run the browser app")
    pw.add_argument("--host", default="127.0.0.1")
    pw.add_argument("--port", type=int, default=8000)
    pw.add_argument("--no-browser", action="store_true")
    pw.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
