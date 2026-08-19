# resin_supports_for_fdm_pipeline

Drop in any STL. Get it back rotated to a sensible print pose and fitted with
**resin-style supports that an FDM printer can actually make** — thin pillars,
tiny snap-off contact tips, diagonal cross-braces — ready to slice.

## Why

Pre-supported miniatures come with resin supports: hair-thin pillars ending in a
sub-0.1 mm point. No filament printer can extrude that, so the file is unusable
as shipped. The existing fix, [Resin2FDM](https://painted4combat.gumroad.com/l/Resin2FdmLite),
is a Blender add-on that *thickens supports that already exist* — it cannot help
with a model that came bare.

This does the other half: it generates the supports itself, at FDM dimensions,
on any model.

Models are taken **as posed in the file** — rotate it how you want it printed
before you drop it in, and the pipeline just sets it down on the bed. There is
an experimental auto-orient button, but choosing a print pose is a judgement
call and yours beats a scoring function's.

## The support spec

Resin supports fail on FDM because of their dimensions, not their shape. The
shape is good — a single point of contact scars far less than a scaffold welded
to the surface. So the geometry stays resin-like and the numbers change:

| | |
|---|---|
| Pillar | 1.2–2.0 mm — survives the print, still snaps off |
| Contact tip | 0.3–0.6 mm — as small as the printer can reliably manage |
| Tip cone | narrows *upward*, so every layer is smaller than the one below and the support needs no support of its own |
| Cross-braces | diagonal, ≥45°, so slender pillars hold each other up instead of needing fatter contacts |
| Support layer height | 2× the model's — supports carry no detail, so print them coarse |

Every one of those is derived from your nozzle diameter, not hardcoded.

## Install

Python 3.11+ (developed on 3.14). No Node, no build step.

```bash
pip install -e .
```

## Use

```bash
python -m rsupport.cli serve
```

...then drop an STL on the page.

Or headless:

```bash
python -m rsupport.cli supports mini.stl -o ready.3mf --nozzle 0.2
```

The `.3mf` output carries two objects with per-object slicer settings already
set, so the miniature slices at 0.08 mm while the supports slice at 0.16 mm.
Export a `.stl` instead and you get one welded mesh — slice that with supports
switched **off**.

## Layout

See [CLAUDE.md](CLAUDE.md) for the module map, the git rules, and the
invariants. The full design is in [docs/PLAN.md](docs/PLAN.md).
