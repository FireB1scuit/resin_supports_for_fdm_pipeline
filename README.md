# resin_supports_for_fdm_pipeline

Drop in any STL. Get it back fitted with **resin-style supports that an FDM
printer can actually make** — a tree of thin branches that merge on the way down,
ending in tiny snap-off contact tips — ready to slice.

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
| Branches | merge on the way down into limbs and trunks, so the structure carries load through its junctions instead of balancing a stick on each foot |
| Support layer height | 2× the model's — supports carry no detail, so print them coarse |

Every one of those is derived from your nozzle diameter, not hardcoded.

---

# Using it

## Install

Python 3.11+ (developed on 3.14). No Node, no build step.

```bash
pip install -e .
```

## Start the app

```bash
python -m rsupport.web
```

It serves on `http://127.0.0.1:8000` and opens your browser. Nothing leaves your
machine; there is no database and no account. Closing the window throws the
session away.

`--port 9000` to move it, `--no-browser` if you'd rather open the tab yourself.

## Or run it in Docker

If you'd rather not install Python and the geometry stack at all:

```bash
docker compose up -d
```

Then open `http://localhost:8000`. The app *is* the container's only process, so
it is running for exactly as long as the container is — starting the container
starts the app, and there is nothing else to launch.

`restart: unless-stopped` in `compose.yaml` means it also comes back on its own
after a crash and after a Docker or machine restart, so it is up again as soon
as Docker is. It stays down only if you stopped it deliberately:

```bash
docker compose stop
```

Same privacy story as running it locally: sessions are in memory, exports go to
a tempdir, and `tmpfs` wipes both when the container stops. Nothing is written
to a volume, so nothing survives a restart — by design.

To publish it somewhere other than port 8000, change the left-hand number in
`compose.yaml`'s `ports:`. Note the container binds `0.0.0.0` internally, which
is how the published port reaches it — if you map it to an interface other than
localhost, anyone who can reach that address can use the app.

## The workflow

1. **Pose your model first.** Rotate it in whatever tool you like so it stands
   the way you want it printed, and export. This tool does not rotate it for
   you — it sets it down on the bed exactly as authored.
2. **Drop the STL onto the page** (or click to browse). `.stl`, `.obj`, `.ply`,
   `.3mf` and `.off` all load.
3. It runs immediately — finds the overhangs and islands, places contact points,
   builds the supports. A few hundred milliseconds for a typical mini. The log
   at the bottom right tells you what it did and flags anything odd.
4. **Look at it, adjust, download.**

## Reading the viewport

Buttons top-left toggle each layer:

| | |
|---|---|
| **model** | the miniature, grey |
| **supports** | the generated geometry, orange |
| **contact points** | a red dot at every point a support touches. **Off by default — turn it on to edit supports** |
| **wireframe** | see through the model |

Drag to orbit, scroll to zoom.

## Editing supports by hand

The automatic placement is a starting point, not a verdict.

- **Delete one** — turn on **contact points**, then click a red dot. (Clicking
  does nothing while that layer is hidden.)
- **Add one** — shift-click anywhere on the model. Hand-placed supports are
  marked mandatory and never get thinned away.

Either edit rebuilds only the geometry, so it comes back in well under a second.

## The controls

| Control | What it does | Reach for it when |
|---|---|---|
| **preset** | swaps the whole parameter set for a nozzle | you change nozzles |
| **style** | tree (branches that merge and route around the model) or pillars (one straight column per contact) | trees are the default; pillars are the simpler, older behaviour |
| **branch lean** | how far a branch may tilt off vertical | branches cannot get around an obstacle (raise), or thin branches are failing (lower) |
| **merging** | how eagerly branches seek each other out | fewer, thicker trunks and less to clean up (raise); easier removal, more independent branches (lower) |
| **tip ø** | how wide the support is where it touches | supports snap off during the print (raise) or leave visible scars (lower) |
| **pillar ø** | thickness of the column | pillars snap or wobble (raise) |
| **spacing** | how far apart contact points sit | too many supports to clean up (raise), or an overhang sags (lower) |
| **overhang** | the angle at which a face is judged to need support | steep walls are being supported unnecessarily (lower) or a shallow slope droops (raise) |
| **tip style** | conical or spherical contact | spherical snaps off cleaner but grips less |
| **cross-braces** | diagonal struts between slender pillars | turn off only if removal is a nightmare — they exist so tips can stay thin |

Changing **tip ø**, **pillar ø**, **tip style** or **braces** rebuilds geometry
only, which is fast. Changing **spacing** or **overhang** re-decides where
supports go, which takes a moment longer. Sliders act on release, not on drag.

## Orientation (optional)

The panel shows *"The model is used exactly as posed in the file"* — that is the
default and usually what you want.

- **try auto-orient** scores candidate poses and applies the best, then offers
  the runners-up as buttons. It tries to keep supports off the detailed side.
- **file's pose** puts it back.

Treat this as a second opinion. It has a known weakness: a model with a flat
base almost always wins on stability, so the tilted poses that would move
supports onto a figure's back rarely get chosen.

## Downloading

| Button | You get | Slice it with |
|---|---|---|
| **3MF** | one file, two objects, per-object settings already applied | supports **off** — the layer heights are already set |
| **STL** | model and supports welded into one mesh | supports **off** |
| **both** | a zip of two separate STLs | your own arrangement |

**3MF is the one to use.** It carries the miniature and the supports as separate
objects, with the miniature set to 0.08 mm layers and the supports to 0.16 mm
with 2 walls and no infill. Supports carry no detail, so printing them coarse
costs nothing and saves a lot of time. PrusaSlicer, OrcaSlicer and Bambu Studio
read those per-object settings; anything else opens two plain objects and you
set it yourself.

Whatever you export, **turn your slicer's own support generation off.** These
supports are already in the mesh.

## Slicer settings that matter

Beyond what the 3MF sets for you:

| Setting | Value |
|---|---|
| Nozzle | 0.2 mm sharpest · 0.25 mm works well · 0.4 mm possible but unforgiving |
| Model layer height | 0.08–0.12 mm |
| Support layer height | ~2× the model's |
| Support walls | 2 |
| Minimum sparse infill threshold | 0 |
| Print speed | ~50 mm/s; drop outer wall to 35–40 mm/s if tips snap |
| Support interface layers | 1 (not the default 2–3) — better surface |

A trick worth knowing if you have a multi-material setup: print the tips in a
different filament from the pillars (PETG tips under PLA supports) and they
release almost by themselves.

## The command line

Everything the app does is scriptable, which is what makes batch runs possible.

```bash
# what am I looking at?
python -m rsupport.cli info mini.stl

# supports on a single model
python -m rsupport.cli supports mini.stl -o ready.3mf --nozzle 0.2

# a whole folder
for f in supported/*.stl; do
  python -m rsupport.cli supports "$f" -o "out/$(basename "${f%.stl}").3mf"
done
```

Useful flags for `supports`:

| Flag | |
|---|---|
| `-o FILE` | output; `.3mf` gives the two-object file, `.stl` the welded one |
| `--mode {auto,combined,separate,3mf}` | override what the extension implies |
| `--preset NAME` | see the table below |
| `--nozzle MM` | re-derives every dimension from scratch |
| `--tip`, `--pillar`, `--spacing`, `--overhang` | override one value |
| `--tip-style {conical,spherical}` | |
| `--style {tree,pillar}` | structure; tree is the default |
| `--branch-angle DEG` | how far a branch may lean (tree only) |
| `--merge 0..1` | how hard branches merge (tree only) |
| `--no-braces` | drop the cross-braces |
| `--auto-orient` | re-pose first (off by default) |

`python -m rsupport.cli serve` starts the same web app.

## Presets

| Name | Nozzle | Model layer | Tip ø | Pillar ø | Spacing | Overhang |
|---|---|---|---|---|---|---|
| `mini_0.2` *(default)* | 0.2 | 0.08 | 0.30 | 1.2 | 3.0 | 45° |
| `mini_0.25` | 0.25 | 0.10 | 0.375 | 1.5 | 3.0 | 45° |
| `mini_0.4` | 0.4 | 0.12 | 0.60 | 2.0 | 3.0 | 45° |
| `mini_0.2_sparse` | 0.2 | 0.08 | 0.40 | 1.2 | 4.5 | 45° |
| `mini_0.2_dense` | 0.2 | 0.08 | 0.30 | 1.2 | 2.0 | 55° |

## When something goes wrong

| Symptom | Try |
|---|---|
| Supports snap off mid-print | raise **pillar ø** toward 1.5 mm; keep braces on; slow the outer wall to 35–40 mm/s |
| Ugly scars after removal | lower **tip ø** toward 0.3 mm; set support interface layers to 1 |
| Supports won't come off | lower **pillar ø**, switch tip style to spherical, or turn braces off |
| Far too many supports | raise **spacing**, or lower **overhang** so fewer faces qualify; try `mini_0.2_sparse` |
| Too many separate feet to snap off | raise **merging** — branches will collapse into fewer trunks |
| A branch is fused to the model instead of the plate | it had no way down; that is reported in the log. Raise **branch lean** so it can reach further sideways |
| An overhang sagged | lower **spacing**, raise **overhang**; or shift-click extra supports exactly where it drooped |
| A thin part printed in mid-air | that region had no support — shift-click to add one; report it, islands are meant to be caught automatically |
| Log says *"N pillar(s) leaned to clear the model"* | normal — those pillars tilted to avoid passing through the sculpt |
| Log says *"land on a feature too small to rest a pad on"* | also normal, and cosmetic: the base of those few supports bridges a short gap |

## Known limits

- **Never yet tested against a real miniature.** All development used a
  synthetic model. Support density is the first thing likely to need tuning.
- A handful of support pad undersides overhang where a *pillar* lands on a feature
  narrower than itself — short bridges off anchored material, not floating
  islands. Reported in the log and budgeted in `tests/test_pipeline.py`. Trees do
  not have this problem.
- Trees produce roughly twice the triangle count of pillars for the same model,
  and take about three times as long to build (still well under two seconds).
- Auto-orientation rarely picks a tilted pose for a model with a flat base.
- Detail detection cannot tell a sculpted face from a machined edge, so the
  "keep supports off the detail" scoring suits organic models, not brackets.

## Development

See [CLAUDE.md](CLAUDE.md) for the module map, the git rules and the invariants.
The full design is in [docs/PLAN.md](docs/PLAN.md).

```bash
python -m pytest                              # 137 tests
python scripts/make_sample.py samples/synthetic_mini.stl
```
