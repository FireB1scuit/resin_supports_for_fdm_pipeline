# resin_supports_for_fdm_pipeline

Drop in any STL. Get it back fitted with **resin-style supports that an FDM
printer can actually make** — a scaffold of thin vertical shafts, cross-linked
into a lattice, ending in tiny snap-off contact tips — ready to slice.

## Why

Pre-supported miniatures come with resin supports: hair-thin pillars ending in a
sub-0.1 mm point. No filament printer can extrude that, so the file is unusable
as shipped. The existing fix, [Resin2FDM](https://painted4combat.gumroad.com/l/Resin2FdmLite),
is a Blender add-on that *thickens supports that already exist* — it cannot help
with a model that came bare.

This does the other half: it generates the supports itself, at FDM dimensions,
on any model.

Models are taken **as posed in the file** — rotate it how you want it printed
before you drop it in. There is an experimental auto-orient button, but choosing
a print pose is a judgement call and yours beats a scoring function's.

By default the model is then **lifted 5 mm off the plate** and the scaffold
carries it, which is what a resin printer does. Nothing of the sculpt is printed
against glass: no elephant's foot, no squashed first layer, and its underside is
supported like any other overhang instead of being ignored. Slide the lift to 0
if you would rather set it down flat.

## The support spec

Resin supports fail on FDM because of their dimensions, not their shape. The
shape is good — a single point of contact scars far less than a scaffold welded
to the surface. So the geometry stays resin-like and the numbers change:

| | |
|---|---|
| Shaft | 1.2–2.0 mm — survives the print, still snaps off |
| Contact tip | 0.3–0.6 mm — as small as the printer can reliably manage |
| Tip cone | narrows *upward*, so every layer is smaller than the one below and the support needs no support of its own |
| Arms | several tips fan off one shaft, so there are far fewer feet to snap off than there are contact points |
| Cross-links | diagonal struts tying the shafts into a lattice, so the structure braces itself instead of balancing a stick on each foot |
| Base | a 5 × 2 mm disc under every shaft. Lifted, the model needs a shaft every few mm, so the discs overlap into a raft |
| Support layer height | 2× the model's — supports carry no detail, so print them coarse |

Every one of those is derived from your nozzle diameter, not hardcoded — except
the base, which is sized in plain millimetres because it answers to the build
plate rather than to the nozzle.

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

One trap: Docker records `stop` **and `kill`** as a deliberate stop, and a
deliberately stopped container does not come back on its own — not on a crash,
not when Docker next starts. `docker compose start` (or `up -d`) re-arms it.

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
   you — it uses the pose exactly as authored, and floats it above the plate.
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
| **contact points** | a dot at every point a support touches, in two reds. **Off by default — turn it on to edit supports** |
| **wireframe** | see through the model |

Drag to orbit, scroll to zoom.

The two reds are the whole point of that third toggle:

| | |
|---|---|
| faded red | a contact that got a support |
| solid deep red, slightly larger | a contact **nothing could reach** — that overhang is going to print unheld |

Held points are washed out deliberately: there are hundreds of them and they are
covering the surface you are trying to look at. The unheld ones are usually a
handful, and they are the ones worth doing something about — raise **strut
lean**, drop the **lift**, or delete the point and shift-click a replacement
somewhere with a clearer run down to the plate. The count is in the stats panel
too, but a number in a sidebar does not tell you *where*.

## Editing supports by hand

The automatic placement is a starting point, not a verdict.

- **Delete one** — turn on **contact points**, then click a dot, either colour.
  (Clicking does nothing while that layer is hidden.)
- **Add one** — shift-click anywhere on the model. Hand-placed supports are
  marked mandatory and never get thinned away.

Either edit rebuilds only the geometry, so it comes back in well under a second.

## The controls

| Control | What it does | Reach for it when |
|---|---|---|
| **preset** | swaps the whole parameter set for a nozzle | you change nozzles |
| **lift off plate** | how far the model floats above the bed, 0–20 mm | the underside matters and you want it held rather than squashed (raise); you want the model printed flat on the plate (0) |
| **parenting** | how many tips share one shaft | fewer feet to snap off (raise); shorter, more direct arms (lower) |
| **strut lean** | how far anything may tilt off vertical | a shaft cannot get around an obstacle (raise), or leaning struts are failing to print (lower) |
| **tip ø** | how wide the support is where it touches | supports snap off during the print (raise) or leave visible scars (lower) |
| **shaft ø** | thickness of the vertical strut | shafts snap or wobble (raise) |
| **spacing** | how far apart contact points sit | too many supports to clean up (raise), or an overhang sags (lower) |
| **overhang** | the angle at which a face is judged to need support | steep walls are being supported unnecessarily (lower) or a shallow slope droops (raise) |
| **tip style** | conical or spherical contact | spherical snaps off cleaner but grips less |
| **cross-links** | diagonal struts between slender shafts | turn off only if removal is a nightmare — they exist so tips can stay thin |
| **link ø** | thickness of those struts | the lattice flexes (raise); it is fighting you at cleanup (lower) |
| **max span** | furthest apart two shafts may be and still be linked | outlying shafts stand unbraced (raise); links are stretching across gaps you want left open (lower) |
| **link spacing** | height from one link to the next up the *same* pair of shafts | tall pillars flex between their braces (lower, for more rungs); there is too much to cut off (raise) |
| **start height** | how far up the scaffold the links begin, measured from the plate | the bottom of the lattice is tangled up with the raft, or you want the first cut to be an easy one (raise). 0 starts them inside the feet, where they cost nothing |
| **link angle** | how steeply a link climbs, above horizontal | shallower spans further per millimetre of rise, steeper packs the lattice tighter. Only moves inside the band that prints — 40–50° at the default overhang limit — and is clamped there |
| **headroom** | clear air kept below the shaft tops | you cannot get a blade in under the model (raise). It comes out of the height a link has to work with, so it drops links on short supports first |
| **supports from plate only** | every support must reach the bed; one that cannot is left unheld and reported | leave it on. Unticking it only lets a blocked shaft stop where it is — supports that *start* on the model are not implemented yet |
| **base ø** | width of the disc each shaft stands on | supports peel off the plate mid-print (raise); the raft is welded to the bed and impossible to remove (lower) |
| **base height** | how tall that disc is | the same trade, but height buys grip without eating more bed area |

Changing **tip ø**, **shaft ø**, **tip style**, any of the **cross-link**
controls, **supports from plate only** or either **base**
dimension rebuilds geometry only, which is fast. Changing **spacing** or
**overhang** re-decides where supports go, which takes a moment longer. Changing
**lift** moves the model, so it redoes both. Sliders act on release, not on drag.

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
different filament from the shafts (PETG tips under PLA supports) and they
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
| `--tip`, `--shaft`, `--spacing`, `--overhang` | override one value |
| `--lift MM` | how far the model floats above the plate; `--lift 0` sets it down flat |
| `--base-width MM`, `--base-height MM` | the adhesion disc under each shaft |
| `--tip-style {conical,spherical}` | |
| `--lean DEG` | how far a strut may lean off vertical |
| `--parenting 0..1` | how many tips share a shaft |
| `--no-braces` | drop the cross-links |
| `--link-thickness MM` | cross-link diameter |
| `--link-span MM` | furthest apart two shafts may be and still be linked |
| `--link-spacing MM` | height from one link to the next up the same pair |
| `--link-start MM` | height above the plate below which no link is laid |
| `--link-angle DEG` | how steeply a link climbs; clamped into the printable band |
| `--link-headroom MM` | clear air left below the shaft tops |
| `--allow-model-landings` | let a blocked shaft stop on the model instead of being refused |
| `--auto-orient` | re-pose first (off by default) |

`python -m rsupport.cli serve` starts the same web app.

## Presets

| Name | Nozzle | Model layer | Tip ø | Shaft ø | Spacing | Overhang |
|---|---|---|---|---|---|---|
| `mini_0.2` *(default)* | 0.2 | 0.08 | 0.30 | 1.2 | 3.0 | 45° |
| `mini_0.25` | 0.25 | 0.10 | 0.375 | 1.5 | 3.0 | 45° |
| `mini_0.4` | 0.4 | 0.12 | 0.60 | 2.0 | 3.0 | 45° |
| `mini_0.2_sparse` | 0.2 | 0.08 | 0.40 | 1.2 | 4.5 | 45° |
| `mini_0.2_dense` | 0.2 | 0.08 | 0.30 | 1.2 | 2.0 | 55° |
| `esun_pla_02_A1m` | 0.2 | 0.08 | 0.30 | 1.2 | 3.0 | 45° |

## When something goes wrong

| Symptom | Try |
|---|---|
| Supports snap off mid-print | raise **shaft ø** toward 1.5 mm; keep cross-links on; slow the outer wall to 35–40 mm/s |
| Ugly scars after removal | lower **tip ø** toward 0.3 mm; set support interface layers to 1 |
| Supports won't come off | lower **shaft ø**, switch tip style to spherical, or turn cross-links off |
| Far too many supports | raise **spacing**, or lower **overhang** so fewer faces qualify; try `mini_0.2_sparse` |
| Too many separate feet to snap off | raise **parenting** — more tips will share a shaft |
| An overhang sagged | lower **spacing**, raise **overhang**; or shift-click extra supports exactly where it drooped |
| A thin part printed in mid-air | that region had no support — shift-click to add one; report it, islands are meant to be caught automatically |
| Log says *"N contact point(s) left unsupported"* | nothing could be routed to the plate from there. Raise **strut lean** so a shaft can travel further sideways per layer, lower the model's **lift** so there is more room beside it, or shift-click a support somewhere with a clearer run down |
| Log says *"N contact point(s) could not be reached by a tip"* | the contact sits in a pocket every approach angle runs into — usually a dimple on an upward-facing surface that barely qualified as an overhang. Normally safe to ignore; raise **overhang** so it stops qualifying |
| Log says *"N shaft(s) stand on the model"* | only with **supports from plate only** unticked. There was no way down to the plate; those shafts end in a tip, so they still snap off |

## Known limits

- **Never yet tested against a real miniature.** All development used a
  synthetic model. Support density is the first thing likely to need tuning.
- A handful of support undersides overhang where a support lands on a feature
  narrower than itself — short bridges off anchored material, not floating
  islands. Reported in the log and budgeted in `tests/test_pipeline.py`.
- **Supports cannot start on the model.** A shaft routes around obstacles to
  reach the plate, and where no route exists the contact is refused rather than
  propped off the sculpt. A support rooted *on* the model — the thing a resin
  slicer does under a cloak or between two arms — is not built. On the sample
  mini that leaves a few percent of contacts unheld; they are reported in the
  log, and unticking **supports from plate only** currently buys only a
  last-resort landing where the shaft stops, not a real branch off the model.
- A shaft that has to route around something comes down beside it rather than
  under its contact, so its base disc can end up smaller than **base ø** asks
  for — it is shrunk to whatever fits without fusing into the wall it passed.
- Auto-orientation rarely picks a tilted pose for a model with a flat base.
- Detail detection cannot tell a sculpted face from a machined edge, so the
  "keep supports off the detail" scoring suits organic models, not brackets.

---

# The maths

Two questions get decided by arithmetic: **where should a support touch the
model** (stage 2, `overhang.py` + `sampling.py`), and **what shape can be built
under that point without the support itself needing support** (stage 3,
`avoidance.py` + `supports.py` + `resin.py`). This section is what those modules
are actually computing.

Only two conventions are needed throughout:

- **Angle below horizontal.** A face's angle below horizontal is `asin(-n_z)`
  for unit normal `n`: `0°` for a vertical wall, `90°` for a flat downward face,
  negative for anything facing up. This is the quantity `printable_overhang_deg`
  caps.
- **Severity.** `s = n · (-Z) = -n_z`, clipped to `0..1`. The cosine of the angle
  between the face normal and straight down: `0` for a wall, `1` for a face
  pointing straight down.

---

## Where contacts go

### 1. Which faces are overhanging

One dot product per face. A face is flagged when

```
s  >=  cos(overhang_angle)
```

At the default 45° that is `s >= 0.707` — the familiar slicer rule. Note the
sign convention here is the opposite of a slicer's slider: a **larger** angle
flags **more** faces, because it widens the cone of normals counted as
"pointing downward".

Faces sitting on the build plate are removed first — a face is on the plate if
its *highest* vertex is within two layer heights of **z = 0**. Without that, the
underside of anything standing on a flat base is the most heavily supported
surface on the model, and it is already touching the bed.

That test is against the plate, not against the model's own lowest point, and
the difference is the whole lift feature. Lift the model and its underside is no
longer within two layers of anything — it is a 90° overhang hanging in air, and
it gets held like one, across its entire footprint.

### 2. Islands: cross-sections that start in mid-air

An overhang angle cannot see a sword tip that begins 20 mm above the plate with
nothing under it — that surface is not *steep*, it is *absent*. So the model is
sliced into layers and each polygon is asked one question:

```
does this polygon intersect ANY polygon in the layer below?
```

If not, its first printed layer lands on air. Polygons under `island_min_area`
(0.05 mm²) are discarded as slivers from a plane grazing a curved surface. The
lookup against the layer below goes through an R-tree, so it is `O(k log n)`
rather than every-polygon-against-every-polygon.

Slicing dominates this stage, so there is a cost trick. Slicing at the true
layer height on a 41 mm mini is 512 planes (~1.5 s); slicing at
`4 × layer_height` is 128 planes (~0.4 s). The coarse pass runs, and then each
island found is **bisected** back down to full precision: the bracket `[known
empty, detected]` halves on every probe, so

```
error after k probes = 4·h / 2^k     →     k = 2 probes to get back to h
```

Two extra single-plane slices per island, and islands are rare. The one thing
the coarse step genuinely gives up: a feature shorter in Z than 0.32 mm can fall
between two planes — but a 0.3 mm tip could not hold it anyway.

### 3. Blue-noise coverage of the overhang

Candidates are drawn over the flagged faces with probability proportional to
face area, so density is uniform per mm² rather than per triangle. How many:

```
N = 20 · (overhang area) / (tightest spacing)²        clamped to [512, 60000]
```

i.e. about 20 candidates for every point that will survive — enough that the
thinning pass has real choices.

Each candidate then gets its **own** minimum spacing, tightened as the overhang
steepens. With `t` the severity rescaled so the overhang threshold is 0 and
straight-down is 1:

```
t = clamp( (s − cos θ) / (1 − cos θ), 0, 1 )
r = spacing · (1 − 0.5·t)
```

A face right on the threshold keeps the full 3 mm; a flat underside gets 1.5 mm.
Halving the spacing means up to four times as many points per mm², which is what
a flat underside needs — it sags far worse than a 50° slope.

Thinning is **greedy Poisson-disk elimination**: build one KD-tree over all
candidates, walk them in priority order, and each time one is accepted, kill
every candidate inside its radius. One tree, no rebuilds. Priority is

```
priority = severity − 0.5 · detail
```

where `detail` is the normalised per-face detail-density metric. That term only
ever decides **ordering** — where two candidates compete for the same spot, the
one on the plain cloak beats the one on the sculpted face. It can never remove a
support that is needed.

Island contacts are fed in as pre-committed seeds: candidates within `r` of one
are killed before the loop starts, so nothing crowds a mandatory point.

### 4. Span fill: nothing left further than a tip can bridge

A 0.3 mm tip cannot bridge, so after blue noise a **farthest-point insertion**
runs. Keep an array `d` of each candidate's distance to the nearest existing
support, then repeat:

```
i = argmax d
stop if d[i] <= max_unsupported_span
accept i,  then  d = min(d, ‖candidates − candidates[i]‖)
```

Updating `d` in place is `O(N)` per insertion with no tree rebuild, and the loop
terminates with a guaranteed covering radius: nothing on the sampled surface is
further than `max_unsupported_span` from a support. This pass runs against a
*relaxed* mask (`1.25 × overhang_angle`), because a shallow face the strict mask
skipped can still droop over a long unsupported run.

### 5. Rejecting points nothing can be built under

Two vetoes, both applied to candidates before they reach the thinning pass:

- **Buried.** Probe one layer height below the surface and run a parity test —
  count the surface crossings in the column *above* the probe; odd means inside
  the mesh. Crossings at the same height are merged first, or a column running
  exactly along a shared triangle edge counts one surface twice and flips the
  answer.
- **No room.** `clearance = z_point − z_of_highest_surface_below`, and the point
  is dropped unless `clearance > tip_length`. Stage 3 could not fit a contact
  cone in there.

### The raycaster underneath all of this

Every ray in this project points straight down, which is a much smaller problem
than general raycasting. Triangles are projected to XY once and bucketed into a
uniform grid; a query tests only the triangles in its own cell, using barycentric
coordinates *in the projection*:

```
hit  ⟺  u >= 0,  v >= 0,  u + v <= 1
z    =  a_z + u·(c_z − a_z) + v·(b_z − a_z)
```

An exactly vertical triangle projects to a line segment, its barycentric
denominator collapses to zero, and it is dropped — correctly, since a vertical
wall contributes no crossing to a vertical column.

---

## What gets built under them

### The one inequality everything obeys

Every part of a support is a surface of revolution: horizontal circles of radius
`r(z)` whose centres drift sideways at slope `m` (mm horizontal per mm of height
— a leaning strut). At azimuth `u` on such a surface the normal's vertical
component is proportional to `−(m·u + r'(z))` while its horizontal magnitude is
constant, so the steepest overhang anywhere on it is

```
angle_below_horizontal = atan( |m| + max(r', 0) )   ≤   printable_overhang_deg
```

That single line decides nearly every design choice in stage 3:

- **A profile that narrows going up (`r' ≤ 0`) is always self-supporting**,
  whatever else is happening. Hence the contact tip is wide at the bottom and
  thin at the top, the base only ever narrows on its way up, and the "spherical"
  tip is a dome rather than a ball — a free-floating sphere's underside is a 90°
  overhang.
- **Leaning by `t` costs exactly `t` degrees** of the budget (`m = tan t`), so
  every lean in the project is clamped to `strut_lean`, itself clamped to
  `printable_overhang_deg − 2`.
- **A flat downward face is a 90° overhang, always** — including one buried deep
  inside another support. So a shaft is lofted as one continuous stack of rings
  (base → flare → shaft), not as three capped primitives stacked face to face,
  and every buried cap is suppressed rather than left touching its neighbour.

Cross-links are the exception that proves it. A plain strut laid at angle `a`
above horizontal has **side walls** overhanging by `90 − a` and **end caps**
overhanging by `a`, so both are printable only inside the band

```
90 − printable  ≤  a  ≤  printable        →   40° ≤ a ≤ 50° at the defaults
```

which is why a link is placed at a *chosen* angle — pick the two attachment
heights so the angle comes out right — rather than by connecting two convenient
points. The shallowest angle in the band wins (42°, two degrees of margin):
every degree shallower is more horizontal span for the same vertical run, and
vertical run is the scarce thing on a short support.

### Reachability: one sweep, every guarantee

"Can a strut here get down to the plate?" is a question about the entire column
of layers below it, so no local "is something directly beneath me" test can
answer it. It is precomputed bottom-up instead, per layer and per radius:

```
free[r][i]   = bed  −  dilate( model_cross_section_i,  r + xy_clearance )
reach[r][0]  = free[r][0]
reach[r][i]  = free[r][i]  ∩  dilate( reach[r][i−1],  max_move )
max_move     = collision_pitch · tan(strut_lean)
```

`dilate` is a Minkowski sum with a disc (shapely's `buffer`) — "grow this region
outward by that much". `max_move` is simply how far a strut may travel sideways
in one layer without exceeding its lean budget.

Read the recursion as induction and `reach[r][i]` is *exactly* the set of XY
positions on layer `i` from which a strut of radius `r` has a legal descent all
the way to the plate: it is legal here, and one layer's travel takes it somewhere
that was already provably legal below. Everything else falls out:

- a position inside `reach` on the layer below drops straight down;
- a position outside it moves to the nearest point of `reach` — which is what
  makes a shaft step around an arm rather than stop dead at it;
- a shaft lands on the model only when `reach` is genuinely empty beneath it, not
  as a preference;
- and since every position is inside `free` for its own radius, no support can
  intersect the model at all.

All of that is structural, not a check-and-reject afterwards. The descent that
reads this out is `resin._route_to_plate`, and the induction above is exactly
why it needs no search and cannot dead-end: every position it is standing on has
a legal successor one layer down, by construction.

What the sweep does *not* cover is anything not routed through it — arms, tips
and cross-links are placed by geometry, so each is tested separately by
`resin._strut_clear` (a parity test against the model, plus the same
`xy_clearance`). A shaft was never the only thing that could cross a sculpt.

Buffering every layer at every distinct radius would be ruinous, so collision is
sampled at **6 radii spaced geometrically** from `tip_diameter/2` to
`max_strut_diameter/2`, and a lookup rounds **up** — a strut judged against a
radius thinner than its own could be routed into the model. This is CuraEngine's
tree-support avoidance, descended from Vanek et al., *Clever Support* (2014).

### From a contact point to a shaft

**The tip axis.** A resin tip leaves the surface along its normal, so it meets
the model at roughly a right angle and snaps off leaving a dot. But a tip is a
small strut like any other and may not out-lean the printable limit:

```
axis = −n · tip_length                 (from the tip's base up to the contact)
if axis_z ≤ 0:                         come straight up instead
if ‖axis_xy‖ > axis_z · tan λ:         scale axis_xy down until equality holds
```

The last line keeps the *direction in plan* and steepens the climb — the tip
still approaches from the right side, it just does so at a printable angle.

**The elbow** is `contact − axis`: the joint where the tip's own run ends. Every
measurement below is taken from there, never from the contact. The tip has
already covered that horizontal step, and charging the arm a second vertical rise
for the same distance is what used to push the shaft top below the build plate
for contacts near the bed.

**Arms and parenting.** An arm may lean at most `min(arm_angle, λ)` off vertical,
so covering `d` horizontally costs

```
rise(d) = d / tan( min(arm_angle_deg, λ) )
```

A shaft feeding several arms must sit below all of them:

```
shaft_top(xy) = min over members of  ( elbow_z − rise(‖elbow_xy − xy‖) )
```

Clustering is therefore bounded by geometry, not taste. Contacts within
`tip_length + parenting · support_spacing` of each other are candidates for one
shaft, highest first, and a candidate only joins if the resulting `shaft_top`
stays above zero — otherwise the shaft would have to start below the plate.
Raising **parenting** widens that radius; it cannot buy a physically impossible
arm.

The shaft stands at the **mean of its members' elbow XYs**, then is *settled*: if
that point is not standable, it moves to the nearest point of the reachable
region, and is accepted only if the move is within `4 × max_move`. Further than
that and the group is abandoned and each contact retried on its own — one awkward
neighbour should not cost the rest their supports.

Afterwards, shafts closer together than `1.5 × shaft_lower_diameter` are folded
into one. Two shafts half a millimetre apart are one shaft with extra steps:
double the feet to snap off, and no distance for a cross-link to span. An arm is
handed to the survivor only if it can still meet its contact at a printable angle
from over there.

**Descent.** Scan layers downward from the shaft top until `free` fails. If it
never does, the shaft lands at `z = 0`. If it does, take the last clear height,
ask the raycaster for the highest surface strictly below it, and land there — on
the model, which is why that case ends in a tip too and gets reported in the log.

### Cross-links

A lattice that holds the model up can still be a mess to look at and worse to
cut off, so *which* shaft braces which, and *at what height*, are decided for the
whole field at once rather than shaft by shaft.

**Which pairs.** Nearest-first bracing is the obvious rule and a bad one: it picks
the same popular shaft from every side of a crowd, leaves the shaft on the far
edge of it with nothing, and links two shafts straight over the top of a third
standing between them. Instead:

1. **Delaunay triangulation** of the shaft positions in plan. This is the graph
   of shafts that are genuinely adjacent, it is planar — so no two links cross —
   and it does not depend on the order the shafts arrived in.
2. **Span filter**: longer than half the shaft and link diameters (below that
   they are one column already), no longer than `brace_max_span`.
3. **Gabriel filter**: drop a link if another shaft stands inside the circle
   that has the link as its diameter. That shaft is nearer to both ends than
   they are to each other, so the link is reaching over its head; two short
   links through it brace the same pair better. This is also what trims
   Delaunay's long thin border triangles.
4. **Spend the cap globally**, shortest link first: a pair is taken while both
   its shafts are under `_LINKS_PER_SHAFT` = 3 *neighbours* — neighbours, not
   struts, so a tall pair with a four-rung ladder still counts once.
5. **Reconnect**: a cap can cut a corner of the field adrift, so a second pass
   over the runners-up puts back the shortest link across each remaining split.

On an evenly spaced field that is the grid you would have drawn by hand.

**At what height.** Each pair has a window of heights it can hold a link in, at
link angle `a`:

```
rise   = span · tan a
window = [ max(land_z of both) + foot_height/2 ,  min(top_z of both) ]
```

and the pair is linked only if the window is at least `rise` tall — a short
support on a low overhang has no vertical run to spend and gets no link.

Six things are adjustable here, and they all trade against that window.
`brace_max_span` caps `span`, and so caps `rise`. `brace_angle_deg` sets `a`,
clamped into the band that prints (`90 − printable_overhang_deg` to
`printable_overhang_deg`, so 40–50° by default); left unset it takes the
shallowest angle there is, because that is the most span per millimetre of a
scarce quantity. `brace_diameter` is the strut thickness. `brace_interval` is the height from one
rung to the next up the same pair — see below. `brace_start_height` lifts the
floor of every window, measured from the plate rather than from wherever the
shaft happens to stand, because "how far up do the links begin" is a question
about the scaffold and not about one shaft. And `brace_headroom`
takes the top off the window: a shaft's top is where its arms leave for their
contacts, so a link that goes all the way up arrives in the middle of the arm
fan, directly under the model — the busiest part of the scaffold and the worst
place to have to reach with a blade.

The upper end of a link answers to the **shorter** of the two shafts. Letting it
climb to the top of the one it is ascending sends a link from a stub to a tower
straight on up the tower, past the stub's own arms, with nothing under its far
end — worth six millimetres on the sample mini, and invisible on any test scene
where the shafts are all much of a height.

Headroom is deliberately 0 by default and deliberately not derived from the
nozzle, which is the one place the usual rule does not hold. It is spent out of
the window, and the window is set by how *tall* the shafts are, which is a
property of the model. A coarser nozzle makes shafts fatter, not taller, so
scaling headroom with it takes the same millimetres out of a shorter window:
measured on the mini, one link diameter of headroom costs nothing at a 0.2
nozzle and a third of the lattice at 0.4. How much room you want for a cutter is
a judgement, so it is left to whoever is holding the cutters.

Letting each pair start its own ladder from its own base is correct and looks
like noise: on the sample mini it put 153 links at 20 distinct heights. A fixed
grid at `brace_interval` is not the fix either — the windows are narrow, and a
grid walks straight past most of them. So the heights come *from* the windows:
cover them all with the fewest distinct heights, which is a stabbing problem with
an exact greedy answer (sort windows by their top; whenever one is still
uncovered, put a storey at its top). Each storey then slides to the middle of the
windows it covers. The mini's 153 links become 83, on **2** storeys, with no
shaft left unbraced.

That cover is minimal by construction — two heights held the whole sample mini —
which is right for reaching everything and wrong for holding it. A pair of 40 mm
pillars tied twice near the plate is a pair of stilts. So the set also carries a
plain ladder at `brace_interval`, and a pair hangs a rung on every storey inside
its window: one near the plate for a short support, all the way up for a tall one.
A ladder rung landing within half an interval of a cover storey is dropped rather
than doubled, and the floor under `brace_interval` is physical — two links closer
than their own combined thickness are one lump with a hole in it, not two links.

**The ladder is anchored at the bottom of the structure and climbs**, the way
scaffolding is built. Hanging it off the cover instead is the obvious thing and is
wrong: the cover is a stabbing of the windows, so where the shafts are all much of
a height — which is exactly what a large `lift_height` produces, every one of them
standing from the plate to the model's underside — the windows are near enough
identical, the cover collapses to a single storey in the *middle* of them, and a
ladder counted from there leaves the whole lower half bare. At a 20 mm lift on the
Templar that put the lowest link 14.8 mm off the plate with the windows open from
1.5 mm, and the foot of a pillar is the last place to leave unbraced. Anchored at
the bottom it is 1.5 mm.

Because the ladder is one grid for the whole field, the extra rungs line up across
it exactly as the cover storeys do; stacking links does not undo the arrangement.
Every rung leans the same way — uphill toward increasing x — so a storey reads as
a row rather than a scribble.

`SupportBuild.n_braces`, and the **N links** in the UI, count *struts*. A pair
tied at four heights is four links to look at and four to cut.

**Serving the UI.** The static assets go out with `Cache-Control: no-cache`. The
sidebar and the script that drives it are two files, this is a tool people leave
open across restarts, and a browser left to its own freshness guess may pair a new
`index.html` with a cached `app.js` — which puts controls on screen with nothing
listening to them. They move, they show their value, and nothing happens, which
looks exactly like a bug in the generator. `no-cache` still allows the 304, so it
costs a conditional request per file per load. For the same reason `/api/supports`
reports any override key the running server does not recognise, instead of
letting `SupportParams.with_` drop it in silence.

**Exceptions, in order.** A chosen pair can turn out unbuildable, because the
model is in the way of every rung between them; and a tall shaft can be adjacent
only to stubs, none of which is tall enough to hold a printable diagonal — which
is the case that most needs bracing. A shaft left with nothing therefore gets the
runners-up, and then, failing that, may reach past its own neighbours to anything
within `brace_max_span` tall enough to hold it. Only a shaft with nothing at all
is allowed either, so tidiness gives way exactly where it costs a brace.

**Having links is not the same as being held.** A link finishes at the height of
the *shorter* of the two shafts it ties, so a tall shaft in a thicket of stubs is
braced to the top of the stubs and free above — the half that flexes, and the
half nothing else is holding. On the Templar the worst of them stood 21 mm tall
with its highest link at 10. The graph will never fix that on its own, because
that shaft has three perfectly good links and is not destitute, so a final pass
asks a different question of every shaft: *is anything holding the part that
needs holding?* If some shaft within `brace_max_span` could hold a link a rung's
worth higher than the topmost one it already has, it is worth crossing the field
for. Where nobody can do better, nothing happens. The same measurement is what
the pass is judged on — worst bare top run on a shaft over 10 mm, 10.9 mm before
and 3.7 mm after, against a floor of `headroom + rise/2`.

Two rungs are allowed off the grid, one at each end, and for the same reason: a
pair's floor and ceiling are its own numbers and the storeys are the field's, so
the highest and lowest storeys a pair can reach may each sit a whole rung short of
what it could actually hold. Those shortfalls land on the two parts of a pillar
nothing else is holding — the free top, and the foot that carries every bending
moment above it — so where either gap is worth a rung, one goes at the boundary. A
course following the roofline, or the plate, reads as deliberate in a way that a
bare end does not.

Each candidate link is tested against the model by sampling 10 interior points
along it; both ends are left uncapped, since they are buried inside the shafts and
a cap there is a 90° overhang in the middle of solid plastic.

### Profiles that can never need support

| Part | Profile `(z, r)` | Why it is safe |
|---|---|---|
| Base | `(z0, r_foot) → (z0+h−f, r_foot) → (z0+h, r_shaft)` | a straight-walled disc (`r' = 0`), then a flare in to the shaft (`r' < 0`). The disc is what grips the plate; a bare cone is at full width for one layer only |
| Shaft | `(bottom, r_lower) → (top, r_upper)` | slight upward taper, `r' < 0` |
| Conical tip | `(base_r) → (contact, r_tip) → (contact+pen, r_tip)` | narrows upward, then sunk `tip_penetration` into the model |
| Spherical tip | `(z_eq + R·sin φ, R·cos φ)` for `φ = 30°, 55°, 75°, 85°` | only the **upper** hemisphere is emitted; the lower half is replaced by the taper running up to the equator |
| Strut / arm / link | constant `r`, centre drifting at `m = tan(tilt)` | sheared, not rotated, so end caps stay horizontal — a rotated cap would overhang by `90 − tilt` |

### Where the numbers come from

Nothing in the geometry code contains a millimetre value. Every dimension is
derived from the nozzle:

```
tip             = clamp(1.5 · nozzle, 0.25, 0.6)
shaft           = clamp(6.0 · nozzle, 1.0, 2.0)
link            = clamp(4.0 · nozzle, 2 · nozzle, 0.8 · shaft)
tip_length      = max(1.0, shaft)
tip_penetration = max(0.05, 1.25 · layer_height)
max span        = max(4.0, 4 · shaft)
xy_clearance    = max(0.3, 2 · nozzle)
collision pitch = max(0.4, 6 · layer_height)
support layer   = 2 · layer_height
```

The 1.5 on the tip is "at least about one and a half extrusion widths, or the
printer cannot lay it down at all"; the 6.0 on the shaft is what puts a 0.2 mm
nozzle on the 1.2 mm shaft the Resin2FDM documentation recommends. Change the
nozzle and the whole support system rescales coherently.

The base — 5 mm wide, 2 mm tall — is the one dimension deliberately left out of
that. It is not a feature the nozzle has to draw, it is a footprint on a piece of
glass, and a plate needs the same square millimetres of contact whether a 0.2 or
a 0.4 nozzle is filling them in. The lift is absolute for the same reason: 5 mm
of air is 5 mm of air.

---

# Development

See [CLAUDE.md](CLAUDE.md) for the module map, the git rules and the invariants.
The full design is in [docs/PLAN.md](docs/PLAN.md).

```bash
python -m pytest                              # 142 tests
python scripts/make_sample.py samples/synthetic_mini.stl
```
