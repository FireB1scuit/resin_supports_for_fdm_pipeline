# Resin-Style FDM Supports — Browser App

## Context

Pre-supported miniature STLs ship with resin supports: hair-thin pillars ending in a sub-0.1 mm point. An FDM printer cannot extrude that, so those files are unusable on filament without rework. The known fix is Resin2FDM, a Blender add-on that *thickens existing* resin supports; it needs Blender, a paid tier for the good features, and only works on models that already came pre-supported.

We are building the thing that doesn't exist: drop **any** STL into a browser page, get it automatically rotated to a good print orientation and automatically fitted with resin-style supports already dimensioned for FDM — thin pillars, tiny snap-off contact points, diagonal cross-braces — then download it ready to slice.

Target: miniatures, 0.2 mm nozzle, runs locally on Windows (Python 3.14, no Node.js).

## Geometry spec (from the Resin2FDM docs — this is what "resin-style but FDM" means)

| Feature | Spec | Reason |
|---|---|---|
| Pillar diameter | 1.2–2.0 mm (default 1.2 at 0.2 nozzle) | wide enough not to snap, narrow enough to remove |
| Tip contact diameter | 0.3–0.6 mm (default 0.3) | "as small as your printer can reliably print"; minimal scarring |
| Tip cone | 1.2 mm → 0.3 mm over ~1.5 mm | narrows going *up*, so every layer is smaller than the one below → self-supporting |
| Cross-braces | diagonal, ≥45°, Ø0.8 mm | lets isolated tall pillars brace each other instead of needing fat contacts |
| Foot | Ø5 mm cone, 0.6 mm tall | bed adhesion for a 1.2 mm pillar |
| Model layer height | 0.08 mm | fine detail |
| Support layer height | 0.16 mm (2× model) | supports print fast and coarse; needs 3MF export to set per-object |
| Support walls / infill | 2 loops, min sparse infill threshold 0 | from the docs |

Everything above is a parameter derived from `nozzle_diameter`, not a hardcoded constant.

## Architecture

Python does all geometry; the browser is a viewer and control panel. One command starts a local server and opens the page. No Node, no build step — `three.js` is vendored as a plain ES module and served statically.

```
E:\caude_work\resin2fdm\
  pyproject.toml
  src/rsupport/
    mesh_io.py      STL/OBJ/3MF load + save
    raycast.py      fast straight-down raycaster (see note below)
    detail.py       per-face "detail density" metric (curvature x area)
    orient.py       candidate orientations + scoring
    overhang.py     overhang faces + per-layer island detection
    sampling.py     support point placement (Poisson-disk + forced island points)
    supports.py     pillar / tip / brace / foot mesh construction
    presets.py      parameter sets, all derived from nozzle diameter
    cli.py          headless CLI (same engine, no browser)
    web/app.py      FastAPI
    web/static/     index.html, app.js, three.module.js
  tests/  samples/
```

Deps: `numpy` + `scipy` already installed on 3.14. Add `trimesh` (pure Python), `shapely`, `fastapi`, `uvicorn`. **Do not depend on `embreex` or `manifold3d`** — wheels for 3.14 are unreliable, and neither is needed:

- **Raycasting**: every ray we cast points straight down. So instead of a general BVH, project all triangles to XY once, bucket them into a uniform grid, and for a query point look up its cell, run a vectorized point-in-triangle test, and solve the plane equation for z. Milliseconds for thousands of points, no native dependency.
- **Booleans**: not needed. Supports are concatenated with the model, not unioned — which is what every slicer expects anyway.

## Stage 1 — Mesh core + auto-orientation

The hard, interesting part. Ship this as CLI-only first so it can be iterated without UI.

**Candidate orientations**

1. Every convex-hull facet normal, clustered to ~2° and area-weighted (the standard candidate set — any resting pose puts a hull facet on the bed).
2. The largest planar patch in the mesh — an integral round mini base. If a mini has one, a human would set it flat; give this candidate a bonus so it wins ties.
3. PCA principal axis vertical (upright).
4. Lean variants: principal axis tilted 0–35° in 5° steps, to move overhangs onto the back.

**Scoring** (miniature preset weights; the same function with different weights becomes the "functional part" preset later)

- `support_cost` — Σ over overhang faces of area × severity × height above bed.
- `detail_penalty` — support contacts landing on high-detail regions. `detail.py` gives each face a curvature-density score; a mini's face, hands and front crest score high, a cloak back scores low. This is the term that makes the app put supports on the back.
- `stability` — bed-contact footprint area, plus a hard penalty if the centre of mass projects outside the contact hull.
- `height` — total Z, mild weight.
- `island_count` — number of per-layer islands (each is a forced support).

Pick the argmin. Return the top 3 so the UI can offer alternates.

**Verify**: run on 5–10 sample minis; the chosen pose should be recognisably "how you'd print it", with the detail-bearing front tilted up.

## Stage 2 — Where supports go

- **Overhang faces**: `n · (−Z)` beyond the overhang angle (default 45°).
- **Islands**: slice with `trimesh.intersections.mesh_multiplane` at the layer height; for each layer, any cross-section polygon that does not overlap (shapely) a polygon in the layer below is an island — a floating cape edge, an outstretched sword tip. These get a **forced** support point at the polygon's centroid, non-negotiable.
- **Distribution**: Poisson-disk sampling over the overhang region, spacing default 3 mm, tightened where the overhang is steeper. Plus a max-unsupported-span rule (~5 mm) so nothing is left to bridge.
- Output is a plain list of `(point, normal, forced)`. The UI edits this list and Stage 3 turns it into geometry — keeping the two stages separate is what makes interactive add/remove cheap.

## Stage 3 — Support geometry

Per point: cast down (Stage 1 raycaster) to find whether the pillar lands on the bed or on the model below.

1. **Tip** — cone, `tip_d` at the model widening downward to pillar width, sunk `0.1 mm` into the model so it isn't a floating first layer. Conical or spherical tip style, like the reference tool.
2. **Pillar** — cylinder, `sections=12` (low-poly is invisible at 1.2 mm and keeps the STL small).
3. **Collision** — if the pillar's path intersects the model, first try tilting it up to 30°, then try landing it on the model surface, then drop the point and warn.
4. **Bracing** — KD-tree over pillar positions. Any pillar whose free height exceeds ~15× its diameter gets a diagonal Ø0.8 mm strut to its nearest neighbour at ≥45°; if it has no neighbour in reach, an angled prop to the bed.
5. **Foot** — Ø5 mm cone on bed-landing pillars; a small pad on model-landing ones.

**Self-printability assertion** (a real test, run in CI): every triangle of the generated support geometry must have an overhang angle within printable limits. If our own supports need supports, the generator is wrong.

## Stage 4 — The browser app

`python -m rsupport.web` → serves on localhost, opens the browser.

- Drag-drop STL onto the page. Uploads, then shows the mesh in three.js on a bed grid.
- Auto-runs orient → sample → generate, streaming progress (server-sent events; each stage takes seconds, not minutes).
- Renders model grey, pillars orange, tips red. Toggle supports, toggle wireframe, toggle "show what's still unsupported" (overhang faces heat-mapped).
- Right panel: nozzle, tip Ø, pillar Ø, spacing, overhang angle, lean limit, brace on/off. Changing a *geometry* param re-runs Stage 3 only (fast). Changing a *placement* param re-runs Stage 2+3.
- Alternate orientations offered as thumbnails; clicking one re-runs from Stage 2.
- Click a support to delete it; shift-click the model to add one.

**API**: `POST /api/model` → id · `POST /api/orient/{id}` · `POST /api/points/{id}` · `POST /api/supports/{id}` · `GET /api/export/{id}?mode=` · `GET /api/preview/{id}` (glTF for the viewer — smaller and faster over the wire than STL).

## Stage 5 — Export

- **Combined STL** — model + supports, drop into any slicer with supports off.
- **Separate STLs** — model and supports as two files.
- **3MF with two objects** — the one that matters: lets the slicer give the support object its own 0.16 mm layer height and 2 walls while the mini stays at 0.08 mm, exactly as the reference docs recommend. 3MF is a zip of XML; write a minimal exporter rather than hunting for library support.
- Optional "level cube" helper so the slicer sees the bed plane.

## Verification

1. `pytest` — round-trip mesh IO; raycaster checked against brute force on a random mesh; support-geometry overhang assertion; islands detected on a synthetic floating-tip model; orientation of a synthetic "mini" (detailed front, plain back) leans the front up.
2. `rsupport supports samples/mini.stl -o out.stl` — headless, must produce a mesh a slicer opens without complaint.
3. Load `out.stl` in your slicer at 0.08 mm, supports off — inspect the preview: no floating layers, tips are single-extrusion dots, pillars solid.
4. Print a small test plate: one mini plus a torture piece (an outstretched thin arm) and check the tips snap off without scarring. This is the only test that really counts.

## Risks

- **Orientation scoring is a taste problem, not a correctness problem.** Weights will need tuning against real models. Mitigation: expose the top-3 alternates in the UI from day one so a bad pick is one click to fix, and keep weights in `presets.py`.
- **0.3 mm tips on a 0.2 mm nozzle are at the edge of reliability.** The docs warn they may snap. Mitigation: tip size is a live slider, and the brace system exists precisely so tips don't have to be fat.
- **Python 3.14 is new** — if `trimesh`/`shapely` misbehave, both are replaceable (shapely's role is small; trimesh's mesh ops we largely reimplement anyway).
