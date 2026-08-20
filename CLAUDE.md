# resin_supports_for_fdm_pipeline

Drop in any STL → it gets auto-oriented for printing and fitted with **resin-style
supports dimensioned for FDM** (thin pillars, tiny snap-off contact tips, diagonal
cross-braces), then exported ready to slice. Runs as a local web app in the browser.

Full design: [docs/PLAN.md](docs/PLAN.md).

---

## Git rules — READ THIS BEFORE ANY GIT COMMAND

Remote: <https://github.com/FireB1scuit/resin_supports_for_fdm_pipeline>

1. **NEVER push to `main`.** Not `git push origin main`, not a force push, not a
   fast-forward, not "just this once". There is no situation in this project where
   pushing to `main` is correct.
2. **`main` only ever changes by merge.** Work lands on `main` through a pull request
   that gets merged. Nothing else touches it.
3. **All work happens on a branch.** Branch off `main`, name it
   `feat/…`, `fix/…`, `docs/…`, or `chore/…`.
4. **Push regularly.** Commit and push the working branch often — at minimum at the end
   of any response that changed code. No need to ask for confirmation to commit or push
   a *working branch*. Small, frequent commits over one big one.
5. Never use `--no-verify`, and never bypass signing.

Normal loop:

```bash
git checkout -b feat/my-thing && git add -A && git commit -m "..." && git push -u origin feat/my-thing
```

If a push to `main` is ever attempted, stop and open a PR instead.

---

## Project rules

- **Python 3.14**, on Windows. `numpy`, `scipy`, `trimesh`, `shapely`, `fastapi`,
  `uvicorn`. Do **not** add `embreex` or `manifold3d` — 3.14 wheels are unreliable and
  neither is needed (see PLAN.md: rays are all straight down; supports are concatenated,
  not boolean-unioned).
- **No Node.js.** The frontend is plain ES modules with a vendored `three.module.js`.
  Never introduce a build step, a bundler, or a `package.json`.
- **All support dimensions derive from `nozzle_diameter`** in `presets.py`. Never hardcode
  a millimetre value in geometry code.
- **The self-printability invariant**: generated support geometry must not contain an
  overhang steeper than the printable limit. If our supports would need supports, the
  generator is broken. `tests/test_supports.py` enforces it exactly on constructed
  scenes.
  On a real sculpt there is a **measured residual**, guarded by `VIOLATION_BUDGET` in
  `tests/test_pipeline.py`: a few flat pad undersides where a pillar lands on a feature
  narrower than itself. They are short bridges off anchored material, not floating
  islands, and `build_supports` reports them in `SupportBuild.warnings`. The budget is a
  measurement, not a target — if it rises something regressed, and if it falls, tighten
  it. Do not raise it to make a test pass.
- **Models arrive pre-posed.** The input STL is assumed to already be rotated the way it
  should print; the pipeline only sets it down on the bed. Auto-orientation exists in
  `orient.py` but is **opt-in** (`--auto-orient`, or the button in the UI) and must never
  be on the default path.
- Geometry stages stay separate: `points` → `geometry` (with `orient` optional in front).
  The UI edits the point list between stages, so stage 3 must be cheap to re-run alone.
- **The resin scaffold is the default support style** (`resin.py`). It is an SLA support
  system — contact tip, angled arm, thin vertical shaft, join cone, base, cross-links —
  with FDM dimensions. It is deliberately **not** an FDM organic tree: nothing fuses into
  a thickening trunk. If a change starts making shafts merge and grow, it is drifting back
  into `tree.py`'s territory and is wrong for this style.
  `tree.py` (organic branches) and `supports.py` (plain pillars) stay selectable via
  `support_style`, and `supports.build_supports` dispatches. Both reuse the ring/profile
  machinery in `supports.py`, so those imports are deliberately *inside* the functions —
  a top-level one would be circular.
- Resin cross-links are horizontal; ours cannot be, because no FDM printer bridges a
  horizontal strut hanging in air. `resin._link_angle` lays them at the shallowest angle
  inside the printable band. Adapting resin conventions to what a nozzle can do is the
  point of the project, so make the adaptation and say why — do not copy the resin value
  and hope.
- **Two guarantees, both structural rather than checked after the fact.** A support never
  enters the model, and only ever rests on the model when the plate is genuinely
  unreachable. Both come from the bottom-up reachability sweep in `tree.AvoidanceField`,
  which every style shares; do not replace it with a local "is there something below me"
  test, which cannot answer either question. `tests/test_tree.py` pins both.
- A flat downward face is a 90° overhang wherever it is, including buried inside another
  support. That is why merge junctions and tip undersides are built with `cap_bottom=False`
  rather than stacking capped primitives.
- Run `pytest` before pushing.

## Layout

```
src/rsupport/
  types.py     shared dataclasses (SupportParams, SupportPoint, Orientation)
  presets.py   nozzle-derived parameter presets
  mesh_io.py   load / save STL, OBJ, 3MF
  raycast.py   grid-bucketed straight-down raycaster
  detail.py    per-face detail-density metric
  orient.py    candidate orientations + scoring
  overhang.py  overhang faces + per-layer island detection
  sampling.py  support point placement
  supports.py  pillar / tip / brace / foot mesh construction
  export.py    combined STL, separate STLs, two-object 3MF
  cli.py       headless CLI
  web/         FastAPI app + static three.js viewer
```

## Commands

```bash
python -m pytest
python -m rsupport.cli supports samples/mini.stl -o out.stl
python -m rsupport.web
docker compose up -d   # same app, containerised, on :8000
```

`Dockerfile` pins **3.12**, not 3.14, for the same wheel reason as above — every
dependency has a cp312 manylinux wheel, so the image needs no compiler. The
container runs `python -m rsupport.web` as its only process, so keep that
entrypoint working; `compose.yaml`'s `restart: unless-stopped` is what makes the
app come back with Docker.
