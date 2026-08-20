# resin_supports_for_fdm_pipeline

Drop in any STL → it gets auto-oriented for printing and fitted with **resin-style
supports dimensioned for FDM** (thin shafts, tiny snap-off contact tips, diagonal
cross-links), then exported ready to slice. Runs as a local web app in the browser.

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
  a millimetre value in geometry code. The one deliberate exception is the **base**
  (`foot_diameter`, `foot_height`): it answers to the build plate, not to the nozzle, and a
  plate needs the same square millimetres of contact whatever is extruding onto it. It is
  left at its default in `from_nozzle` on purpose — do not "fix" that by deriving it.
- **The self-printability invariant**: generated support geometry must not contain an
  overhang steeper than the printable limit. If our supports would need supports, the
  generator is broken. `tests/test_supports.py` enforces it exactly on constructed
  scenes.
  On a real sculpt there is a **measured residual**, guarded by `VIOLATION_BUDGET` in
  `tests/test_pipeline.py`: a few flat undersides where a support lands on a feature
  narrower than itself. They are short bridges off anchored material, not floating
  islands, and `build_supports` reports them in `SupportBuild.warnings`. The budget is a
  measurement, not a target — if it rises something regressed, and if it falls, tighten
  it. Do not raise it to make a test pass.
- **Models arrive pre-posed.** The input STL is assumed to already be rotated the way it
  should print; the pipeline only sets it down on the bed. Auto-orientation exists in
  `orient.py` and is reachable from the CLI (`--auto-orient`) and from `POST /api/orient`,
  but is **opt-in** and must never be on the default path. The UI has no button for it —
  only manual rotation sliders (`POST /api/rotate`) and the "file's pose" reset.
- **The plate is z=0, and by default the model is not on it.** `lift_height` (default
  5 mm, slider 0-20) floats the whole model and the scaffold carries it, exactly as a
  resin print does. Consequences, all of them load-bearing:
  - Anything asking "where is the floor" means **z=0**, never `mesh.bounds[0][2]`. That is
    `sampling.PLATE_Z`. Reading the floor off the mesh is what makes a lifted model's
    underside look supported when it is hanging in mid-air.
  - A lifted model's flat bottom is a 90 degree overhang like any other and gets contact
    points across its whole footprint. On a grounded model it is printed against glass and
    gets none. Both behaviours are pinned in `tests/test_sampling.py`.
  - The lift is a property of the *model*, not of the supports, so stage 1 applies it
    (`mesh_io.drop_to_bed(..., lift=)` / `mesh_io.set_lift`). The web session keeps the
    posed mesh at z=0 as `Session.grounded` and floats a copy, so moving the slider is a
    translation rather than another orientation search — but it does move every contact
    point, so stage 2 has to re-run with it.
- Geometry stages stay separate: `points` → `geometry` (with `orient` optional in front).
  The UI edits the point list between stages, so stage 3 must be cheap to re-run alone.
- **The resin scaffold is the only support structure** (`resin.py`). It is an SLA support
  system — contact tip, angled arm, thin vertical shaft, base, cross-links —
  with FDM dimensions. There is no style selector: `supports.build_supports` builds this
  and nothing else. Do not add one back; the earlier organic-tree and plain-pillar
  generators were deliberately deleted rather than left switchable.
  It is deliberately **not** an FDM organic tree: nothing fuses into a thickening trunk.
  If a change starts making shafts merge and grow, it is drifting into tree territory and
  is wrong for this project. `tests/test_resin.py::test_a_shaft_never_thickens_into_a_trunk`
  is the guard.
  `resin.py` lofts its geometry from the ring/profile primitives in `supports.py`, so
  `supports.build_supports` imports `resin` *inside the function* — a top-level import
  would be circular.
- Resin cross-links are horizontal; ours cannot be, because no FDM printer bridges a
  horizontal strut hanging in air. `resin._link_angle` lays them at the shallowest angle
  inside the printable band. Adapting resin conventions to what a nozzle can do is the
  point of the project, so make the adaptation and say why — do not copy the resin value
  and hope.
- **Nothing the generator builds may enter the model. This is a guarantee, not a
  preference**, and it covers every strut, not just the shafts:
  - A **shaft** is routed, not dropped. `resin._route_to_plate` walks the reachability
    maps in `avoidance.AvoidanceField` down a layer at a time, sliding sideways to the
    nearest still-reachable position wherever the column below is blocked. Because
    `reach[i] = free[i] ∩ reach[i-1].buffer(max_move)`, a position inside `reach` always
    has a successor within one layer's travel — so the descent needs no search, no
    backtracking, and cannot dead-end. Something directly below a contact is a reason to
    lean, never a reason to stop. Do not replace the sweep with a local "is there
    something below me" test, which cannot answer the question at all.
  - An **arm, tip or cross-link** is placed by geometry rather than routed, so it is
    asked: `resin._strut_clear` parity-tests it against the model and holds it
    `xy_clearance` clear. This is not optional decoration. The generator once let
    `_absorb_neighbours` hand a stubby shaft the arms of a contact 25 mm above it, and
    what got built was a 26 mm near-vertical spear through the middle of the sculpt —
    invisible to every shaft-only check in the suite.
  - A **base disc** is several times fatter than its shaft, and a routed shaft comes down
    hard against the clearance boundary of whatever it stepped around. `resin._foot_radius`
    shrinks the disc to fit rather than fusing it into a wall. Overlap at z=0 is still
    fine and deliberate — that is the raft.
  `tests/test_avoidance.py` pins the sweep; `tests/test_resin.py` pins the guarantees it
  buys, and samples shafts along `xy_at` rather than assuming one XY.
- **`plate_only` is on by default: the build plate is the only landing.** A contact with
  no collision-free route down is left unheld and reported in `SupportBuild.warnings`,
  rather than propped off the sculpt. That trade is deliberate and measured both ways in
  `tests/test_pipeline.py`: `DROP_BUDGET` (routing allowed to land on the model, ~1%)
  against `PLATE_ONLY_DROP_BUDGET` (the shipped default, ~8% on the sample mini — dimples
  on the upper surface of the head, where every route in crosses the head itself).
  Turning it off enables only the crude last-resort landing in `resin._drop_shaft`.
  Supports that *start* on the model — a shaft rooted on the sculpt and branching from
  there — are **not implemented**; the UI checkbox says so. Do not quietly implement them
  behind the flag.
- The base is a **disc, then a flare**, not a plain cone. A cone is at its full width for
  exactly one layer, so what grips the glass is a ring of extrusion; the straight-walled
  disc is the part that sticks, and with a lifted model the discs of neighbouring supports
  overlap into what amounts to a raft. `supports.foot_profile` owns this and
  `tests/test_resin.py::test_the_bases_of_a_lifted_model_overlap_into_a_raft` pins it.
- A flat downward face is a 90° overhang wherever it is, including buried inside another
  support. That is why arm joins and tip undersides are built with `cap_bottom=False`
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
  avoidance.py bottom-up reachability sweep: where a support may stand
  supports.py  ring/profile primitives + build_supports, the stage-3 entry point
  resin.py     the scaffold itself — shafts, arms, tips, cross-links
  export.py    combined STL, separate STLs, two-object 3MF
  cli.py       headless CLI
  web/         FastAPI app + static three.js viewer
```

## Commands

```bash
python -m pytest
python -m rsupport.cli supports samples/mini.stl -o out.stl
python -m rsupport.cli supports samples/mini.stl -o out.stl --lift 0   # set it down instead
python -m rsupport.web
docker compose up -d   # same app, containerised, on :8000
```

`Dockerfile` pins **3.12**, not 3.14, for the same wheel reason as above — every
dependency has a cp312 manylinux wheel, so the image needs no compiler. The
container runs `python -m rsupport.web` as its only process, so keep that
entrypoint working; `compose.yaml`'s `restart: unless-stopped` is what makes the
app come back with Docker.
