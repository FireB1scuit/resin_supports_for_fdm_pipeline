# Handoff — resin_supports_for_fdm_pipeline

Written 2026-08-29 at the end of a long session. Read `CLAUDE.md` first — it is the
authority on every project rule and this file does not repeat it.

---

## Where things stand

Five GitHub issues were worked in parallel this session. **Four are merged and closed.**
One is open and mid-review.

| Issue | What it was | State |
|---|---|---|
| #20 | contact point algorithm missed small overhangs | ✅ merged (PR #25) |
| #22 | Lychee-style overhang region highlighting | ✅ merged (PR #26) |
| #24 | export/download rework | ✅ merged (PR #27) |
| #28 | Pages served the README instead of the app | ✅ merged (PR #29), **verified live** |
| #23 | variable/longer strut lengths + smarter reach angle | ✅ merged (PR #30) |
| **#21** | **multi-model support + rotation gizmo** | **🔄 open — needs user review** |

The default branch is **`Master`**. Note that `CLAUDE.md`'s git-rules section talks about
`main`, which has never existed; that section is stale in that respect. `Master` is what
everything merges into and what Pages deploys from.

---

## The one open thread: issue #21

Branch: **`feat/multi-model-rotation-wheels`**, tip `ffdc1a3`, pushed, **0 commits behind
`Master`** so it will merge clean.

Compare/merge:
<https://github.com/FireB1scuit/resin_supports_for_fdm_pipeline/compare/Master...feat/multi-model-rotation-wheels?expand=1>

### What it contains

Delivered across three rounds of user feedback:

1. **Multi-model support.** One `ModelEntry` per upload in `web/static/app.js`
   (`state.models`, `state.order`, `state.activeSid`). Click a model in the viewport, or
   its chip, to make it active. Each model keeps its own meshes, points, settings,
   rotation, lift and last build. Server side there is a `Workspace` in `web/core.py`
   mirrored across `app.py` and `browser.py`.
   Deliberately **not** implemented: building/exporting several models concurrently as one
   operation was scoped out early; export still writes the **active model only**.
2. **In-viewport rotation gizmo**, replacing the sidebar rotation sliders/wheels. Three
   rings on the model, dragged to rotate, committing through the existing
   `POST /api/rotate` on release. Two mutually exclusive top-left chips: `rotation mode`
   and `movement mode`. Movement mode drags the model in XY only (Z belongs to the
   `lift_height` slider) and is purely client-side — XY never reaches the server.
3. **One press builds every model** (`generate` → `runBatch` → `runOne`), sequentially,
   each from its own settings and its own staleness.
4. **View chips are global** — model, supports, contact points, wireframe, overhang all
   toggle every model at once, while the active model still renders solid and the others
   dim. That solid/dim distinction is about *selection*; the chips are about *layers*.
   Keep them separate — the user explicitly asked for the highlighting to stay.
5. **The camera is never moved out from under the user.** Selecting a model does not
   reframe, and neither does a rotation. Only a model arriving on an empty plate is
   framed.

### Two things a future session must know

**The gizmo ring colours are a deliberate, user-requested exemption from a documented
rule.** `CLAUDE.md` and the comment at the top of `index.html`'s `<style>` say each hue in
the viewport means exactly one thing, and red means "nothing could support this". The
rings are nonetheless X=red / Y=green / Z=blue, because that axis convention is the one
every CAD/slicer user reads without being told. The exemption and its reasoning are
recorded in both `index.html` (`--c-gizmo-*`) and `CLAUDE.md`. **Do not "fix" it back to
the old violet/lime/indigo** — an earlier agent chose those for exactly the rule-abiding
reason, and the user overrode it on purpose.

**The visible rings are thin; an invisible fatter proxy is what the pointer hits**
(`gizmoRings` vs `gizmoPicks` in `app.js`). Thinning the visible tube alone would have
made the gizmo fiddly to grab. If you restyle the rings, keep the pick proxies.

### What is NOT verified, and should be before merging

The last agent hit an account rate limit **mid-browser-testing** and never finished. I
recovered its uncommitted work, and verified:

- ✅ `node --check` on `app.js` — syntax OK (node used as a system tool only; the project
  still has no Node dependency, no bundler, no `package.json`, per `CLAUDE.md`)
- ✅ full `pytest` — exit 0
- ✅ full `RSUPPORT_AVOIDANCE=raster pytest` — exit 0
- ✅ the dev server serves the new markup (`rotation mode`, `movement mode`,
  `--c-gizmo-x: #ff0000`)

**Not verified: the actual interactive behaviour in a browser.** Nothing has driven two
models through a generate-all press, confirmed each used its own settings rather than the
active model's dials, confirmed the chips toggle both models, or dragged the thinned rings.
Those are exactly the things pytest cannot see. **Exercise this at
<http://127.0.0.1:8005> (or a fresh `python -m rsupport.web`) before merging #21.**

The highest-risk spot is per-model settings in `runOne`/`settingsFor`: the sidebar dials
only ever show the *active* model's values, so a bug there would silently apply the active
model's settings to every model. That is the thing to check first with two differently
configured models.

---

## Environment notes that will save you time

- **`gh` CLI is not installed.** No `gh pr create`, no `gh issue list`. Read issues via
  `WebFetch` on the GitHub URL; open PRs by handing the user a `compare/...?expand=1` link.
- **Python is one shared global environment** with `rsupport` installed editable from the
  main checkout. Several agents ran concurrently this session, so each got its own
  `.venv` inside its worktree. **Never `pip install -e .` against the global environment
  while other agents may be running** — it repoints everyone's editable install.
- **`pyproject.toml` sets `-q` in `addopts`.** A passing run therefore prints *no*
  "N passed" line. Check the exit code; do not infer success from absent output. This
  wasted time once already.
- The raster suite can exceed a 120s Bash timeout. Give it room.
- Review servers were run per branch on ports 8001-8005 out of the agent worktrees under
  `.claude/worktrees/`. They do not survive a session restart; restart with
  `<worktree>/.venv/Scripts/python.exe -m rsupport.web --port <n> --no-browser`.
- Several agent worktrees still exist under `.claude/worktrees/`. `git worktree list`
  shows them; prune what you do not need. There is also `E:/caude_work/rsfp-issue-28`.

### The recurring merge conflict

Every branch this session conflicted on **`todo notes.md` and the `#todolist` `<ul>` in
`index.html`**, because each branch removes the bullet it finished. The resolution is
almost always "both sides' removals are correct — drop both bullets". After resolving,
verify the two lists match one-for-one, in order, verbatim (`CLAUDE.md` requires this;
it is user-facing copy on the live site).

---

## Pages / deployment

Issue #28's root cause is worth remembering: the default branch was renamed
`feat/pipeline-foundation` → `Master`, but `.github/workflows/pages.yml` still triggered
on the old name, so **the deploy workflow silently never fired** on any merge, and
GitHub's own branch-source Jekyll build of `README.md` had the site to itself. Fixed by
pointing the trigger at `Master`.

Confirmed working: <https://fireb1scuit.github.io/resin_supports_for_fdm_pipeline/> now
serves the app. If it ever reverts to the README, check
**Settings → Pages → Build and deployment → Source = "GitHub Actions"** — CI is not
permitted to set that itself, and `pages.yml` only warns when refused.

---

## Suggested next steps

1. Have the user exercise #21 in a browser (see the unverified list above), then merge.
2. Remaining `todo notes.md` bullets are the natural next issues — big-overhang-area
   support, supports not on the printbed only, watermark, traffic tracking, print profiles.
3. Open questions deliberately left alone, worth raising rather than silently doing:
   - **Export is still active-model only.** Now that one press builds every model,
     "export all" is a plausible want, but it is a real decision (one file? several? how
     named?) and the user has not asked for it.
   - **There may now be no way to recentre the camera** on a model you have lost track of,
     since auto-framing was removed from selection and rotation. If that turns out to be a
     gap, it needs a small fit-view affordance — ask before building one.
