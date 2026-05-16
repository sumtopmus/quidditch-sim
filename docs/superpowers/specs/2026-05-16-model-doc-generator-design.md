# Model Doc Generator — Design

**Date:** 2026-05-16
**Status:** Spec — ready for implementation plan
**Scope:** Auto-generate a `MODEL.md` per-model human-readable spec sheet that lives alongside the run, the vendored model, and the W&B artifact.

## Goal

Every promoted model — and every training run that produced one — gets a single Markdown file that answers "what is this model" without forcing the reader to open `.hydra/config.yaml`, walk reward-term `_target_`s, or grep through code. The file is auto-generated, so it stays in sync with what actually ran. One file, three surfaces:

1. **On disk in the run dir** (`runs/<name>/<ts>/MODEL.md`) — written at train end, available before promotion.
2. **On disk in the vendored model** (`models/<name>/MODEL.md`) — copied by `scripts/promote.py` alongside `best_model.zip` + `.hydra/`.
3. **Inside the W&B artifact** (file in the artifact tree) — uploaded by `scripts/_artifact_io.py:log_run_artifact`, visible in the wandb model registry UI.

Same content; one generator; one source of truth. Reads frozen-at-train-time `.hydra/{config,meta}.yaml` so the doc reflects what ran, never the current YAML state. Regeneration against the run-dir state is available on demand for stale runs (and for backfilling the 7 legacy promoted models that pre-date this feature).

## Non-goals

- **Not a replacement for `brain/models.md`.** That umbrella file stays narrative/curated: catalog overview, status flags (active/superseded/queued), lineage relationships across multiple models, opponent specs glossary, promotion workflow notes. `MODEL.md` is mechanical/factual for one model only. They complement.
- **Not a replacement for `.hydra/config.yaml`.** The YAML stays the machine-readable source. `MODEL.md` is a dereferenced, prose-and-table view for humans. Anyone wanting the raw bytes reads the YAML.
- **No training-loop changes.** Doc generation is best-effort: a generator failure does not fail the training run. The doc is nice-to-have.
- **No W&B-side metadata changes.** Artifact metadata (`obs_spec`, `n_stack`, …) is unchanged; the new MODEL.md is an additional file inside the artifact, not a metadata edit.
- **No "compare two MODEL.md files" tooling.** That belongs to W&B's run-comparator + `make lineage` output. Not in scope.

## Audience

Brainstormed and selected: **both** future-self/collaborator browsing a vendored model on disk AND the W&B-side artifact viewer. Same content serves both. Content depth tuned to "Balanced" scope (chosen from {Lean, Balanced, Comprehensive}): obs spec, reward stack, env config, hyperparams, lineage, eval results, W&B pin.

## Content Layout

Sections in render order. Total ~150-200 lines.

### 1. Header

```markdown
# MODEL: <run_name>_<timestamp>

**Status:** promoted  ·  **Trained:** <YYYY-MM-DD HH:MM ZONE>  ·  **Git:** `<git_hash>`
**W&B:** `wandb://<run_name>:prod` (v<N>, aliases: <list>)
```

- `Status` derived from whether `_wandb_metadata.json` is present:
  - present → render `**Status:** promoted` AND the `**W&B:**` line below it
  - absent → render `**Status:** run-only` AND omit the entire `**W&B:**` line
- `Trained` derived from the run dir's timestamp (`runs/<name>/<ts>/`).
- Git hash from `.hydra/meta.yaml:git_hash`.

### 2. Summary

A one-paragraph narrative. Two sources, in priority order:

- **Override:** if `cfg.description` (new optional string field on the top-level `Config` schema) is non-empty, use it verbatim.
- **Auto-template:** otherwise, render from cfg fields:

  > `{learner_id}` learner trained from `{init.mode}` (parent: `{init.parent}`) against `{opponent_spec}` on `{obs.name}` × n_stack=`{obs.n_stack}`, reward stack `{reward.name_or_filename}`, lr=`{trainer.lr}`, `{curriculum.name}`, `{trainer.total_timesteps:,}` steps.

- For single-agent runs, the templated summary omits `against {opponent_spec}` and `{learner_id}` reads from `cfg.env.learner_id` (which defaults to `drone_0` for `simple_env`).
- The override path lets curated curriculum-rung experiment YAMLs carry intent ("Curriculum step 1 from blue_v4: pretrain + switch to random_start. Tests whether …") while canaries / one-offs get the free auto-template.

### 3. Lineage

```markdown
## Lineage

- **init mode:** `<init.mode>`     (one of: scratch, pretrain, resume, warm_start)
- **parent:** `<init.parent>`
- **parent chain total:** <parent_chain_total> steps (this run is <total_timesteps> of that)
- **resolved wandb URI:** `wandb://<run>:<v|alias>` (if parent path matches a vendored model with `_wandb_metadata.json`)
```

- If `init.mode=scratch`, render only "init mode: scratch — no parent" and skip the rest.
- "resolved wandb URI" is best-effort: if `init.parent` is already a `wandb://...` URI, copy it; else if it's a filesystem path under `models/<name>/`, look up `<name>/_wandb_metadata.json` and report `wandb://<name>:v<version>`. Omit the line if neither path resolves.

### 4. Obs spec

```markdown
## Obs spec

**Name:** `<cfg.obs.name>` (<dim>-d)  ·  **n_stack:** <cfg.obs.n_stack>  ·  **Input dim:** <dim × n_stack>

| Slot | Block | Dim | Frame | Notes |
|------|-------|-----|-------|-------|
| 0:3   | ang_vel  | 3 | body | |
| ...
```

- The table is built from `SPEC_BY_NAME[cfg.obs.name].offsets()` — column slot ranges, dims, frames, and notes pulled directly from the `ObsBlock` dataclasses.
- If `cfg.obs.name` isn't in `SPEC_BY_NAME` (e.g., a legacy run with a deprecated name), render an error blockquote and a fallback line "see `.hydra/config.yaml:obs` for the recorded spec name".

### 5. Reward stack

```markdown
## Reward stack

**Source:** `conf/reward/<filename>.yaml`

| # | Term | Key coefficients | Agents |
|---|------|------------------|--------|
| 1 | TagEntryPulse | magnitude=5.0 | blue_0 (gainer), red_0 (loser) |
| ...
```

- Source line: derived from `.hydra/hydra.yaml:hydra.runtime.choices.reward` — Hydra records which group choice was picked (e.g., `team_v2`). The renderer reads that key + emits `conf/reward/<choice>.yaml`. If `.hydra/hydra.yaml` is absent or the key is missing (e.g., direct `_target_` override), the source line falls back to "(in-line override / unknown source)".
- Table rows: walk `instantiate(cfg.reward, _convert_="all").terms`. For each term, render `type(term).__name__` + a coefficient summary (read `dataclasses.fields(term)`, filter to numeric/dict types, render `name=value` joined by commas) + an agents summary (read the fields named `agents`, `gainer`/`loser`, `scorer`/`zero_sum_opponent`, `aggressor`/`victim`, `agent_to_target`, `agent_to_crash_flags`).
- **Adding a new reward term class doesn't require touching the renderer** — its fields show up automatically via `dataclasses.fields()`. Custom rendering hooks can be added later as a per-class dispatch if needed.

### 6. Env config

```markdown
## Env config

- **Opponent:** `<cfg.opponent.spec_string or _target_short>`  ·  **Learner:** `<cfg.env.learner_id>`
- **Episode:** <episode_seconds> s  ·  **Curriculum:** `<cfg.curriculum.name>`
- **Tag radius:** <m> m  ·  **Crash velocity threshold:** <m/s> m/s
- **Midpoint α:** <cfg.env.team_cfg.midpoint_alpha>  ·  **Crash aftermath:** <s> s
```

- For single-agent runs, render only `Episode`, `Curriculum` (and skip team-only lines).
- "Opponent" is a short identifier: `beeline_red`, `frozen:wandb://red_v1:prod`, `mixture:0.5*beeline,0.5*frozen`. Built by reading `cfg.opponent` and emitting the same human shorthand `opponent.from_spec` accepts.

### 7. Training hyperparams

```markdown
## Training hyperparams

- **Algorithm:** PPO  ·  **lr:** <lr>  ·  **total_timesteps:** <comma_formatted>
- **n_envs:** <n>  ·  **batch_size:** <n>  ·  **n_epochs:** <n>
- **gamma:** <γ>  ·  **gae_lambda:** <λ>  ·  **ent_coef:** <c>  ·  **clip_range:** <r>
```

Reads directly from `cfg.trainer`. Algorithm name is hardcoded "PPO" — current project only uses PPO; if/when other algorithms land, expand to read from `cfg.trainer._target_`.

### 8. Eval results

```markdown
## Eval results

- **best_eval_reward:** <reward> @ step <step,>  (if best-kind)
- **completed_steps:** <n>  ·  **wall_clock:** <Hh MMm SSs>
- **model_kind:** `best` | `final`
```

- Read from `.hydra/meta.yaml:final_stats`.
- If meta.yaml is missing (training crashed before writing it), render "(meta.yaml absent — eval section unavailable)" and skip the section's interior.
- If `model_kind == final` (eval skipped), render the section but omit `best_eval_reward @ step` lines.

### 9. W&B

```markdown
## W&B

- **Project:** `<entity>/<project>`
- **Run id:** `<run_name>_<timestamp>`
- **Artifact:** `<artifact_name>:v<N>`  (aliases: <list>)
```

- Read entirely from `_wandb_metadata.json` in the model/run dir.
- Section omitted entirely if `_wandb_metadata.json` is absent.

## Architecture

### Component map

```
scripts/_render_model_doc.py        (NEW — private, the pure renderer)
├── render_model_doc(run_dir: Path) -> str        # one public entrypoint
├── _load_run_context(run_dir) -> dict            # gather config, meta, wandb_meta
├── _section_header(ctx) -> str
├── _section_summary(ctx) -> str
├── _section_lineage(ctx) -> str
├── _section_obs_spec(ctx) -> str
├── _section_reward_stack(ctx) -> str
├── _section_env_config(ctx) -> str
├── _section_hyperparams(ctx) -> str
├── _section_eval_results(ctx) -> str
└── _section_wandb(ctx) -> str

scripts/render_model_doc.py         (NEW — public, the CLI entrypoint)
└── main(): parses --run-dir, calls render_model_doc, writes <run_dir>/MODEL.md

scripts/backfill_model_docs.py      (NEW — one-shot, backfill legacy)
└── loops over models/*/ and runs/*/*/, idempotent (skip if MODEL.md exists unless --force)

scripts/train.py                    (MODIFY — train-end call in finally:)
└── after meta.yaml write:
     try: (run_dir / "MODEL.md").write_text(render_model_doc(run_dir))
     except Exception as e: log.warning(...)

scripts/_artifact_io.py             (MODIFY — include MODEL.md in artifact)
└── log_run_artifact: art.add_file(MODEL.md) if it exists in run_dir

scripts/promote.py                  (MODIFY — copy MODEL.md alongside .hydra)
└── shutil.copy(run_dir / "MODEL.md", dest / "MODEL.md") if it exists

config_schema.py                    (MODIFY — add optional description field)
└── add `description: str = ""` to the top-level Config schema

Makefile                            (MODIFY — add describe-run target)
└── describe-run: invokes `python -m scripts.render_model_doc --run-dir <resolved>`
```

### Interfaces

- **`render_model_doc(run_dir: Path) -> str`** — pure function. No filesystem writes, no wandb network calls, no logging. Reads from `run_dir`. Returns the assembled Markdown string. Callers decide where to put it.
- **`_load_run_context(run_dir: Path) -> dict`** — gathers `{cfg, meta, wandb_meta, run_dir}` once. Each `_section_*` function takes this dict + uses what it needs. Keeps section functions independent; testing passes fixture dicts directly without touching disk.
- **`_section_*(ctx: dict) -> str`** — each section is a pure function. May raise; caller (`render_model_doc`) wraps each in try/except for per-section isolation.

### Data flow (train-time path)

```
train.py finally:
  1. write .hydra/meta.yaml                                (existing)
  2. doc = render_model_doc(run_dir)                       (NEW)
     reads:  .hydra/config.yaml                         (required)
             .hydra/meta.yaml                           (optional — degrades eval section)
             .hydra/hydra.yaml                          (optional — for reward group-choice filename)
             _wandb_metadata.json                       (optional — omits W&B section + URI fallback)
     uses:   instantiate(cfg.reward) for term coefficients
             SPEC_BY_NAME[cfg.obs.name].offsets() for block table
  3. (run_dir / "MODEL.md").write_text(doc)                (NEW)
  4. log_run_artifact(run_dir, ...)                        (existing, now picks up MODEL.md)
```

`render_model_doc` runs *before* `log_run_artifact` so the artifact's file list includes `MODEL.md`. Ordering is enforced by code, not by file presence.

### Key design choices

- **Renderer is pure.** All I/O at the boundaries (`render_model_doc` reads files at entry; callers write at exit). Makes testing trivial — pass a fixture dict to `_section_*` functions and assert on the returned string.
- **Renderer is best-effort, never load-bearing.** Train-time call wraps `render_model_doc` in try/except + warning log. Promotion/backfill propagate exceptions (they're explicit, the operator should see the failure).
- **Reward-term introspection via `dataclasses.fields()`.** Adding a new reward term class doesn't require touching the renderer. Custom per-class rendering hooks can be added later if a term has fields that need special formatting (e.g., a callable, a nested dict that's awkward as `name=value`).
- **Markdown tables for obs spec + reward stack.** Both are column-rich; tables are easier to scan than prose AND render correctly in the wandb UI's Markdown panel. Plain-text describe()-style output stays available via `make obs-specs` for terminal use.
- **W&B section is optional, omitted entirely when `_wandb_metadata.json` is absent.** Handles offline-mode runs and not-yet-uploaded runs without a clutter section saying "not available".
- **Frozen-at-training-time content.** All inputs (`.hydra/config.yaml`, `.hydra/meta.yaml`, `_wandb_metadata.json`) are snapshots from training time. Re-running `render_model_doc` against a run-dir post-hoc generates the same MODEL.md (unless the underlying snapshot files are mutated, e.g., during our 2026-05-15 obs-spec migration which rewrote `cfg.obs.name` in all 3 promoted models — that's an opt-in mutation, and `make describe-run` lets the operator refresh MODEL.md after such migrations).

## Error Handling

Three layers of defense:

1. **Per-section isolation.** `render_model_doc` wraps each `_section_*` call in try/except. A bad section renders as a Markdown blockquote: `> ⚠ Section <name> failed: <reason>` (visually flagged, scannable). Other sections still render. Catches: missing fields, unexpected cfg shape, Hydra instantiation errors in the reward stack section, unknown obs spec name in `SPEC_BY_NAME`.
2. **Missing-files tolerance.** `_load_run_context` reads each input with explicit fallbacks:
   - `.hydra/config.yaml` missing → fatal, raise (everything depends on it)
   - `.hydra/meta.yaml` missing → `ctx["meta"] = None` → eval section renders "(meta.yaml absent — partial doc)" + hyperparam section degrades gracefully where possible
   - `.hydra/hydra.yaml` missing → reward stack's `**Source:**` line falls back to "(in-line override / unknown source)"; rest of the reward stack section still renders from `cfg.reward` instantiation
   - `_wandb_metadata.json` missing → `ctx["wandb_meta"] = None` → header `**Status:**` reads `run-only` + `**W&B:**` line omitted + section 9 (W&B) omitted entirely + lineage section's "resolved wandb URI" line omitted
3. **Train-time isolation.** `train.py`'s `finally:` block wraps `render_model_doc` + write in try/except. Logs `MODEL.md generation failed: <reason>` at WARNING level. The training run succeeds regardless. The doc is best-effort; never load-bearing.

## Testing

New test file: `tests/scripts/test_render_model_doc.py`. Plus extension of one existing smoke test.

| Test category | What it verifies | Approx count |
|---|---|---|
| **Section units** | Each `_section_*` function on a fixture `ctx` dict produces expected substrings. No filesystem. | ~9 cases (one per section) |
| **End-to-end snapshot** | `render_model_doc(tmp_path)` on a hand-crafted `.hydra/` + `_wandb_metadata.json` matches `tests/scripts/fixtures/expected_model_doc.md`. | 1 case |
| **Graceful degradation** | (a) missing meta.yaml → no raise + "partial doc" marker. (b) missing `_wandb_metadata.json` → no `## W&B` section in output. (c) unknown obs spec name → error blockquote on obs section, other sections render normally. | 3 cases |
| **train.py smoke** | Extend `tests/scripts/test_train_smoke_wandb_disabled.py` to assert `<run_dir>/MODEL.md` exists after a short training run and contains the expected top-level header. | +1 assertion in existing test |
| **Backfill idempotence** | `tests/scripts/test_backfill_model_docs.py` — first run writes MODEL.md, second run without `--force` skips it, second run with `--force` overwrites. | 3 cases |
| **Reward-term introspection robustness** | A term class with a previously-unseen field name renders without raising (renderer falls through `dataclasses.fields()` generically). | 1 case |

Expected net: **+17 tests, 0 deletions.** Canary fingerprints (`test_scoring_canary` `step 434 / 7.3837`, `test_team_env_canary`) MUST stay byte-identical — we're not touching the train loop's reward arithmetic.

## Migration / Rollout

1. Land the feature as a single commit on `develop` (or a feature branch — operator's call at implementation time).
2. Run `python -m scripts.backfill_model_docs` once to seed `MODEL.md` for the 7 legacy promoted models in `models/`. Idempotent.
3. Update `brain/index.md` Recent Context + `brain/changelog.md` with a feature-landing entry.
4. Update `brain/models.md`: mention that each promoted model now has a `models/<name>/MODEL.md` companion (catalog entries can link to it).

## Open Implementation Questions

These are details that don't change the design but will need decisions during planning:

- **Exact substring format for `cfg.opponent`.** The opponent group is `_target_`-instantiated. Need a small helper to derive a "shorthand" string from the instantiated object — likely a `__repr__`-on-each-class change or a `Opponent.shorthand()` method. Could be folded into the existing `opponent.from_spec()` round-trip.
- **`description` field placement in `config_schema.py`.** Top-level `Config` dataclass is the cleanest spot. Alternative: experiment-specific subkey. Top-level is simpler.
- **Snapshot test update flow.** If the layout changes (new section, reordering), the fixture `expected_model_doc.md` needs an update. Options: (a) `pytest --snapshot-update` via a snapshot lib (e.g., syrupy), (b) plain string-comparison with a manual fixture edit, (c) generate the fixture from the test itself when missing. (b) is simplest and matches the project's existing test style (no external snapshot lib in use yet).
- **Wall-clock format in eval results.** `.hydra/meta.yaml:final_stats` may carry wall_clock as float seconds. Render as `Hh MMm SSs`. Small format helper.

## Files Touched (Summary)

**New (3):**
- `scripts/_render_model_doc.py` — pure renderer module
- `scripts/render_model_doc.py` — CLI entrypoint wrapping `_render_model_doc`
- `scripts/backfill_model_docs.py` — one-shot legacy backfill

**Modified (5):**
- `scripts/train.py` — call render in `finally:` after meta.yaml write
- `scripts/_artifact_io.py:log_run_artifact` — include MODEL.md in artifact
- `scripts/promote.py` — copy MODEL.md to `models/<name>/`
- `config_schema.py` — add `description: str = ""` field
- `Makefile` — `describe-run` target

**New tests (2 files + 1 extension):**
- `tests/scripts/test_render_model_doc.py` (~14 cases)
- `tests/scripts/test_backfill_model_docs.py` (~3 cases)
- `tests/scripts/test_train_smoke_wandb_disabled.py` — +1 MODEL.md-exists assertion

**Brain (post-landing, umbrella-level, not part of the commit):**
- `brain/index.md` Recent Context + `brain/changelog.md` feature entry
- `brain/models.md` — mention MODEL.md companions

## Cross-References

- Brainstormed in conversation 2026-05-16 (this spec).
- Prior: 2026-05-15 reward-magnitude unification onto YAML (commit `1db54a7` on develop) is a load-bearing dependency — the renderer relies on `instantiate(cfg.reward)` returning a `RewardStack` with introspectable term dataclasses, which is only true now that the YAML-built stack is what actually runs.
- Prior: 2026-05-15 obs-spec rename (commit `b525662` on develop) + `make obs-specs` describer (`c281f5c`) — the renderer reuses `SPEC_BY_NAME` + `ObsSpec.offsets()` for the obs table.
- Catalog overview: `brain/models.md` (umbrella-level, curated narrative — complements but does not overlap with per-model MODEL.md).
