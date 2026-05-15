# W&B integration — Design

**Goal.** Adopt Weights & Biases as the single dashboard, the canonical artifact registry, and the sweep controller for this project. TensorBoard is removed. The `models/` directory becomes a two-tier mix of vendored prod checkpoints (committed) and a wandb-download cache (gitignored). Hyperparameter sweeps run through the wandb sweep controller invoked against the existing Hydra entrypoint.

**Why now.** Part 1 (Hydra + structured experiment management) shipped on 2026-05-14. With the `conf/` tree, `_target_`-based instantiation, structured `@dataclass` schemas, and a single `scripts/train.py` entrypoint, the prerequisites for an integrated experiment tracker are in place. The current setup has three friction points Part 2 is sized to remove:

1. **Cross-run comparison is manual.** Today, comparing reward curves between `blue_v4` and `blue_v5` means opening two TensorBoard tabs and eyeballing. W&B's run comparator and parallel-coordinates plots are the obvious fix.
2. **Promotion is a directory copy.** `models/<name>/` is hand-managed by `make promote`. There's no audit trail, no version history, no place to record "this is the model we ran the 2026-05-08 video session against." Artifacts with aliases solve all three.
3. **No sweep machinery exists at all.** The 2026-05-11→13 push tuned reward shaping iteratively by hand-running configs one at a time. A wandb sweep over `(lr, ent_coef, HOOP_ANCHOR_SCALE)` could have done in a single overnight what took two days of session work.

Part 2 covers all three pillars (charts, registry, sweeps) in one feature branch, in that order during implementation so each layer can be smoke-tested before the next lands on top.

**Non-goals.**

- RLlib migration — open question in `brain/index.md`, but unrelated to tracker choice; this design assumes SB3 stays.
- Auto-promotion of sweep winners — a footgun (eval metric ≠ video-quality judgment); promotion stays manual.
- Mid-sweep cross-experiment comparison — sweep YAMLs target one base experiment each.
- `scripts/eval_team.py` argparse-to-Hydra migration — still out of scope (was deferred from Part 1).
- Backwards compat with TensorBoard — TB is fully removed; the `runs/<...>/events.out.tfevents.*` writes, the `make tensorboard` target, and the `_log_to_tensorboard` callback path all go away in the same commit. No dual-write phase.
- Git LFS — not needed at current model sizes (~1–5 MB per `.zip`); revisit only if vendored count grows by an order of magnitude.

## Design Decisions

### 1. Three subsystems with clean surfaces

Part 2 lands as three subsystems, each with one clear ownership boundary. They ship in the order below so each is smoke-testable before the next layers on top.

- **Run logger** (`scripts/_wandb_init.py` + a new `WandbCallback`). Owns `wandb.init`, the run config snapshot, scalar metrics, and per-cam video uploads. Replaces the `tensorboard_log=` argument on `PPO.__init__`.
- **Artifact registry** (`scripts/_artifact_io.py`). Two functions: `log_run_artifact(run, run_dir)` called at end of every training run; `resolve_parent(uri_or_path) → Path` called by `train.py` and `eval_team.py` when loading a parent checkpoint. The latter is the URI-to-local-path shim that lets `init.parent` accept either a filesystem path or a `wandb://` URI.
- **Sweep adapter** (`sweeps/*.yaml` + `command:` templates). wandb sweep YAMLs invoke `python -m scripts.train +experiment=<base> ${args_no_hyphens}`. No agent-side Python; wandb's template tokens handle the override forwarding.

These three are independent enough to be tested in isolation. The registry can ship without sweeps; sweeps can be tested with `WANDB_MODE=offline` before any artifacts get logged.

### 2. TensorBoard fully retired

TB removal is a hard cut, not a shadow phase:

- `PPO.__init__(..., tensorboard_log=tb)` → `tensorboard_log=None`. SB3's internal logger still fires; the `WandbCallback` from `wandb.integration.sb3` reads from that logger and forwards to wandb.
- `runs/<...>/events.out.tfevents.*` no longer written.
- `make tensorboard` target removed from the Makefile.
- `VideoRecorderCallback._log_to_tensorboard` (`scripts/callbacks.py:215`) renamed to `_log_to_wandb`; method body rewritten to use `wandb.Video(np_array, fps=..., format="mp4")`. The on-disk mp4 grid write is *unchanged* — it's still useful offline.
- `tensorboard` dropped from `requirements.txt`. `wandb` added. `moviepy` pin kept (still used by the on-disk mp4 path).

The cost of dual-writing TB + wandb would be ~80 lines of conditional plumbing for a temporary fallback that nobody uses once the new dashboard is live. Not worth it.

### 3. Run identity: `name == id == "<run_name>_<timestamp>"`

`scripts/_wandb_init.py:init_wandb(cfg, run_dir, role)` is the single call site. Called from `scripts/train.py` (after Hydra composes `cfg`) and from `scripts/eval_team.py` when `WANDB=1`.

Init signature:

```python
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "drone-quidditch"),
    entity=os.environ.get("WANDB_ENTITY"),     # None → user's default entity
    name=f"{cfg.run_name}_{timestamp}",        # matches Hydra run dir basename
    id=f"{cfg.run_name}_{timestamp}",          # wandb id == dir basename for easy crosswalk
    dir=str(run_dir),                          # wandb/ writes land under runs/<...>/
    group=cfg.run_name,                        # all timestamps of one experiment cluster
    job_type=_resolve_job_type(role),          # "train" | "sweep-train" | "eval"
    tags=[
        cfg.env.learner_id,
        cfg.obs.name,
        cfg.init.mode,
        _opponent_short_name(cfg.opponent),
    ],
    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    mode=os.environ.get("WANDB_MODE", "online"),
    resume="allow",                            # `make resume-team` re-attaches to parent id
    notes="",
)
```

Key choices and why each:

- **`name == id == "<run_name>_<timestamp>"`.** Wandb's run id is the same string as the on-disk dir basename. Any error message from either side points to the same artifact; `scripts/lineage.py` doesn't have to maintain a name↔id table.
- **`group=cfg.run_name`** (not the timestamped variant). The N timestamps of `ppo_hoop_blue_5` (multiple attempts of the same `conf/experiment/blue_v5.yaml`) cluster together in the dashboard's run-group view.
- **`job_type`** dispatches on whether `WANDB_SWEEP_ID` is set in the environment (set automatically by `wandb agent`); the sweep-child case becomes `"sweep-train"`. Free dashboard filter between manual training and sweep children.
- **`tags`** are derived from cfg — `learner_id` ("red_0" / "blue_0"), `obs.name` ("AUGMENTED_OBS"), `init.mode` ("pretrain" / "scratch" / "warm_start" / "resume"), opponent class short name ("BeelineRed" → "beeline_red"). Tags are W&B's primary filter mechanism. A per-experiment `tags_extra` list in `conf/wandb/default.yaml` can be added in experiment YAMLs for ad-hoc tagging.
- **`config`** is the fully-resolved Hydra config, dumped via `OmegaConf.to_container(..., resolve=True)`. Interpolations are baked in so the dashboard shows exactly what training ran with. Nested keys (e.g. `trainer.lr`) become wandb's faceted-config keys for parallel coords.
- **`mode`** is read from `WANDB_MODE`. `OFFLINE=1` in the Makefile maps to `WANDB_MODE=offline`. A `WANDB_MODE=disabled` is set in `tests/conftest.py` so the pytest suite never touches the network.
- **`resume="allow"`.** When `make resume-team` resumes an existing run, the resumed call re-uses the parent's id; the SB3 callbacks append to the same wandb run rather than starting a new one. Crash-restart timelines stay continuous.

A new conf group `conf/wandb/`:

- `conf/wandb/default.yaml` — `project`, `entity_override` (null), `tags_extra` (empty list), `notes` (empty string), `log_gradients` (false).
- `WandbConfig` registered in `config_schema.py` with these fields.
- `conf/config.yaml` defaults list gains `- wandb: default`.

### 4. Charts and metrics

`WandbCallback` from `wandb.integration.sb3` hooks SB3's internal `Logger` and forwards every recorded value to `wandb.log`. The full set of metrics that already exist today flow without code changes:

- `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `time/fps`, `time/iterations`
- `train/loss`, `train/value_loss`, `train/policy_gradient_loss`, `train/entropy_loss`, `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`
- `eval/mean_reward`, `eval/mean_ep_length` (from `EvalCallback`)

Custom additions in this Part 2:

- **Optional gradient/weight histograms** via `WandbCallback(log="gradients")`. Off by default (`conf/wandb/default.yaml:log_gradients: false`); experiments can opt in.

**Per-term reward curves are deferred.** Today `RewardStack.compute_step` returns only per-agent totals — per-term contributions are summed in-place and discarded. Surfacing them would require a small change to `RewardStack` (e.g., stash `self.last_per_term` after each step) plus a sister change to both env `step()` methods to forward the dict into `info[]`. That's a real chunk of work touching the env hot path and best done with its own canary check. Pushed out of Part 2 — see Open Questions.

`wandb.log` is called with `step=model.num_timesteps`, matching the existing TB x-axis. For pretrained/resumed runs, `model.num_timesteps` is the *cumulative* step count across the lineage (already true today — SB3's `reset_num_timesteps=False` on resume; `parent_chain_total` is also in `.hydra/meta.yaml`). So a `blue_v5` chart starts at step 10M, not 0 — lineage charts overlay continuously in the comparator.

System metrics (GPU / CPU / RAM / disk usage) are enabled by default via `wandb.init` and require no opt-in.

### 5. Videos

`VideoRecorderCallback._log_to_tensorboard` is renamed `_log_to_wandb`. The per-cam fanout is preserved — each cam (`south`, `east`, `top`, `fixed`) becomes a separate `eval/video/<cam>` panel in the dashboard. wandb groups them automatically.

Implementation: `wandb.log({"eval/video/south": wandb.Video(frames, fps=cfg.eval.video.fps, format="mp4"), ...}, step=model.num_timesteps)`. wandb accepts `(T, H, W, C)` numpy arrays directly and encodes mp4 in-memory.

The on-disk stitched mp4 grid (`runs/<run>/<ts>/videos/eval_step_<N>.mp4`) continues to be written for offline viewing and scrubbing without the dashboard. The on-disk path's moviepy 1.x pin stays load-bearing for that path only.

Cost estimate: per-eval ≈ 4 cams × ~3 MB = 12 MB; per 10M-step run with eval cadence 200k and video every 2nd eval ≈ 25 evals × 12 MB = 300 MB uploaded. Well inside W&B's free-tier 100 GB.

`wandb.Video` uploads silently retry on transient failures. If a single video fails mid-run, wandb buffers and retries; no callback-side retry logic needed.

### 6. Artifacts: auto-log every run, aliases mark canonical

Every successful training run logs one artifact at the end:

```python
artifact = wandb.Artifact(
    name=cfg.run_name,                  # e.g. "ppo_hoop_blue_5" (no timestamp)
    type="model",
    metadata={
        "git_sha":             meta.git_sha,
        "obs_spec":            cfg.obs.name,
        "n_stack":             cfg.obs.n_stack,
        "learner_id":          cfg.env.learner_id,
        "init_mode":           cfg.init.mode,
        "parent_uri":          cfg.init.parent,           # path or wandb URI
        "total_steps":         model.num_timesteps,
        "parent_chain_total":  parent_chain_total,
        "best_eval_reward":    best_eval_reward,
    },
)
artifact.add_file(run_dir / "best_model.zip")
artifact.add_dir(run_dir / ".hydra")                       # config.yaml + meta.yaml
wandb.log_artifact(artifact, aliases=["latest"])           # promotion adds more
```

Multiple training attempts of `ppo_hoop_blue_5` produce `ppo_hoop_blue_5:v0`, `:v1`, … wandb's `:latest` alias always points to the newest version.

**Promotion (`make promote RUN=<run_dir>`)** becomes a two-step thin wrapper, implemented as `scripts/promote.py`:

1. **Wandb side.** Look up the artifact for this run via `wandb.Api().run(...).logged_artifacts()[0]`; add aliases `["prod", cfg.run_name]`. Aliases are mutable and move as new versions get promoted.
2. **Repo side.** Copy `run_dir/{best_model.zip, .hydra/}` → `models/<cfg.run_name>/`, and write `models/<cfg.run_name>/_wandb_metadata.json` pinning the *immutable* `v<N>` (not the alias). Print a hint: `→ models/<run_name>/ updated. Run 'git add models/<run_name>/ && git commit' to vendor this checkpoint.`

The git commit is *manual*. This lets the user promote-without-vendoring (a "current best" you're not ready to commit) or vendor-without-promoting (a sentimental milestone artifact you want in the repo even though `:prod` has moved on).

### 7. The `models/` two-tier shape

```
models/
├── .cache/                                   # gitignored — wandb downloads land here
├── .gitignore                                # ignores .cache/
├── ppo_hoop_red_1/                           # committed: promoted-and-vendored
│   ├── best_model.zip                        # ~1–5 MB
│   ├── .hydra/{config,meta}.yaml
│   └── _wandb_metadata.json                  # pins artifact name + immutable v<N>
├── ppo_hoop_blue_4/                          # committed
│   └── ...
└── ...
```

`scripts/_artifact_io.py:resolve_parent(uri_or_path) → Path` is the single resolver:

1. **Filesystem path** (`models/ppo_hoop_blue_4_20260511_202612/best_model` or `runs/...`) → returned as-is. Legacy.
2. **`wandb://<run>:<alias>`** (shorthand, current project context) or **`wandb-artifact://<entity>/<project>/<run>:<alias>`** (fully qualified):
   - Resolve the alias to an immutable `v<N>` via `wandb.Api().artifact(uri)` (cheap metadata fetch).
   - Check `models/<run>/_wandb_metadata.json`; if it pins the same `<run>:v<N>`, return `models/<run>/best_model.zip` (no further network).
   - Else download via `artifact.download(root=models/.cache/<run>_v<N>/)`, return that path.
   - In *both* cases, also call `wandb.use_artifact(uri)` on the current wandb run so the lineage DAG records the edge. (`use_artifact` is idempotent within a run — calling it twice for the same artifact is fine.)
3. **`wandb://<run>:latest`** in a *checked-in* `conf/experiment/*.yaml` → schema validation error. `:latest` is allowed interactively, banned in committed configs. Reason: `:latest` silently shifts as new runs land; checked-in lineage parents must pin a stable alias (`:prod` or `:<run_name>`).

The two-tier shape buys: (a) `make eval-team BLUE=frozen:models/...` paths keep working with zero change; (b) fresh `git clone` has the vendored prod checkpoints available without any wandb credentials; (c) `ls models/` shows what's locally usable without hitting the wandb API; (d) cache pollution stays in `.cache/`, easy to wipe.

The `_wandb_metadata.json` pins the immutable version (e.g. `"v3"`), *not* the alias. If a future `make promote` shifts `:prod` to a newer version, the resolver correctly detects the committed copy is stale and falls through to downloading the new version. The committed dir doesn't silently change meaning under a moving alias.

### 8. Lineage walker — dual-walk with offline fallback

`scripts/lineage.py` is split into two walkers, sharing one CLI entry point.

**Walker A — local-only.** Walks `_wandb_metadata.json` files in `models/<run>/`. Each file carries `parent_uri`; resolve → read sibling `_wandb_metadata.json` → recurse. Works fully offline as long as each link is vendored. Output is a chain of `(run_name, version, step_count, git_sha)`.

**Walker B — wandb API.** Uses native artifact lineage. Every training run that calls `wandb.use_artifact(parent_uri)` and `wandb.log_artifact(child)` records the edge in W&B's DAG. The walker uses `artifact.logged_by().used_artifacts()` recursively. Richer output: shows un-vendored experiments, abandoned branches, crashed runs.

**`make lineage RUN=<spec>`** dispatches:

- `WANDB_MODE=offline` or `--local` flag → Walker A only.
- Default → Walker B with Walker A as fallback on API failure (warning printed).
- `--both` → print both side-by-side. A divergence usually means "the vendored chain skips intermediate runs that the wandb DAG shows" — informative, not an error.

**Resume special-case.** `init.mode=resume` is conceptually the same training process continuing, not a derivation. The walker collapses resume chains visually: `red_v1 (resumed 2026-05-02, +5M steps)` instead of `red_v1 → red_v1.resumed_1`. Driven by the `init_mode` field in `_wandb_metadata.json` / wandb artifact metadata.

### 9. Sweeps

Wandb sweep YAMLs live in `sweeps/` at the repo root, *not* under `conf/`. They use wandb's schema, not Hydra's; nesting under `conf/` would mislead Hydra into trying to compose them.

Example `sweeps/blue_v5_lr_ent.yaml`:

```yaml
program: scripts/train.py                          # nominal; overridden by command:
method: bayes
metric:
  name: eval/mean_reward
  goal: maximize
parameters:
  trainer.lr:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
  trainer.ent_coef:
    values: [0.005, 0.01, 0.02]
  trainer.batch_size:
    values: [256, 512, 1024]
early_terminate:
  type: hyperband
  min_iter: 5                                      # ≥5 evals (≈1M steps) before a run can be killed
  s: 2
command:
  - ${env}
  - python
  - -m
  - scripts.train
  - +experiment=blue_v5                            # base experiment
  - ${args_no_hyphens}                             # → "trainer.lr=3e-4 trainer.ent_coef=0.01 ..."
  - hydra.run.dir=runs/${run_name}_sweep/${now:%Y%m%d_%H%M%S}_${oc.env:WANDB_RUN_ID}
```

Key choices:

- **`${args_no_hyphens}`** is wandb's template token for "all hyperparams as `key=value` (no leading dashes)." Hydra's CLI override syntax is exactly `key=value`. Zero glue code.
- **`+experiment=<base>`** sets the base config; each sweep YAML targets one experiment. Cross-experiment sweeps would mix `eval/mean_reward` scales from different reward stacks and corrupt the Bayes posterior; the design deliberately rules this out.
- **`hydra.run.dir`** override interpolates `${oc.env:WANDB_RUN_ID}` (OmegaConf's env resolver — `${WANDB_RUN_ID}` alone wouldn't expand) so sweep children land in `runs/<exp>_sweep/<ts>_<wandb_id>/` (predictable bulk-delete location).
- **Hyperband early termination** is opt-in per sweep YAML. Disabled by default in the first sweep until the eval signal is trusted. `min_iter=5` over `eval_freq_steps=200_000` = no kill before 1M steps.

**Makefile surface:**

```makefile
sweep:                                             # creates the controller, prints sweep_id
	@test -n "$(SWEEP)" || (echo "set SWEEP=<name>" && exit 1)
	wandb sweep --project drone-quidditch sweeps/$(SWEEP).yaml

sweep-agent:                                       # runs one agent
	@test -n "$(ID)" || (echo "set ID=<sweep_id>" && exit 1)
	wandb agent $(ID)

sweep-agents:                                      # N parallel agents on same machine
	@test -n "$(ID)" && test -n "$(N)" || (echo "set ID and N" && exit 1)
	@for i in $$(seq 1 $(N)); do wandb agent $(ID) & done; wait
```

Default daily use: `make sweep SWEEP=blue_v5_lr_ent` → copy ID → `make sweep-agent ID=<id>`. `N=1` is the right default for CPU-bound trainings on a laptop; parallel agents thrash unless individual runs are short.

**No auto-promotion of sweep winners.** After a sweep completes, the user inspects the W&B parallel-coordinates view, picks a winner manually, runs `make promote RUN=<winner_run_dir>` like any other run. Auto-promotion based on `eval/mean_reward` is a footgun: the real quality signal often comes from watching the per-cam video, which Bayes can't see.

**Future helper (out of scope for Part 2).** A `scripts/render_sweep.py` (~30 lines) that takes a sweep template + an experiment name and emits a concrete sweep YAML. Ship when the second sweep YAML duplicates >50% of the first.

### 10. Migration: `scripts/upload_legacy_models.py`

One-shot, idempotent. Uploads the 7 currently-promoted models in `models/` as wandb artifacts so the lineage DAG and `:prod` aliases exist before Part 2 ships. Sibling to the existing `scripts/migrate_legacy_models.py`.

For each `models/<dir>/`:

1. Build a wandb artifact named after the run (`ppo_hoop_red_1`, `ppo_hoop_blue_4`, etc.), type `model`.
2. `add_file(best_model.zip)`, `add_dir(.hydra/)`.
3. `metadata` populated from the existing `.hydra/meta.yaml` (git_sha, parent_chain_total) plus the hand-curated `LEGACY_SPECS` map from `migrate_legacy_models.py` (obs spec, n_stack, learner_id).
4. `log_artifact(artifact, aliases=["prod", cfg.run_name, "v0"])`. The `prod` alias marks it canonical; the `<run_name>` alias is per-experiment-current.
5. Write `_wandb_metadata.json` into the existing `models/<dir>/`, pinning `name=<run_name>` and `version=v0`. So existing `models/ppo_hoop_*` directories work as vendored dirs immediately, no rename needed.

Run via `python -m scripts.upload_legacy_models` (no Hydra; standalone). Idempotent — re-runs detect existing artifacts and skip. Run *once*, by the user, as part of merging this branch. Not auto-invoked.

**The `parent_uri` problem for legacy artifacts.** The 7 promoted models reference each other via filesystem paths in their `init.parent` (e.g. `red_v1` points to `models/ppo_hoop_rand_start_*/best_model`). The migration script translates these filesystem references into wandb URIs (`wandb://ppo_hoop_rand_start:v0`) using the same lookup table that names the new artifacts. After migration, the artifact metadata's `parent_uri` is a wandb URI for newly-trained runs and a path-then-URI mapping for the legacy seeds.

**Order of operations on the branch:**

1. Implement the run logger, artifact registry, and resolver.
2. Run `upload_legacy_models.py` on the dev machine; commit the resulting `_wandb_metadata.json` files into `models/<dir>/`.
3. Land the wandb dependency, the conf changes, the Makefile changes, the TB removal in one PR.

### 11. Tests

`tests/conftest.py` already sets `KMP_DUPLICATE_LIB_OK=TRUE`. It gains one more line:

```python
os.environ["WANDB_MODE"] = "disabled"
```

In `"disabled"` mode `wandb.init` returns a no-op stub; no network, no files written, no environment variables set elsewhere matter. The full test suite runs without touching wandb.

New tests:

- `tests/unit/test_wandb_init.py` — assert `init_wandb(cfg, ...)` builds the expected name/group/tags/config payload for canary_single and canary_team experiment YAMLs. Doesn't actually call `wandb.init`; mocks at the SDK boundary.
- `tests/unit/test_artifact_resolve.py` — `resolve_parent` cases: filesystem path returned as-is; wandb URI with matching cache hits committed dir; wandb URI with stale cache falls through to mocked download; `:latest` in YAML-loaded cfg raises ConfigValidationError.
- `tests/unit/test_promote.py` — `scripts/promote.py` correctly extracts the run from a run dir, builds the alias list, copies into `models/<name>/`, writes pinned-version metadata. wandb API mocked.
- `tests/integration/test_train_smoke_wandb_disabled.py` — single rollout (`n_steps=1024`, `total_timesteps=2048`, eval skipped) with `WANDB_MODE=disabled`. Confirms `scripts/train.py` runs end-to-end with the new logger callback wired in. Canary fingerprint preserved.

Existing tests touched:

- `tests/integration/test_scoring_canary.py` — canary stays `step 434 / reward 7.3837`. wandb being disabled in conftest means this is purely a regression check, not a wandb test.
- `tests/integration/test_team_env_canary.py` — same.
- `tests/integration/test_warm_start.py` — already broken against `blue_4` (gated on `MODEL=`), not touched.

### 12. Conf and schema changes

New group `conf/wandb/`:

- `conf/wandb/default.yaml`:
  ```yaml
  project: drone-quidditch
  entity_override: null                            # null → WANDB_ENTITY env var or user default
  tags_extra: []                                   # appended to derived tags
  notes: ""
  log_gradients: false
  ```

`config_schema.py` additions:

```python
@dataclass
class WandbConfig:
    project: str = "drone-quidditch"
    entity_override: str | None = None
    tags_extra: list[str] = field(default_factory=list)
    notes: str = ""
    log_gradients: bool = False
```

Registered via `cs.store(group="wandb", name="schema", node=WandbConfig)` in `register_configs()`.

`conf/config.yaml` defaults list gains:

```yaml
defaults:
  - trainer: ppo
  - env: team
  - obs: augmented
  - reward: team_v2
  - opponent: beeline_red
  - eval: default
  - init: scratch
  - curriculum: random_start
  - wandb: default                                 # new
  - _self_
  - optional local: default
```

`InitConfig` is *not* modified — `parent` is already `str | None`; the wandb URI scheme is parsed by `_artifact_io.py`, not by the schema. But `__post_init__` validation gains: when `mode != "scratch"` and `parent` starts with `wandb://`, reject `parent.endswith(":latest")` with a clear error. (Schema-level ban on `:latest` for checked-in configs.)

### 13. File footprint

**New files (9):**

- `scripts/_wandb_init.py` — `init_wandb(cfg, run_dir, role)`, ~80 lines.
- `scripts/_artifact_io.py` — `log_run_artifact`, `resolve_parent`, ~120 lines.
- `scripts/promote.py` — promotion wrapper, ~60 lines. Replaces whatever `make promote` calls today (currently a shell `cp -r` inline in the Makefile).
- `scripts/upload_legacy_models.py` — one-shot migration, ~150 lines.
- `conf/wandb/default.yaml` — ~10 lines.
- `sweeps/blue_v5_lr_ent.yaml` — first concrete sweep, shipped as the initial example.
- `models/.gitignore` — one-line, ignores `.cache/`.
- `tests/unit/test_wandb_init.py`, `tests/unit/test_artifact_resolve.py`, `tests/unit/test_promote.py`, `tests/integration/test_train_smoke_wandb_disabled.py` — ~250 lines total across all four.

**Modified files (10):**

- `scripts/train.py` — `tensorboard_log=tb` → `tensorboard_log=None`; add `init_wandb` call; add `WandbCallback`; wire `_artifact_io.log_run_artifact` at end; wire `_artifact_io.resolve_parent` on `init.parent` load (3 sites: pretrain, resume, warm_start branches).
- `scripts/_train_common.py` — `build_callbacks` gains wandb-aware callbacks.
- `scripts/callbacks.py` — `VideoRecorderCallback._log_to_tensorboard` → `_log_to_wandb`.
- `scripts/eval_team.py` — optional `--wandb` flag (or `WANDB=1` env); when set, calls `init_wandb` with `role="eval"`. `resolve_parent` on the `--red` / `--blue` checkpoint specs.
- `scripts/eval_ppo.py` — `resolve_parent` on the `--model` argument so single-agent eval also accepts `wandb://` URIs. No `--wandb` flag (single-agent training is closed per the brain; eval is occasional and doesn't need its own dashboard run).
- `scripts/lineage.py` — split into Walker A (local) and Walker B (wandb API); CLI dispatch.
- `config_schema.py` — `WandbConfig` dataclass + `register_configs()` line + `InitConfig.__post_init__` `:latest` ban.
- `conf/config.yaml` — defaults list gains `wandb: default`.
- `requirements.txt` — drop `tensorboard`, add `wandb`.
- `Makefile` — drop `tensorboard` target; add `sweep`, `sweep-agent`, `sweep-agents`; thin `promote` wrapper around `python -m scripts.promote`.
- `tests/conftest.py` — add `WANDB_MODE=disabled`.

**Deleted (1):**

- The `tensorboard` Makefile target (line removal, not a file).

Tests expected to grow ~46 → ~60. Canary fingerprints (`step 434 / reward 7.3837` + team canary) byte-identical (wandb disabled in tests means no behavior change).

## Open Questions

1. **`WANDB_ENTITY` default.** The design assumes the user has a wandb account and `WANDB_ENTITY` is either set in the environment or left to wandb's per-user default. If a contributor without an account tries to `make train`, the failure mode should be a clear error (not a stalled init). Worth a startup-time check: if `mode=online` and no entity resolvable, exit early with "set WANDB_ENTITY or use OFFLINE=1."
2. **Lineage walker output format.** Today's `make lineage` prints a plain-text chain. The dual-walk design adds an option of `--both` side-by-side; format TBD during implementation. Probably aligned columns; possibly JSON via `--json` for tooling.
3. **`scripts/eval_team.py` and Hydra.** Eval still uses argparse. The `--wandb` flag added here is the *only* eval-side argparse change; full Hydra migration of eval is still deferred (Part 3 candidate).
4. **`upload_legacy_models.py` and the `parent_chain_total` correctness.** The script copies `parent_chain_total` from each model's existing `.hydra/meta.yaml`. If any meta.yaml is missing or wrong for a legacy model, the wandb artifact's `total_steps` will inherit the bad value. Worth a manual spot-check against the brain's "Best models" section before merging.
5. **Per-term reward curves.** Deferred from Part 2 (Section 4). Would require a small extension to `RewardStack` (stash `last_per_term`) plus an env-side change to forward it into `info[]`. Likely a Part 3 candidate or a small standalone change once Part 2 is live.
