# Hydra config + structured experiment management — Design

**Goal.** Replace the single committed `config/training.toml` (plus its gitignored local-tunings counterpart) with a Hydra-based config tree where each experiment is a named YAML, reward shaping is config-composable, and run state lives in `.hydra/config.yaml` + a tiny `meta.yaml` instead of the current `info.toml`. Both training scripts (`train_ppo.py`, `train_team_ppo.py`) consolidate into a single Hydra entrypoint with `_target_`-based instantiation of env, opponent, and reward stack.

**Why now.** Three pressures.

1. Per-experiment configuration is currently a smear: `run_name` and key hyperparams live in `config/training.toml` (committed), local hardware tunings (`n_envs=16`, `frame_stack=3`, eval cadence) live in a gitignored copy of the same file, and per-experiment overrides require either editing files or extending argparse. The 2026-05-11→13 Phase 2b iteration push ran into this — three reward-shaping iterations, two obs revisions, and a `--pretrain` flag all needed plumbing through both training scripts plus a gitignored TOML.
2. Reward magnitudes (`HOOP_ANCHOR_SCALE`, `TAG_DURATION_REWARD_MAX`, `DIST_REWARD_SCALE`, etc.) are module-level Python constants in `envs/quidditch/rewards.py`. A/B testing reward stacks requires code edits; sweeps require parametrizing the imports.
3. The pending W&B integration (Part 2 of this user request) needs structured config groups to drive sweeps. Doing it in TOML with argparse passthrough is technically possible but fights both Hydra's and W&B's idiomatic patterns.

The Part 1 scope (this spec) is *only* the config migration. Part 2 (W&B for sweeps + model registry + charts) and Part 3 (eval framework + checkpoint UX, possibly retiring or finishing the `feature/tui-launcher` branch) are separate brainstorming/spec cycles.

**Non-goals.**
- W&B integration, sweep authoring, model registry — Part 2.
- Eval framework (DeepEval and alternatives) — Part 3.
- TUI launcher fate — Part 3.
- `config/camera.toml` migration — a small follow-up after Part 1 lands; the camera config is a separate concern, not part of training config.
- `scripts/eval_ppo.py` / `scripts/eval_team.py` argparse-to-Hydra migration — eval doesn't need composition; can move later.
- PyTorch Lightning adoption — explicitly out of scope. SB3's training loop is hard-coded around PPO's rollout/update cadence and porting to Lightning is high-cost for no clear win.
- Backwards compat with the old TOML config format. The migration is a hard cut; both training scripts retire in the same change.

## Design Decisions

### 1. Hydra scope: instantiation-based, no Lightning

Hydra + `_target_` instantiation, keeping SB3 as the training loop. Borrow lightning-hydra-template's structural ideas (config groups, named experiment YAMLs, `_target_` for runtime-instantiated objects, optional gitignored local-overrides file) without buying into the Lightning module itself.

Rationale: instantiation is the right tool for opponent specs (currently parsed from strings via `from_spec`), reward shaping (currently module constants), and env construction (currently inlined in scripts). The user already thinks in dataclass terms (the 2026-05-13 obs-spec system is the precedent — first-class persisted dataclasses for obs layout).

### 2. Config group structure

```
repo/conf/
├── config.yaml                            # top-level defaults + global keys (run_name, seed)
├── trainer/
│   ├── ppo.yaml
│   └── ppo_finetune.yaml                  # lower-lr variant for warm-start runs
├── env/
│   ├── simple.yaml                        # _target_: envs.quidditch.env_factories.SimpleEnvFactory
│   └── team.yaml                          # _target_: envs.quidditch.env_factories.TeamEnvFactory
├── obs/
│   ├── simple.yaml                        # name: SIMPLE_ENV_OBS, n_stack: 1
│   ├── team.yaml                          # name: TEAM_ENV_OBS, n_stack: 1
│   └── augmented.yaml                     # name: AUGMENTED_OBS, n_stack: 3
├── reward/
│   ├── single_agent.yaml                  # scoring-canary stack
│   ├── team_v1.yaml                       # pre-2026-05-12 stack (ablation reference)
│   └── team_v2.yaml                       # current stack
├── opponent/
│   ├── none.yaml                          # for single-agent
│   ├── beeline_red.yaml
│   ├── beeline_blue.yaml
│   ├── intercepter_blue.yaml
│   ├── frozen.yaml                        # parameterized by model_path
│   └── mixture.yaml                       # parameterized by component list + weights
├── eval/
│   ├── default.yaml
│   └── fast.yaml                          # rare eval, fewer episodes, used on long runs
├── init/
│   ├── scratch.yaml                       # mode: scratch
│   ├── pretrain.yaml                      # mode: pretrain; mandatory: parent
│   ├── resume.yaml                        # mode: resume; mandatory: parent_run; parent_checkpoint defaults null (auto-resolve latest)
│   └── warm_start.yaml                    # mode: warm_start; implies obs_surgery=true
├── curriculum/
│   ├── fixed_start.yaml                   # randomise_start: false
│   └── random_start.yaml                  # randomise_start: true
├── experiment/                            # named ladder rungs + canary fixtures
│   ├── canary_single.yaml                 # pins single-agent scoring canary inputs
│   ├── canary_team.yaml                   # pins team env canary inputs
│   ├── red_v1.yaml                        # already-trained, recorded for reproducibility
│   ├── blue_v4.yaml                       # already-trained
│   ├── blue_v5.yaml                       # pending: pretrain blue_v4, beeline_red, random_start
│   └── blue_v6.yaml                       # planned: pretrain blue_v5, frozen red_v1, random_start
└── local/
    └── .gitignore                         # local/default.yaml is gitignored; included via `defaults: - optional local: default`
```

Each group's job: be the thing you swap when changing one axis of an experiment, independently of the others.

### 3. Hybrid structured/raw configs

Structured `@dataclass` schemas (registered via `ConfigStore`) for the stable foundation; raw DictConfig for `experiment/` YAMLs.

**Structured groups** (data-only, schemas live in `repo/config_schema.py`):
- `TrainerConfig` (PPO hparams: n_steps, batch_size, n_epochs, lr, gamma, gae_lambda, clip_range, ent_coef, total_timesteps).
- `EvalConfig` (eval_freq_steps, n_eval_episodes, video.* nested).
- `InitConfig` (mode, parent, parent_run, parent_checkpoint, obs_surgery — fields valid per mode; per-mode YAMLs leave others at sentinel defaults).
- `CurriculumConfig` (randomise_start, episode_seconds).
- `ObsConfig` (name selecting the canonical `ObsSpec` constant, n_stack).
- `RootConfig` (top-level: run_name, seed, plus references to the group configs above).

**Instantiation groups** (Python class IS the schema via `_target_`):
- Env factories (`SimpleEnvFactory`, `TeamEnvFactory`) — constructors take dependencies that Hydra instantiates first.
- Opponent classes (`BeelineRed`, `FrozenPolicyOpponent`, `MixtureOpponent`, etc.).
- Reward terms (9 dataclasses in `envs/quidditch/rewards/terms.py`) and `RewardStack`.

**Raw groups**: `experiment/*.yaml`. Each is a thin composition that selects entries from the other groups and overrides specific values. `# @package _global_` so overrides reach the top-level keys.

Example experiment YAML:

```yaml
# conf/experiment/blue_v5.yaml
# @package _global_
run_name: ppo_hoop_blue_5

defaults:
  - override /env: team
  - override /obs: augmented
  - override /reward: team_v2
  - override /opponent: beeline_red
  - override /init: pretrain
  - override /curriculum: random_start

init:
  parent: models/ppo_hoop_blue_4_20260511_202612/best_model
trainer:
  lr: 3e-4
```

### 4. Reward terms as composable components

Each reward term is a small dataclass in `envs/quidditch/rewards/terms.py`, instantiated via `_target_`. Term constants live in the reward YAML; physics thresholds shared with env termination logic (`TAG_RADIUS`, `CRASH_VEL_THR`, `TAG_COOLDOWN`) live in `env/team.yaml` and are referenced from reward YAML via OmegaConf interpolation `${env.team.tag_radius}`.

**Term roster** (9):
- `HoopDistancePenalty` — continuous, red-only.
- `HoopAnchor` — continuous, blue-only.
- `ZeroSumDistMirror` — continuous, blue-only (mirrors red's hoop-distance penalty).
- `ProximityGradedTag` — gated on tag_during; per-step `scale × max(0, 1 − dist/radius)`.
- `ClosingVelInTagZone` — gated on tag_during; per-step `scale × max(0, −d‖r−b‖/dt)`.
- `TagEntryPulse` — zero-sum entry pulse (Blue+5 / Red−5), cooldown-gated.
- `TakeDown` — `|v_rel·normal| > threshold` event; zero-sum ±20.
- `ScoreEvent` — red-only event; +10.
- `CrashEvent` — per-agent event (floor / wall / drone-drone / OOB); −20.

`RewardStack` (in `envs/quidditch/rewards/stack.py`) holds the term list and accumulates per-step per-agent rewards.

Existing `envs/quidditch/constants.py` keeps `TAG_RADIUS`, `CRASH_VEL_THR`, `TAG_COOLDOWN` as the canonical defaults that env YAMLs reference (`env/team.yaml` has e.g. `tag_radius: 0.3` — written explicitly, not imported from Python). Single source of truth for the *value* is the env YAML; reward terms reference via interpolation.

### 5. Run dir, info.toml retirement, and meta.yaml

**Run dir layout — unchanged:** `runs/${run_name}/${now:%Y%m%d_%H%M%S}/`. This is the standard Hydra pattern for any non-trivial project (`hydra.run.dir` customized to the project's convention). The Hydra "defaults" (`outputs/YYYY-MM-DD/HH-MM-SS/`) are tutorial-grade — every production Hydra project customizes the run dir, including lightning-hydra-template.

**info.toml retired entirely.** The run-state data it carries splits cleanly:

| Field | Lives where now |
|---|---|
| Run name, env mode, obs spec name, lineage edge (parent) | `.hydra/config.yaml` (Hydra-written) |
| Git hash | `.hydra/meta.yaml` (we write at start) |
| Parent chain `total_steps` | `.hydra/meta.yaml` (computed at start by walking parent's `.hydra/config.yaml`) |
| Final stats (wall time, completed steps, best eval reward, peak eval step) | `.hydra/meta.yaml` (appended at end / in `finally:`) |
| Obs spec `slots[]` expanded table | not persisted — recomputed from `cfg.obs.name` via a new `obs_spec.SPEC_BY_NAME` registry (added in Phase 1 scaffolding; trivial dict mapping `"SIMPLE_ENV_OBS" / "TEAM_ENV_OBS" / "AUGMENTED_OBS"` to the existing `ObsSpec` constants) |

Run dir after migration:

```
runs/<run_name>/<timestamp>/
├── .hydra/
│   ├── config.yaml             # Hydra: resolved config (run identity, lineage edge, obs choice, all hparams)
│   ├── overrides.yaml          # Hydra: CLI overrides for this launch
│   ├── hydra.yaml              # Hydra: Hydra's own meta
│   └── meta.yaml               # Ours: git hash, parent_chain_total, final stats
├── best_model.zip
├── checkpoints/
├── tb/
└── train.log
```

### 6. Init modes via Hydra group choice

The four mutex `--pretrain` / `--resume` / `--warm-start` / scratch-default CLI flags become four entries in the `conf/init/` group. Group choice itself enforces mutex — you literally cannot pick two simultaneously.

`obs_surgery` is the *defining* feature of `init=warm_start` — it's the only mode where strict `check_obs_compat` is bypassed and `warm_start_ppo_by_spec` is invoked. The `obs_surgery: true` field lives only in `init/warm_start.yaml`; there is no way to combine surgery with pretrain or resume.

### 7. Unified entrypoint, full-instantiation env construction

`scripts/train.py` is the single Hydra entrypoint. The env-mode decision (single vs team) is a group choice (`env=simple` vs `env=team`); `cfg.env` instantiates the right factory via `_target_`.

Factories take fully-instantiated dependencies (reward stack, opponent, env-config values). Hydra builds the dependency graph bottom-up, then injects.

```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_run_dir(run_dir, cfg)

    reward_stack = instantiate(cfg.reward)
    opponent     = instantiate(cfg.opponent)  # may be None for single-agent
    env_factory  = instantiate(cfg.env,
                               reward_stack=reward_stack,
                               opponent=opponent,
                               obs_spec_name=cfg.obs.name,
                               frame_stack=cfg.obs.n_stack,
                               # ...other env-level deps from cfg
                               )

    train_env = env_factory.build_train_env()
    eval_env  = env_factory.build_eval_env()

    model = build_or_load_model(cfg, train_env)
    callbacks = build_callbacks(cfg, run_dir, eval_env)

    write_meta_yaml(run_dir, cfg, start_stats=compute_start_stats(cfg))
    try:
        model.learn(total_timesteps=cfg.trainer.total_timesteps, callback=callbacks)
    finally:
        append_meta_yaml_final_stats(run_dir, model)
```

This is option C from the brainstorm — the most idiomatic Hydra shape, requiring extraction of `SimpleEnvFactory` and `TeamEnvFactory` classes that don't currently exist.

### 8. Makefile evolution + local overrides

`make train EXP=<name>` becomes the well-trodden path. The `train-team-red`, `train-team-red-warm`, `train-team-blue` targets are removed — they become experiment YAMLs. Direct `python -m scripts.train +experiment=<name>` always available for sweeps and one-offs.

Non-training targets (`resume`, `eval`, `promote`, `lineage`, `tensorboard`) stay; the in-target file paths update to read `.hydra/config.yaml` + `meta.yaml` instead of `info.toml`.

Hardware-specific local tunings (`n_envs=16`, `frame_stack=3`, eval cadence) move to a gitignored `conf/local/default.yaml`, included via `defaults: - optional local: default`. Missing-file is silent. This is the lightning-hydra-template pattern.

`make promote` copies `best_model.zip` AND the entire `.hydra/` directory into `models/<name>_<timestamp>/`. The `overrides.yaml` and `hydra.yaml` are kept (they're cheap and useful — `overrides.yaml` is the human-readable history of what was launched).

### 9. Validation layering

1. **Hydra composition** — wrong group name, missing mandatory `???`, init-mode mutex (free, by group choice).
2. **ConfigStore schema** — bad types in structured groups, missing required fields (free, at startup).
3. **Runtime checks in `train.main()`** — parent path exists, parent has `.hydra/`, `check_obs_compat` strict-raises unless `init.mode=warm_start`, run dir collision.
4. **Training-time** — SB3 exceptions bubble up; `.hydra/` and `meta.yaml` start-fields are written before training so partial state survives crashes.

Explicitly out of scope: cross-group sanity checks (e.g., "obs=team paired with env=simple"), reward magnitude bounds, GPU/MPS pre-flight.

## Migration Plan

One feature branch (`feature/hydra-config`), six internal phases. Canary fingerprints (`step 434 / reward 7.3837` single-agent + the team canary) pass at every phase boundary.

### Phase 1 — Scaffolding (no behavior change)

- Create `conf/` tree with empty `config.yaml`, empty group dirs, `local/.gitignore`.
- Create `config_schema.py` stub.
- Create `envs/quidditch/rewards/__init__.py` empty package, `envs/quidditch/env_factories.py` empty.
- Add `SPEC_BY_NAME: dict[str, ObsSpec]` registry to `envs/quidditch/obs_spec.py` (trivial 3-entry dict).
- Add `hydra-core` + `omegaconf` to `requirements.txt`.

**Acceptance:** `make test` passes unchanged. `make train` and `make train-team-red-warm` still work as before.

### Phase 2 — Reward refactor (TOML still in charge)

- Move per-term logic from `envs/quidditch/rewards.py` to `envs/quidditch/rewards/terms.py` (9 dataclass term classes).
- Add `envs/quidditch/rewards/stack.py` with `RewardStack`.
- `rewards/__init__.py` re-exports the legacy `compute_team_rewards(state)` API as a thin shim that instantiates `RewardStack` internally.
- `envs/quidditch/rewards.py` becomes a thin compat module re-exporting from `rewards/`.
- Add `tests/unit/test_reward_stack.py` including a "team_v2-equivalent stack reproduces current per-step reward exactly" test (pre-Hydra, runs in milliseconds, guards against silent refactor drift).

**Acceptance:** scoring canary `step 434 / reward 7.3837` unchanged. Team canary fingerprint unchanged.

**Risk note:** this is the most fragile phase. If a canary moves here, halt and diff the reward expression before continuing.

### Phase 3 — Env factories (TOML still in charge)

- Add `SimpleEnvFactory` and `TeamEnvFactory` to `envs/quidditch/env_factories.py`.
- `train_ppo.py` and `train_team_ppo.py` rewired to use the factories internally — same argparse surface, factory constructed from TOML values.
- Add `tests/unit/test_env_factories.py`.

**Acceptance:** canaries unchanged. Both train scripts launch correctly with existing Makefile targets.

### Phase 4 — Hydra cutover (both old and new in parallel)

- Populate `conf/` with all group YAMLs (trainer, env, obs, reward, opponent, eval, init, curriculum).
- Register schemas in `config_schema.py`, wire `register_configs()` into the new `scripts/train.py`.
- Create `scripts/train.py` with `@hydra.main` and the unified main flow from Decision 7.
- Create `conf/experiment/canary_single.yaml` and `conf/experiment/canary_team.yaml` pinning the canary's reward+obs+env+curriculum exactly.
- Create `conf/experiment/{blue_v4,blue_v5,red_v1}.yaml` — at least three real experiment YAMLs to validate the format end-to-end.
- Add `tests/unit/test_config_loading.py`, `test_config_schema.py`, `test_meta_yaml.py`, `conftest.py` `hydra_compose()` helper.
- Add new canary tests that compose the canary YAMLs and assert the same fingerprints — these run *alongside* the old canary tests.

**Acceptance:** both old AND new canary tests pass. The new `train.py` launches and completes a 1k-step smoke run matching what the old script would have produced.

### Phase 5 — Cutover (delete old, switch tests, retire TOML)

- Update `tests/integration/test_scoring_canary.py` and `test_team_env_canary.py` to use the Hydra-composed path; delete the old paths.
- Update `tests/integration/test_warm_start.py` for the new `init=warm_start` group.
- Update `tests/unit/test_obs_compat.py` to read parent's `.hydra/config.yaml`.
- Delete: `tests/unit/test_obs_spec_toml.py`, `test_backfill_obs_spec.py`, `test_team_pretrain_args.py`, `test_obs_surgery_args.py`, `test_write_run_info_pretrain.py`.
- Update `_train_common.py`: delete `tomllib` paths, `write_run_info`, info.toml-reading `read_parent_chain_total`. Add `write_meta_yaml`, `read_parent_chain_total_from_hydra`.
- Rewrite Makefile: `train` target takes `EXP=`; `train-team-red-warm` / `train-team-red` / `train-team-blue` deleted; other targets updated for the new file layout.
- Delete: `scripts/train_ppo.py`, `scripts/train_team_ppo.py`, `scripts/backfill_obs_spec.py`, `config/training.toml`, `config/training.toml.bak`.
- Delete `envs/quidditch/rewards.py` shim (consumers now import from `rewards/`).

**Acceptance:** full test suite passes. `make train EXP=canary_single` runs a smoke training to completion. `make resume`, `make eval`, `make lineage`, `make tensorboard`, `make promote` all work on a fresh test run.

### Phase 6 — Legacy model migration

The 7 promoted models on disk need their `run_info.toml` + `config.toml` converted to `.hydra/config.yaml` + `meta.yaml` so `init=pretrain init.parent=models/X/best_model` works against them.

- Write `scripts/migrate_legacy_models.py` (idempotent: skips if `.hydra/` exists).
- Run on one model first (`models/ppo_hoop_blue_4_*`), verify by running `init=pretrain` against it for a 1k-step smoke.
- Run on remaining 6. Spot-check `meta.yaml` `parent_chain_total` values match the old info.toml values.
- Update `scripts/lineage.py` to read `.hydra/config.yaml` chain; delete legacy info.toml reader.
- Add `tests/unit/test_migrate_legacy_models.py` with a fixture model dir.

**Acceptance:** `make lineage RUN_NAME=ppo_hoop_blue_4` produces the same chain output as before migration. A new training run with `init=pretrain init.parent=models/ppo_hoop_blue_4_*/best_model` loads correctly.

## Testing

### New tests

- `tests/unit/test_config_loading.py` — Hydra composition smoke: default compose succeeds, every `conf/experiment/*.yaml` composes cleanly, every `init=*` group entry composes, optional `local/default.yaml` works when absent.
- `tests/unit/test_config_schema.py` — schema validation: bad types raise at compose, missing mandatory `???` raises before training starts.
- `tests/unit/test_reward_stack.py` — empty stack returns zero rewards; each term instantiates from YAML; composed stack sums correctly; per-agent routing works; `team_v2.yaml` reproduces canary per-step reward exactly.
- `tests/unit/test_env_factories.py` — factories produce correct VecEnv types + obs shapes; dependency injection works.
- `tests/unit/test_meta_yaml.py` — write/append/read roundtrip; parent-chain walking on a fixture chain.
- `tests/unit/test_migrate_legacy_models.py` — converts a fixture model dir; idempotent on second run.

### Updated tests

- `tests/integration/test_scoring_canary.py` — composes `experiment/canary_single.yaml`, asserts `step 434 / reward 7.3837`.
- `tests/integration/test_team_env_canary.py` — composes `experiment/canary_team.yaml`.
- `tests/integration/test_warm_start.py` — uses `init=warm_start` group override.
- `tests/unit/test_obs_compat.py` — reads parent's `.hydra/config.yaml`.
- `tests/conftest.py` — gains `hydra_compose(experiment=None, overrides=None)` helper.

### Removed tests

- `tests/unit/test_obs_spec_toml.py` (info.toml `[obs]` block is gone).
- `tests/unit/test_backfill_obs_spec.py` (backfill script deleted; replaced by `migrate_legacy_models.py`).
- `tests/unit/test_team_pretrain_args.py`, `test_obs_surgery_args.py` (argparse flags retired).
- `tests/unit/test_write_run_info_pretrain.py` (function deleted).

### Canary preservation strategy

Canary numbers MUST stay byte-identical across this migration. Strategy:

1. Before refactoring rewards, write a one-shot helper that prints per-term contribution at a fixed canary state. Use it to author `conf/reward/single_agent.yaml` and `conf/reward/team_v2.yaml`.
2. `test_reward_stack.py`'s per-step exact-reproduction test runs in milliseconds — catches drift early.
3. Integration canaries are the end-to-end gate.
4. If a canary moves: **stop, don't repin**. The reward expression changed somewhere — track it down before merging. Repinning is allowed only when a reward term itself is intentionally being changed (excluded from this migration's scope).

## Out of scope / follow-ups

- `config/camera.toml` → `conf/camera/` — small follow-up after Part 1.
- `scripts/eval_ppo.py`, `scripts/eval_team.py` argparse → Hydra — eval doesn't urgently need composition; later cleanup.
- `opponents.from_spec()` legacy bridge deletion — wait until eval scripts migrate.
- W&B integration (sweeps, model registry, charts) — Part 2.
- Eval framework (DeepEval evaluation, or alternative) — Part 3.
- `feature/tui-launcher` rebase / merge / retire decision — Part 3. The branch has ~25 commits touching `_train_common.py`; after this work lands, that branch needs either a substantial rebase or a manual port onto the Hydra-based `_train_common.py`.
- An obs-spec content-hash in `meta.yaml` as a belt-and-suspenders guard against `obs_spec.py` slot-layout drift — deemed unnecessary given the `test_obs_spec.py` pinning test.
