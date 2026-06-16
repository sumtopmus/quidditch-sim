# RLlib Migration Step 6 — Retire SB3

**Date:** 2026-06-16
**Branch:** `feature/rllib-step-6-retire-sb3` (worktree off `develop`)
**Migration step:** 6 of 6 (final) — see `docs/superpowers/specs/2026-06-07-rllib-league-migration-design.md`, build-sequence item 6.
**Depends on:** Steps 1–5 GREEN and merged to `develop` (Step 5 = PR #21 `ecb88d7`).

## Context

The SB3 → RLlib migration is complete through Step 5: the dual-population PFSP
self-play league trains both `main_red` and `main_blue` on the new RLlib API
stack under Ray Tune, with curriculum anneal, eval battery, and a checkpoint /
promote / eval-team adapter — all merged to `develop` and validated by the 5M
run `gridcom/drone-quidditch/9752c_00000`.

The original migration design (2026-06-07) deferred deletion of the SB3 path to
a final step so each intermediate step would "leave a runnable system." That
condition is now met. **Step 6 deletes the dead SB3 training/eval path and makes
RLlib the sole, canonical training framework.**

This is a code-retirement refactor, not RL work. It is explicitly *separate*
from the open Step-5 model-quality loose end (Red never snapshots — eval
score-rate peaked 0.20 < the 0.5 bar). That tuning is deferred to its own later
effort and is **out of scope here.**

### Decisions settled in brainstorming (2026-06-16)

| Axis | Decision |
|------|----------|
| Scope | Step 6 only — pure SB3 retirement. Red-snapshot tuning deferred. |
| `eval_solo.py` | **Delete.** SB3-only; the RLlib eval battery already covers main-vs-scripted diagnostics; `QuidditchSimpleEnv` + the sim-level scoring canary survive (they need no model). |
| Canonical entrypoint | **Rename `scripts/train_rllib.py` → `scripts/train.py`.** RLlib becomes THE training path; `make train-rllib` collapses into `make train`. |
| Historical SB3 champions | **Accept loss of loadability.** Per the migration non-goals, `backup/` `.zip` checkpoints are historical references only; no SB3 load path is retained (recoverable from git if ever needed). |
| Validation depth | **Tests green + short smoke run.** Full pytest suite + a few-iteration RLlib Tune smoke run that checkpoints + `dsim` surfaces still work. No full 5M re-run required. |
| Plan shape | **Single phased plan** (not decomposed like Step 5). The work is one coherent deletion with a forced import-dependency order. |

## Goals

- Delete the SB3 training and single-agent eval path in its entirety.
- Make RLlib the sole training framework; `make train` / `dsim resume` /
  `python -m scripts.train` all drive the RLlib + Tune entrypoint.
- Remove the `stable-baselines3` dependency (the only SB3 dep declared —
  `pyproject.toml:8`; no `sb3-contrib`).
- Leave the framework-agnostic core and the entire `rllib/` package untouched
  and still passing.

## Non-goals

- **No Red-snapshot tuning** (`snapshot_threshold_red`, dense-anneal schedule).
  Separate later effort.
- **No new RLlib features** (video, eval, league behavior). Only deletion +
  the entrypoint rename + repointing.
- **No SB3→RLlib weight bridge / legacy import.** Already a standing non-goal.
- **No full 5M validation re-run** as a gate (smoke run suffices).

## The critical distinction: delete vs split vs keep

The naive view is "delete everything that imports `stable_baselines3`." That is
wrong: three modules are **shared with the live RLlib path** and must be
*surgically split*, not deleted. This is the central risk of the step.

### Delete outright (SB3-only — verified no live RLlib importer)

| Path | What it is |
|------|-----------|
| `scripts/train.py` | SB3 training entrypoint (replaced by the renamed `train_rllib.py`). |
| `scripts/_train_common.py` | SB3 callback/vec-env build helpers. |
| `scripts/callbacks.py` | `SuccessRateEvalCallback`, `ResumeProgressCallback`, `VideoRecorderCallback` (SB3 base classes). |
| `scripts/_wandb_logger.py` | `WandbLoggerCallback(BaseCallback)` — SB3 W&B bridge. |
| `scripts/_wandb_init.py` | SB3 opponent-name → spec map for run init (verify no other importer). |
| `scripts/eval_solo.py` | SB3-only single-agent eval (`PPO.load` on `QuidditchSimpleEnv`). |
| `core/policies/warm_start.py` | Cross-obs-spec input-layer surgery — no role in a scratch league. |
| `scripts/migrate_legacy_models.py` | One-shot legacy `.zip`+TOML → Hydra migrator; obsolete. |
| `conf/trainer/ppo.yaml`, `conf/trainer/ppo_finetune.yaml` | SB3 PPO hyperparameters. |
| `conf/init/{warm_start,pretrain,resume}.yaml` | SB3 init modes (RLlib resume = Tune restore via `dsim resume`). |
| `conf/opponent/{frozen,mixture}.yaml` | Reference PPO `.zip` frozen models; unused by the RLlib league. |
| `dsim sweep` (`create`/`agent`/`agents`) | W&B-sweep dispatch — superseded by Ray Tune. |
| `stable-baselines3` in `pyproject.toml` (line 8) | The dependency itself (no `sb3-contrib` declared). |

### Split carefully (shared with the live RLlib path — surgery)

- **`envs/quidditch/opponents.py`** — `rllib_modules.ScriptedRLModule` imports
  `from_spec` from here (line 24). **Keep:** the scripted `Opponent` classes
  (`ZeroOpponent`, `BeelineRed`, `BeelineBlue`, `IntercepterBlue`), the
  `_beeline_act` helpers, the `OPPONENTS` registry, and `from_spec`'s scripted
  branch. **Delete:** `OpponentControlledEnv` (the single-agent collapse),
  `FrameStackWrapper`, `FrozenPolicyOpponent` (PPO.load), the `frozen:` branch
  of `from_spec`, and the unused `mixture:` branch + `MixtureOpponent` (the
  RLlib league never uses mixtures; mixtures can nest `frozen:` leaves, so
  dropping them removes the last `.zip`-load path).

- **`core/eval_core.py`** — imported by `rllib/eval_battery.py` (line 19:
  `TERMINAL_BUCKETS`, `_classify_terminal`) and `core/eval_report.py`. **Keep:**
  `ScenarioSpec`, `ScenarioResult`, `EpisodeResult`, `TERMINAL_BUCKETS`,
  `_classify_terminal`, and the scripted-learner path of `run_scenario` (the
  `scripted:beeline_blue` sentinel, framework-agnostic). **Delete:** the SB3
  branch of `run_scenario` (`PPO.load` + `OpponentControlledEnv`).

- **`envs/quidditch/env_factories.py`** — **Delete** the SB3 vec-env machinery
  (`DummyVecEnv`/`SubprocVecEnv`/`VecEnv`/`VecFrameStack` imports + the
  factory methods that build them). Keep any framework-agnostic factory bits
  still referenced after the SB3 path is gone; if nothing survives, delete the
  file and its importers.

- **`tests/envs/quidditch/test_simple_env_contract.py`** — swap SB3's
  `check_env` for `gymnasium.utils.env_checker.check_env`; keep the contract +
  zero-action smoke episode.

### Keep untouched

`core/world.py`, `core/quadrotor.py`, `core/position_controller.py`,
`core/mjcf/*`, `core/drone/*`; `envs/quidditch/team_env.py`,
`envs/quidditch/simple_env.py` (+ `test_scoring_canary.py`),
`obs_spec.py`, `rewards/*`, `scene/scoring/tagging/crash`; the entire `rllib/`
package and `envs/quidditch/rllib_modules.py`.

## Make RLlib canonical

1. Delete SB3 `scripts/train.py`.
2. Rename `scripts/train_rllib.py` → `scripts/train.py`.
3. Repoint `make train` → `python -m scripts.train`; drop the now-redundant
   `make train-rllib` target.
4. Repoint **`dsim resume`** (`dsim/commands/resume.py` currently spawns
   `-m scripts.train`, the SB3 path) → the RLlib entrypoint.
5. Sync the CLI-surface docs in `README.md` and `repo/CLAUDE.md`.

## Build sequence (phased; order forced by imports)

1. **Make RLlib canonical** — rename + repoint `make train` / `dsim resume`,
   drop `make train-rllib`, sync docs. **Verify RLlib training still launches**
   (`make train EXP=rllib_league_step5` reaches the Tune loop) *before* any
   deletion or split.
2. **Delete SB3 leaf modules** — `_train_common.py`, `callbacks.py`,
   `_wandb_logger.py`, `_wandb_init.py`, `eval_solo.py`, `warm_start.py`,
   `migrate_legacy_models.py`. Grep importers before each deletion.
3. **Split the shared modules** — `opponents.py`, `core/eval_core.py`,
   `env_factories.py` per the surgery list above.
4. **Clean configs + CLI** — delete `conf/trainer/*`, `conf/init/{warm_start,
   pretrain,resume}.yaml`, `conf/opponent/{frozen,mixture}.yaml`; retire
   `dsim sweep`.
5. **Tests + dependency** — delete/port the SB3 test files; port
   `test_simple_env_contract.py` off SB3 `check_env`; remove `stable-baselines3`
   from `pyproject.toml`; `uv sync`.
6. **Verify** — full pytest green (fast + slow); short RLlib Tune smoke run
   completes + writes a checkpoint; `dsim list-runs` / `eval_team` / `promote`
   still work; `grep -r stable_baselines3 .` over the source tree returns
   nothing.

Each phase leaves a runnable, test-passing system.

## Testing strategy

- **Per-phase greps** — before deleting any module, grep its importers across
  the source tree (excluding `.venv`) to catch hidden consumers (e.g. demo
  scripts using `eval_core.run_scenario`, anything importing `_wandb_init`).
- **Split-module unit tests survive** — the scripted-opponent tests, the
  `eval_core` classify/scenario tests, and the RLlib eval-battery tests must
  stay green after the splits (they exercise the kept surface).
- **Canaries unchanged** — `test_scoring_canary.py` (sim-level, no model) and
  `test_team_env_canary.py` must keep their pinned fingerprints.
- **Smoke run** — a few-iteration `make train EXP=...` (or short Tune stop
  condition) completes, checkpoints, and the run is discoverable via
  `dsim list-runs` + loadable via `eval_team`.
- **Zero-SB3 assertion** — final `grep` confirms no `stable_baselines3` /
  `sb3_contrib` imports remain in source.

## Risks

- **Shared-module surgery is the real risk.** Cutting `opponents.py` /
  `eval_core.py` / `env_factories.py` can break the live RLlib path. Mitigation:
  phase 1 proves RLlib launches before any split; phase 6 smoke run re-confirms;
  the kept-surface unit tests catch regressions between.
- **Hidden importers** of the "delete" modules. Mitigation: per-phase importer
  greps (phase 2/3 gate).
- **`env_factories.py` may fully vanish** — if nothing framework-agnostic
  survives, deleting the file requires updating its importers; treat as part of
  the phase-3 split, not a surprise.

## Open questions

None blocking. The one residual code question — whether `env_factories.py` has
any framework-agnostic survivors — is resolved inline during phase 3 by reading
the file and its importers after the SB3 leaf modules are gone.
