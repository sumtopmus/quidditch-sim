# RLlib Migration Step 6 — Retire SB3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the dead Stable-Baselines3 training/eval path and make RLlib (`scripts/train_rllib.py`, renamed to `scripts/train.py`) the sole, canonical training framework.

**Architecture:** A deletion refactor, not new features. Three modules are *shared* with the live RLlib path (`opponents.py`, `core/eval_core.py`, `conf/env/team.yaml`) and are **split** (surgery), not deleted. Everything else SB3 is deleted outright. Tasks are ordered so each commit leaves the full `pytest` suite green (a test importing a deleted module fails *collection*, so every source deletion bundles its dependent test deletions/ports in the same commit).

**Tech Stack:** Python 3.13, uv, Hydra, Ray RLlib (new API stack) + Ray Tune, W&B, pytest, MuJoCo.

**Spec:** `docs/superpowers/specs/2026-06-16-rllib-step-6-retire-sb3-design.md`

---

## Execution notes (read first)

- **Worktree:** all work happens in this worktree (`worktrees/feature/rllib-step-6-retire-sb3`, branch `feature/rllib-step-6-retire-sb3` off `develop`). It already exists.
- **Sandbox:** `uv`, `make`, `dsim`, and `pytest` (run via `uv run`) need the command sandbox **disabled** (they hit `~/.cache/uv`). Run those with sandbox off from the start.
- **Commits:** GPG-sign every commit (`git commit -S`). Commit-type taxonomy: `refactor:` for moves/restructures/deletions of working code, `chore:` only for true housekeeping, `test:`, `docs:`, `build:` (dependency).
- **Test command:** `uv run python -m pytest` (full) / `uv run python -m pytest -m "not slow"` (fast). 5 macOS-render tests fail in non-GUI shells (`CGLError`) — environmental, not a regression; run from a real Terminal for a clean full run.
- **Green-at-each-commit:** after every task, the suite must collect + pass (modulo the known env-only render failures). If a deletion breaks an unanticipated importer, fix it in the same commit.

## Decisions surfaced during planning (confirm at handoff)

1. **`dsim resume` is SB3-coupled** — it spawns `scripts.train init=resume` (SB3 init-mode semantics). The RLlib entrypoint has **no resume mechanism wired** (`train_rllib.py` never calls `tune.Tuner.restore`). This plan defaults to a **fail-fast stub** (Task 3) — `dsim resume` prints a clear "not supported on RLlib yet" message and exits non-zero — keeping Step 6 a pure retirement. Wiring real `Tuner.restore` is a recommended fast-follow. **Alternative:** wire `Tuner.restore` now (larger, separately testable).
2. **`upload_legacy_models.py`** — same legacy-SB3-model tooling class as `migrate_legacy_models.py`. This plan deletes it (Task 6). **Alternative:** keep it (it would have a dangling reference to the deleted migrator).

## Baseline facts (verified during planning)

- `scripts/train_rllib.py` imports **none** of the SB3 helpers → the rename is clean.
- `rllib/config_builder.py` reads `cfg.algo`, `cfg.multiagent`, `cfg.obs`, `cfg.reward`, `cfg.curriculum`, `cfg.league`, `cfg.seed`, and **`cfg.env.team_env_params`** — never `cfg.trainer`, `cfg.init`, or `cfg.opponent`.
- The RLlib env is `envs/quidditch/rllib_env.py:make_team_env`, registered as `"quidditch_team"` — it does **not** use `env_factories.py` or `OpponentControlledEnv`.
- `envs/quidditch/rllib_modules.py` imports `from_spec` from `opponents.py` → scripted opponents + `from_spec` **survive**.
- `rllib/eval_battery.py` imports `TERMINAL_BUCKETS, _classify_terminal` from `core/eval_core.py` → those + the dataclasses **survive**; `run_scenario` dies wholesale.

---

# PHASE 1 — Entrypoint swap (RLlib becomes canonical)

### Task 1: Establish the green baseline

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite, record the pass count**

Run: `uv run python -m pytest -q` (sandbox off)
Expected: a green baseline (note the count, e.g. "485 passed, 7 ... slow"). This is the number every later task must hold (minus deleted tests).

- [ ] **Step 2: Confirm the RLlib entrypoint launches**

Run: `uv run python -m scripts.train_rllib +experiment=rllib_league_step5 algo.total_timesteps=2000 algo.num_env_runners=0 tune.wandb.enabled=false`
Expected: Ray init + a Tune run that reaches at least one training iteration and writes a checkpoint dir under `runs/rllib_league_step5/<ts>/`. (Kill once it iterates; this just proves the path is alive before we touch anything.)

---

### Task 2: Delete SB3 `scripts/train.py`, rename `train_rllib.py` → `train.py`

**Files:**
- Delete: `scripts/train.py`
- Rename: `scripts/train_rllib.py` → `scripts/train.py` (`git mv`)
- Modify: `Makefile` (drop `train-rllib` target + `.PHONY`)
- Modify: `tests/scripts/test_train_rllib_artifact.py` (re-point imports/patches)
- Delete: `tests/scripts/test_train_resolve_parent_wiring.py`, `tests/scripts/test_train_resume_excludes_self.py`, `tests/scripts/test_train_smoke_wandb_disabled.py`, `tests/scripts/test_train_preflight_guard.py`

- [ ] **Step 1: Find every reference to the old module name**

Run: `grep -rnE "train_rllib|scripts\.train\b|-m scripts.train" --include="*.py" --include="Makefile" --include="*.yaml" . | grep -v ".venv"`
Expected: a list to reconcile (Makefile, dsim/commands/resume.py, the rllib-artifact test). Note them.

- [ ] **Step 2: Delete the SB3 entrypoint and rename the RLlib one**

```bash
git rm scripts/train.py
git mv scripts/train_rllib.py scripts/train.py
```

- [ ] **Step 3: Delete the SB3-train tests**

```bash
git rm tests/scripts/test_train_resolve_parent_wiring.py \
       tests/scripts/test_train_resume_excludes_self.py \
       tests/scripts/test_train_smoke_wandb_disabled.py \
       tests/scripts/test_train_preflight_guard.py
```
(These import `scripts.train._build_or_load_model` / `scripts.train.PPO` / subprocess SB3 `scripts.train` with `trainer.total_timesteps`/`final_model.zip` — all gone. RLlib smoke is covered by `tests/rllib/test_smoke_train.py` + the ported artifact test below.)

- [ ] **Step 4: Re-point the RLlib artifact test to the new module name**

In `tests/scripts/test_train_rllib_artifact.py`, replace every `scripts.train_rllib` with `scripts.train`:
- `from scripts.train_rllib import _log_best_checkpoint` → `from scripts.train import _log_best_checkpoint` (2 occurrences)
- `patch("scripts.train_rllib.wandb")` → `patch("scripts.train.wandb")`
- `patch("scripts.train_rllib.log_rllib_run_artifact")` → `patch("scripts.train.log_rllib_run_artifact")` (2 occurrences)

- [ ] **Step 5: Drop the `train-rllib` Makefile target**

In `Makefile`: remove `train-rllib` from the `.PHONY` line (line 28) and delete the `train-rllib:` target (lines 69-71). Leave the `train:` target unchanged — it already invokes `-m scripts.train`, which is now the RLlib entrypoint.

- [ ] **Step 6: Run tests + commit**

Run: `uv run python -m pytest -q`
Expected: green (4 fewer tests than baseline).
```bash
git add -A && git commit -S -m "refactor(train): rename train_rllib -> train, delete SB3 entrypoint"
```

---

### Task 3: Rework `dsim resume` off the SB3 path (fail-fast stub)

**Files:**
- Modify: `dsim/commands/resume.py`
- Modify: `tests/dsim/test_resume_cli.py`

- [ ] **Step 1: Replace the SB3 spawn with a fail-fast message**

In `dsim/commands/resume.py`, replace the `cmd = [...]` block + the spawn (the lines building `["...","-m","scripts.train","init=resume",...]` and `subprocess.call(cmd)`) with:

```python
    typer.echo(
        f"error: `dsim resume` is not yet supported on the RLlib path.\n"
        f"  RLlib runs resume via ray.tune Tuner.restore, which is not wired\n"
        f"  into scripts/train.py yet (tracked as a follow-up). Start a fresh\n"
        f"  run with:  make train EXP={parent_exp}",
        err=True,
    )
    raise typer.Exit(code=2)
```
Keep the upstream run-name → experiment inference (it still produces a useful message). Remove the now-unused `subprocess`/`sys` imports only if nothing else in the file uses them.

- [ ] **Step 2: Port the resume CLI test**

In `tests/dsim/test_resume_cli.py`: replace the assertion that checks the spawned command array with an assertion that the command exits non-zero and the stderr contains `not yet supported on the RLlib path`. (Use the Typer `CliRunner`/`result.exit_code == 2` pattern already in the file.)

- [ ] **Step 3: Run tests + commit**

Run: `uv run python -m pytest tests/dsim/test_resume_cli.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(dsim): resume fails fast on RLlib (Tuner.restore not yet wired)"
```

---

# PHASE 2 — Delete SB3 leaf modules

### Task 4: Delete SB3 callbacks, train-common, and W&B bridges

**Files:**
- Delete: `scripts/callbacks.py`, `scripts/_train_common.py`, `scripts/_wandb_logger.py`, `scripts/_wandb_init.py`
- Delete tests: `tests/scripts/test_train_common_video.py`, `tests/scripts/test_success_rate_eval_callback.py`, `tests/scripts/test_video_callback_team.py`, `tests/scripts/test_video_callback_wandb.py`, `tests/scripts/test_wandb_logger.py`, `tests/scripts/test_wandb_init.py`, `tests/test_meta_yaml.py`, `tests/envs/quidditch/test_obs_compat.py`
- Modify: `tests/envs/quidditch/test_yaml_obs_loader.py` (drop 3 `read_obs_spec` tests)

- [ ] **Step 1: Confirm these modules have no surviving (non-test) importer**

Run: `grep -rnE "scripts\.(callbacks|_train_common|_wandb_logger|_wandb_init)|from scripts import (callbacks|_train_common)" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: empty (the only source importer was `scripts/train.py`, already deleted). If anything appears, stop and reconcile.

- [ ] **Step 2: Delete the source modules**

```bash
git rm scripts/callbacks.py scripts/_train_common.py scripts/_wandb_logger.py scripts/_wandb_init.py
```

- [ ] **Step 3: Delete the dependent tests**

```bash
git rm tests/scripts/test_train_common_video.py \
       tests/scripts/test_success_rate_eval_callback.py \
       tests/scripts/test_video_callback_team.py \
       tests/scripts/test_video_callback_wandb.py \
       tests/scripts/test_wandb_logger.py \
       tests/scripts/test_wandb_init.py \
       tests/test_meta_yaml.py \
       tests/envs/quidditch/test_obs_compat.py
```
(`test_meta_yaml` imports `_train_common.write_meta_yaml`/`append_meta_yaml_final_stats`/`read_parent_chain_total_from_hydra`; `test_obs_compat` imports `_train_common.check_obs_compat`/`format_obs_block` + `scripts.train._check_obs_compat_from_hydra`. Compat for the live path is covered by `core.obs_compat.preflight` + `tests/core/test_obs_compat_preflight.py`.)

- [ ] **Step 4: Port the YAML obs-loader test**

In `tests/envs/quidditch/test_yaml_obs_loader.py`, delete the 3 tests that import `scripts._train_common.read_obs_spec`: `test_read_obs_spec_translates_legacy_opp_vel_rel_body_mixed`, `test_read_obs_spec_translates_legacy_vec_to_hoop_world`, `test_read_obs_spec_passthrough_for_non_renamed_names`. Remove the now-unused `read_obs_spec` import. Keep all `obs_spec`/`ObsConfig` tests.

- [ ] **Step 5: Run tests + commit**

Run: `uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(scripts): delete SB3 callbacks, train-common, and W&B bridges"
```

---

### Task 5: Delete `warm_start.py`; simplify `InitConfig` to scratch-only

**Files:**
- Delete: `core/policies/warm_start.py`
- Delete configs: `conf/init/pretrain.yaml`, `conf/init/resume.yaml`, `conf/init/warm_start.yaml`
- Modify: `config_schema.py` (`InitConfig`), `conf/init/scratch.yaml`, `Makefile` (drop `test-warm`)
- Delete tests: `tests/core/policies/test_warm_start_by_spec.py`
- Modify tests: `tests/test_init_latest_ban.py`, `tests/test_config_loading.py`

- [ ] **Step 1: Confirm `warm_start_ppo_by_spec` has no surviving importer**

Run: `grep -rnE "warm_start_ppo_by_spec|core\.policies\.warm_start" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: empty (only `scripts/train.py` imported it). Reconcile if not.

- [ ] **Step 2: Delete warm_start source + non-scratch init configs**

```bash
git rm core/policies/warm_start.py conf/init/pretrain.yaml conf/init/resume.yaml conf/init/warm_start.yaml
```

- [ ] **Step 3: Simplify `InitConfig` (scratch-only)**

In `config_schema.py`, replace the `InitConfig` dataclass (lines 52-82) with:

```python
@dataclass
class InitConfig:
    """Pure-scratch league: training always starts from random init.

    (Pre-RLlib SB3 modes — pretrain/resume/warm_start, parent loading, and the
    `:latest`-alias ban — were retired with the SB3 path in migration Step 6.)
    """
    mode: str = "scratch"
```

- [ ] **Step 4: Slim `conf/init/scratch.yaml`**

Replace its contents with just:
```yaml
mode: scratch
```

- [ ] **Step 5: Delete the warm-start test + drop the SB3 Makefile target**

```bash
git rm tests/core/policies/test_warm_start_by_spec.py
```
In `Makefile`: remove `test-warm` from `.PHONY` (line 28) and delete the `test-warm:` target (lines 58-60).

- [ ] **Step 6: Port the init-mode tests**

In `tests/test_init_latest_ban.py`: keep `test_latest_alias_ok_when_mode_scratch` (rewrite it to just construct `InitConfig(mode="scratch")` and assert it succeeds — there is no `parent` field now). Delete `test_latest_alias_rejected_when_mode_pretrain`, `test_latest_alias_rejected_when_mode_warm_start`, `test_stable_alias_accepted`, `test_fully_qualified_uri_latest_also_rejected` (they construct removed non-scratch modes). If the file is left with a single trivial assertion, that is acceptable.

In `tests/test_config_loading.py`: in `test_init_groups_compose` (the `@pytest.mark.parametrize("init_choice", [...])`), drop the `"pretrain"`, `"resume"`, `"warm_start"` cases — leave only `"scratch"` (collapse the parametrize to a single non-parametrized assertion if cleaner). Remove the `if init_choice in ("pretrain", "warm_start"):` parent-injection branch.

- [ ] **Step 7: Run tests + commit**

Run: `uv run python -m pytest tests/test_init_latest_ban.py tests/test_config_loading.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(config): delete warm_start, simplify InitConfig to scratch-only"
```

---

### Task 6: Delete the legacy-model migration/upload tooling

**Files:**
- Delete: `scripts/migrate_legacy_models.py`, `scripts/upload_legacy_models.py`
- Delete tests: `tests/test_migrate_legacy_models.py`, `tests/scripts/test_upload_legacy_models.py`

- [ ] **Step 1: Confirm no surviving importer**

Run: `grep -rnE "migrate_legacy_models|upload_legacy_models|migrate_one|upload_one" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: only string mentions inside the two scripts themselves (which are being deleted). Reconcile any real importer.

- [ ] **Step 2: Delete sources + tests**

```bash
git rm scripts/migrate_legacy_models.py scripts/upload_legacy_models.py \
       tests/test_migrate_legacy_models.py tests/scripts/test_upload_legacy_models.py
```

- [ ] **Step 3: Run tests + commit**

Run: `uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(scripts): delete legacy SB3-model migrate/upload tooling"
```

---

### Task 7: Delete the SB3 eval scripts (`eval_solo`, `eval_battery`)

**Files:**
- Delete: `scripts/eval_solo.py`, `scripts/eval_battery.py`
- Delete configs: `conf/eval_solo/` (if present), `conf/eval_battery/` (if present)
- Delete tests: `tests/scripts/test_eval_solo_hydra.py`, `tests/scripts/test_eval_battery.py`

- [ ] **Step 1: Confirm coverage by the RLlib eval path**

Run: `grep -rnE "scripts\.eval_solo|scripts\.eval_battery" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: empty. (The live battery is `rllib/eval_battery.py`; `scripts/eval_team.py` keeps its RLlib branch — reworked in Task 9.)

- [ ] **Step 2: Delete the scripts + their Hydra config groups**

```bash
git rm scripts/eval_solo.py scripts/eval_battery.py
git rm -r conf/eval_solo conf/eval_battery 2>/dev/null || true
```
(`eval_solo.py` is SB3-only; `eval_battery.py` routes through the dying `core.eval_core.run_scenario`.)

- [ ] **Step 3: Delete their tests**

```bash
git rm tests/scripts/test_eval_solo_hydra.py tests/scripts/test_eval_battery.py
```

- [ ] **Step 4: Run tests + commit**

Run: `uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(eval): delete SB3 eval_solo + eval_battery scripts"
```

---

# PHASE 3 — Split the shared modules

### Task 8: Delete `env_factories.py`; split `conf/env/team.yaml`

**Files:**
- Delete: `envs/quidditch/env_factories.py`, `conf/env/simple.yaml`
- Modify: `conf/env/team.yaml` (keep `team_env_params`, drop the SB3 factory header)
- Delete tests: `tests/envs/quidditch/test_env_factories.py`, `tests/envs/quidditch/test_env_factories_obs_blocks.py`
- Modify test: `tests/envs/quidditch/test_team_env_v3.py` (drop the factory test)

- [ ] **Step 1: Confirm RLlib does not consume env_factories**

Run: `grep -rnE "env_factories|SimpleEnvFactory|TeamEnvFactory" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: empty (only `scripts/train.py`, already deleted, used them). The RLlib env is `make_team_env`.

- [ ] **Step 2: Delete the factory module + the simple env config**

```bash
git rm envs/quidditch/env_factories.py conf/env/simple.yaml
```

- [ ] **Step 3: Slim `conf/env/team.yaml` to just the RLlib-consumed params**

Replace its contents with (the `_target_`/`n_envs`/`team_cfg`/`learner_id`/`opponent_spec`/`obs_*`/`frame_stack`/`seed` factory fields are gone; `config_builder._team_cfg_from` reads only `team_env_params`):

```yaml
# Physics thresholds for QuidditchTeamEnv. Read by rllib/config_builder.py
# (_team_cfg_from -> TeamConfig) and shared with reward terms via interpolation
# in conf/reward/team_v2.yaml.
team_env_params:
  red_prefix:      red_0
  blue_prefix:     blue_0
  hoop_prefix:     hoop_0
  midpoint_alpha:  0.3
  tag_radius:      0.3
  tag_cooldown_s:  1.0
  crash_vel_thr:   1.0
  walls_collide:   true
```

- [ ] **Step 4: Delete the factory tests + drop the factory test in test_team_env_v3**

```bash
git rm tests/envs/quidditch/test_env_factories.py tests/envs/quidditch/test_env_factories_obs_blocks.py
```
In `tests/envs/quidditch/test_team_env_v3.py`: delete `test_subproc_vec_env_does_not_lose_spec_identity_on_pickling` (uses `TeamEnvFactory`) and remove its `from envs.quidditch.env_factories import TeamEnvFactory` import. (The OCE/PPO tests in this file are handled in Task 10.)

- [ ] **Step 5: Verify the RLlib env still composes + run tests**

Run: `uv run python -m scripts.train +experiment=rllib_league_step5 algo.total_timesteps=2000 algo.num_env_runners=0 tune.wandb.enabled=false`
Expected: still reaches a training iteration (proves `cfg.env.team_env_params` still resolves after the slim). Then:
Run: `uv run python -m pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -S -m "refactor(env): delete SB3 env_factories, slim conf/env/team.yaml to team_env_params"
```

---

### Task 9: Delete `core.eval_core.run_scenario`; rework `eval_team.py`

**Files:**
- Modify: `core/eval_core.py` (delete `run_scenario` + orphaned imports; keep dataclasses + `TERMINAL_BUCKETS` + `_classify_terminal`)
- Modify: `scripts/eval_team.py` (keep RLlib branch, drop SB3 fallback)
- Modify test: `tests/core/test_eval_core.py` (drop the `run_scenario` test)
- Delete test: `tests/scripts/test_eval_team_hydra.py`

- [ ] **Step 1: Delete `run_scenario` and reduce eval_core imports**

In `core/eval_core.py`: delete the entire `run_scenario` function (lines 76-246, including its lazy `OpponentControlledEnv`/`FrameStackWrapper`/`from_spec`/`PPO` imports). Reduce the top-of-file imports (lines 25-30) to just:

```python
from dataclasses import dataclass
```
(`Counter`, `Path`, `Callable`, `Literal`, and `numpy` were only used inside `run_scenario`. Keep `dataclass` for the three `@dataclass(frozen=True)`.) Reword the module docstring to describe only the surviving dataclasses + `_classify_terminal`. **Keep:** `TERMINAL_BUCKETS`, `ScenarioSpec`, `EpisodeResult`, `ScenarioResult`, `_classify_terminal` (live consumer: `rllib/eval_battery.py`).

- [ ] **Step 2: Strip the SB3 fallback from `eval_team.py`**

In `scripts/eval_team.py`: the file detects an RLlib checkpoint (via `core.rllib_checkpoint.is_rllib_checkpoint`) and dispatches to `run_rllib_battery`, then `return`s; the trailing `run_scenario(...)` + `_print_summary(result)` is the SB3 fallback. Remove that fallback path: drop the `from core.eval_core import ... run_scenario` import (keep `ScenarioResult`/`ScenarioSpec` only if still referenced; otherwise drop them too), and replace the post-`return` fallback with a clear error for a non-RLlib URI:

```python
    raise SystemExit(
        f"eval_team: {learner_uri!r} is not an RLlib checkpoint. "
        f"The SB3 eval path was retired in migration Step 6."
    )
```

- [ ] **Step 3: Port the eval_core test; delete the eval_team Hydra test**

In `tests/core/test_eval_core.py`: delete `test_run_scenario_produces_episode_results` (calls `run_scenario`). Keep the `_classify_terminal` tests + `test_scenario_spec_is_immutable`.
```bash
git rm tests/scripts/test_eval_team_hydra.py
```
(`test_eval_team_hydra` exercises the `scripted:beeline_blue` learner → the deleted `run_scenario` fallback; no RLlib checkpoint path. RLlib eval is covered by `tests/rllib/test_eval_battery.py`.)

- [ ] **Step 4: Run tests + commit**

Run: `uv run python -m pytest tests/core/test_eval_core.py tests/rllib/test_eval_battery.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(eval): delete SB3 run_scenario, eval_team is RLlib-only"
```

---

### Task 10: Split `opponents.py` — keep scripted, delete SB3/OCE wrappers

**Files:**
- Modify: `envs/quidditch/opponents.py`
- Delete tests: `tests/envs/quidditch/test_opponent_env_world.py`, `tests/envs/quidditch/test_eval_uri_resolution.py`
- Modify tests: `tests/envs/quidditch/test_augmented_obs.py`, `tests/envs/quidditch/test_team_env_v3.py`

- [ ] **Step 1: Delete the SB3/gym classes + the frozen/mixture spec branches**

In `envs/quidditch/opponents.py`, delete:
- `class FrameStackWrapper(gym.Wrapper)` (lines 20-67)
- `class FrozenPolicyOpponent` (lines 138-158)
- `class MixtureOpponent` (lines 161-180)
- the `mixture:` branch (lines 210-219) and the `frozen:` branch (lines 221-227) inside `from_spec` (the lazy `from scripts._artifact_io import resolve_parent` lives inside the `frozen:` branch and goes with it)
- `class OpponentControlledEnv(gym.Env)` (lines 247-322)

Reduce the import block (lines 9-17) to:
```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
```
(`gym`, `from stable_baselines3 import PPO`, and `from envs.quidditch.team_env import QuidditchTeamEnv` were only used by the deleted classes. `numpy` stays.) Trim the module docstring + the `from_spec` docstring's `frozen:`/`mixture:` examples to scripted-only.

- [ ] **Step 2: Verify the surviving `from_spec` body**

After removing the two branches, `from_spec` keeps the scripted path only:
```python
def from_spec(spec: str, *, deterministic: bool = False) -> Opponent:
    """Build a scripted Opponent from a spec string (e.g. "beeline_blue",
    "intercepter_blue:lookahead=0.5", "zero")."""
    spec = spec.strip()
    if not spec:
        raise ValueError("from_spec: empty spec")
    if ":" in spec:
        name, kv_str = spec.split(":", 1)
        kwargs: dict[str, float] = {}
        for kv in kv_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=", 1)
            kwargs[k.strip()] = float(v)
    else:
        name, kwargs = spec, {}
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"from_spec: unknown opponent {name!r}")
    return cls(**kwargs)  # type: ignore[arg-type]
```
**Keep:** `Opponent`, `ZeroOpponent`, `_beeline_act`, `BeelineRed`, `BeelineBlue`, `IntercepterBlue`, `_REGISTRY`, `from_spec`.

- [ ] **Step 3: Confirm `rllib_modules` still imports cleanly**

Run: `uv run python -c "from envs.quidditch.rllib_modules import ScriptedRLModule; from envs.quidditch.opponents import from_spec; print(from_spec('beeline_blue'))"`
Expected: prints a `BeelineBlue` instance — proves the live RLlib scripted path survives the split.

- [ ] **Step 4: Delete the OCE/frozen tests**

```bash
git rm tests/envs/quidditch/test_opponent_env_world.py tests/envs/quidditch/test_eval_uri_resolution.py
```
(`test_eval_uri_resolution` exercises the `frozen:` branch + `FrozenPolicyOpponent`; `test_opponent_env_world` tests OCE + FrameStackWrapper.)

- [ ] **Step 5: Port `test_augmented_obs.py` to use `team_env` directly**

In `tests/envs/quidditch/test_augmented_obs.py`:
- Delete `test_frame_stack_wrapper_doubles_obs_dim` (FrameStackWrapper gone).
- Drop the `FrameStackWrapper, OpponentControlledEnv, from_spec` imports.
- Replace the `_make_blue_env()` helper (which wrapped in `OpponentControlledEnv`) with a direct team env:
  ```python
  def _make_blue_env() -> QuidditchTeamEnv:
      return QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False),
                              learner_id="blue_0", learner_spec=DUEL_V2_WORLD)
  ```
  (mirror the construction in `test_team_env_v3.py`; import `QuidditchTeamEnv`, `TeamConfig`, and the `DUEL_V2_WORLD` spec the file already references).
- In the kept tests (`test_v2_obs_shape_is_25_dim`, `test_v2_vec_to_hoop_slot_points_to_hoop`, `test_v2_closing_rate_zero_at_static_positive_when_closing`, `test_v2_opp_vel_rel_uses_world_frame`), change `.step(np.zeros(4))` to the team-env dict form `.step({"blue_0": np.zeros(4, np.float32), "red_0": np.zeros(4, np.float32)})` and read `obs["blue_0"]` instead of the OCE single-obs return.

- [ ] **Step 6: Drop the OCE tests in `test_team_env_v3.py`**

In `tests/envs/quidditch/test_team_env_v3.py`: delete `test_oce_passes_through_v3_blue_obs_without_re_augmentation` (OCE) and `test_blue_v4_round_trip_through_new_in_env_packer` (SB3 `PPO` + OCE + FrameStackWrapper), and remove the `from envs.quidditch.opponents import OpponentControlledEnv, from_spec` and any `FrameStackWrapper` import. Keep the 7 `QuidditchTeamEnv`-direct tests.

- [ ] **Step 7: Run tests + commit**

Run: `uv run python -m pytest tests/envs/quidditch/test_augmented_obs.py tests/envs/quidditch/test_team_env_v3.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(opponents): keep scripted core, delete OCE/FrameStack/frozen SB3 wrappers"
```

---

# PHASE 4 — Configs, schema, CLI

### Task 11: Delete the `trainer` config group + `TrainerConfig`

**Files:**
- Delete: `conf/trainer/ppo.yaml`, `conf/trainer/ppo_finetune.yaml` (whole `conf/trainer/` dir)
- Modify: `config_schema.py` (remove `TrainerConfig` + its store + `Config.trainer`), `conf/config.yaml` (drop `trainer: ppo` default)
- Modify tests: `tests/test_config_schema.py`, `tests/test_config_loading.py`

- [ ] **Step 1: Confirm nothing reads `cfg.trainer`**

Run: `grep -rnE "cfg\.trainer|\.trainer\.|TrainerConfig|group=\"trainer\"" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: only `config_schema.py` (the definition + store). The renderer (`_render_model_doc.py`) is handled in Task 15. Reconcile anything else.

- [ ] **Step 2: Delete the trainer config group**

```bash
git rm -r conf/trainer
```

- [ ] **Step 3: Remove `TrainerConfig` from the schema**

In `config_schema.py`: delete the `TrainerConfig` dataclass (lines 19-30), delete `cs.store(group="trainer", name="schema", node=TrainerConfig)` (line 161), and delete the `trainer: TrainerConfig = field(default_factory=TrainerConfig)` line from `Config` (line 143).

- [ ] **Step 4: Drop the `trainer` default from `conf/config.yaml`**

In `conf/config.yaml`: delete the `- trainer: ppo` line (line 5). While here, update the now-stale "ignored by the SB3 trainer" comments on the `algo`/`multiagent`/`tune` defaults (lines 14-16) to note they are the canonical RLlib path.

- [ ] **Step 5: Port the config tests**

In `tests/test_config_schema.py`: delete `test_trainer_config_defaults` and (if present) `test_init_config_mode_values` (or rewrite the latter to `InitConfig(mode="scratch")`). Keep `test_register_configs_runs_without_error`, `test_top_level_config_has_description_field`, `test_curriculum_schema_has_difficulty_levers_and_schedules`.

In `tests/test_config_loading.py`: in `test_default_compose_succeeds`, replace any `cfg.trainer.*` / `cfg.env._target_.endswith("Factory")` assertions with RLlib-shape ones (e.g. `assert cfg.algo.lr > 0`, `assert cfg.env.team_env_params.tag_radius == 0.3`). In `test_experiment_composes`, change `cfg.trainer.total_timesteps` → `cfg.algo.total_timesteps`. Keep `test_rllib_league_step5_experiment_composes`.

- [ ] **Step 6: Run tests + commit**

Run: `uv run python -m pytest tests/test_config_schema.py tests/test_config_loading.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(config): delete trainer group + TrainerConfig (SB3 PPO schema)"
```

---

### Task 12: Delete the SB3-only opponent configs

**Files:**
- Delete: `conf/opponent/frozen.yaml`, `conf/opponent/mixture.yaml`

- [ ] **Step 1: Confirm they are unreferenced**

Run: `grep -rnE "opponent.*frozen|opponent.*mixture|frozen\.yaml|mixture\.yaml" --include="*.yaml" --include="*.py" conf/ scripts/ rllib/ dsim/ | grep -v ".venv"`
Expected: nothing depends on them (the league never composes `frozen:`/`mixture:`). The kept scripted opponent yamls (`beeline_*`, `intercepter_blue`, `none`) still resolve via the surviving scripted classes; the `opponent: beeline_red` default in `config.yaml` is now vestigial for RLlib but harmless — leave it.

- [ ] **Step 2: Delete + test + commit**

```bash
git rm conf/opponent/frozen.yaml conf/opponent/mixture.yaml
uv run python -m pytest -q   # expect green
git add -A && git commit -S -m "refactor(config): delete frozen/mixture opponent configs (PPO .zip refs)"
```

---

### Task 13: Retire `dsim sweep`

**Files:**
- Delete: `dsim/commands/sweep.py`
- Modify: `dsim/cli.py` (unregister the `sweep` sub-app)
- Delete tests: any `tests/dsim/test_sweep*.py`

- [ ] **Step 1: Locate the sweep wiring**

Run: `grep -rnE "sweep" dsim/cli.py dsim/commands/ tests/dsim/ | grep -v ".venv"`
Expected: the `sweep` import + `add_typer`/command registration in `dsim/cli.py`, the `dsim/commands/sweep.py` module, and its test(s).

- [ ] **Step 2: Delete the module + unregister it**

```bash
git rm dsim/commands/sweep.py
git rm tests/dsim/test_sweep_cli.py 2>/dev/null || true   # delete whatever sweep test exists
```
In `dsim/cli.py`: remove the `sweep` import and the line that registers it (e.g. `app.add_typer(sweep.app, name="sweep")` or the `@app.command` wiring). 

- [ ] **Step 3: Verify the CLI still loads + test + commit**

Run: `uv run python -m dsim --help`
Expected: the help text lists the surviving commands and no longer shows `sweep`.
Run: `uv run python -m pytest tests/dsim -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(dsim): retire sweep command (superseded by Ray Tune)"
```

---

# PHASE 5 — Dependency, docs, MODEL.md

### Task 14: Remove the `stable-baselines3` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Assert no SB3 imports remain in source**

Run: `grep -rnE "stable_baselines3|sb3_contrib" --include="*.py" . | grep -vE ".venv|/tests/"`
Expected: **empty**. If anything remains, stop and remove it before touching the dependency.

- [ ] **Step 2: Drop the dependency + sync**

In `pyproject.toml`: delete the `"stable-baselines3",` line from `dependencies` (line 8).
Run: `uv sync` (sandbox off)
Expected: the lockfile updates and SB3 is uninstalled from `.venv`.

- [ ] **Step 3: Full suite + commit**

Run: `uv run python -m pytest -q`
Expected: green (proves nothing imported SB3 transitively).
```bash
git add -A && git commit -S -m "build: drop stable-baselines3 dependency"
```

---

### Task 15: Adapt MODEL.md rendering to the RLlib config shape

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify test: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Find where the renderer reads SB3 config shape**

Run: `grep -nE "trainer|cfg\.env|_target_|SimpleEnvFactory|TeamEnvFactory|total_timesteps" scripts/_render_model_doc.py`
Expected: the `_section_hyperparams` / `_section_env_config` helpers reading `cfg.trainer.*` and the team `cfg.env` factory shape. If the renderer reads them defensively (`cfg.get("trainer")`), MODEL.md just renders blank for RLlib runs; adapt it to the live shape.

- [ ] **Step 2: Point the hyperparams + env sections at the RLlib shape**

In `scripts/_render_model_doc.py`: change `_section_hyperparams` to read `cfg.algo` (`lr`, `gamma`, `lambda_`, `clip_param`, `entropy_coeff`, `num_epochs`, `minibatch_size`, `train_batch_size_per_learner`, `total_timesteps`) instead of `cfg.trainer`. Change `_section_env_config` to read `cfg.env.team_env_params` (the slimmed team config) instead of the factory `_target_`/fields.

- [ ] **Step 3: Rework the two renderer tests**

In `tests/scripts/test_render_model_doc.py`: rewrite `test_section_hyperparams_renders_trainer_fields` to set `cfg.algo.*` and assert those render; rewrite `test_section_env_config_renders_team_fields` to set `cfg.env.team_env_params.*`. Keep the other renderer tests.

- [ ] **Step 4: Run tests + commit**

Run: `uv run python -m pytest tests/scripts/test_render_model_doc.py -q && uv run python -m pytest -q`
Expected: green.
```bash
git add -A && git commit -S -m "refactor(model-doc): render hyperparams/env from RLlib cfg.algo + team_env_params"
```

---

### Task 16: Update the docs (README + repo CLAUDE.md)

**Files:**
- Modify: `README.md`, `CLAUDE.md` (repo root — the AI-orientation one)

- [ ] **Step 1: Update the CLI-surface tables in `README.md`**

Remove/repoint the retired surfaces:
- Hydra apps row: drop `eval_solo` and `eval_battery` examples; keep `train`, `eval_team`.
- `dsim` row: drop `dsim sweep create <name>`; note `dsim resume` is RLlib-stubbed.
- `make` row: drop `make test-warm` and the `make train-rllib`/`sweep` lines; `make train EXP=` now drives RLlib.
- The macOS-OpenMP note (line ~137): replace the `SubprocVecEnv`/SB3 wording — `KMP_DUPLICATE_LIB_OK=TRUE` now reaches Ray env-runners via `ray.init(runtime_env=...)` (see `rllib/runtime.py`).

- [ ] **Step 2: Update `CLAUDE.md` (repo) CLI surface**

In the "Hydra apps" bullet, change `train, eval_team, eval_solo, eval_battery` → `train, eval_team`. In the `dsim` bullet, drop `sweep` from the dispatch list. In the `make` bullet, drop `train-rllib`/`test-warm`.

- [ ] **Step 3: Sanity-check + commit**

Run: `grep -rnE "eval_solo|eval_battery|train-rllib|test-warm|dsim sweep|SubprocVecEnv" README.md CLAUDE.md`
Expected: no stale references remain.
```bash
git add -A && git commit -S -m "docs: update CLI surface for RLlib-only training path"
```

---

# PHASE 6 — Final verification

### Task 17: Whole-system verification

**Files:** none (verification + optional final commit).

- [ ] **Step 1: Zero-SB3 assertion across the whole tree**

Run: `grep -rnE "stable_baselines3|sb3_contrib|OpponentControlledEnv|FrameStackWrapper|FrozenPolicyOpponent|MixtureOpponent|run_scenario|env_factories|warm_start|migrate_legacy|_wandb_init|train_rllib" --include="*.py" --include="*.yaml" . | grep -vE ".venv|/docs/|brain/"`
Expected: empty (or only historical mentions in `docs/`/specs). Any hit in live code/config is a miss — fix it.

- [ ] **Step 2: Full test suite (from a GUI Terminal)**

Run: `uv run python -m pytest -q`
Expected: green. Render-only macOS failures (`CGLError`) are acceptable iff run in a non-GUI shell; from a real Terminal they pass.

- [ ] **Step 3: RLlib smoke run end-to-end**

Run: `uv run python -m scripts.train +experiment=rllib_league_step5 algo.total_timesteps=20000 tune.wandb.enabled=false`
Expected: completes a few iterations and writes a checkpoint dir under `runs/rllib_league_step5/<ts>/`.

- [ ] **Step 4: `dsim` surfaces still work**

Run: `uv run python -m dsim list-runs` then `uv run python -m scripts.eval_team +eval_team=default +learner=red learner.uri=<the run's checkpoint dir>` (and `dsim describe-run <run>`).
Expected: `list-runs` finds the smoke run; `eval_team` loads the RLlib checkpoint and runs the battery; `describe-run` renders.

- [ ] **Step 5: Update the brain (out-of-repo) + finish the branch**

Update `brain/index.md` (Current State + Active Priorities #1: Step 6 done → migration complete), `brain/changelog.md`, `brain/tasks.md` (check off Step 6), and note the RLlib-resume follow-up in Known Issues. Then use the **superpowers:finishing-a-development-branch** skill to merge (`--no-ff`) into `develop`.

---

## Self-review (completed by author)

- **Spec coverage:** every spec delete/split/keep item maps to a task — leaf deletions (Tasks 2,4,5,6,7), splits (Tasks 8,9,10), configs/CLI (Tasks 11,12,13), entrypoint rename + repoint (Tasks 2,3), dependency (Task 14), docs (Task 16). Discovered necessary consequences added: InitConfig simplification (5), MODEL.md adaptation (15), `dsim resume` rework (3), `team_env_params` relocation (8), `config.yaml` defaults (11).
- **Placeholder scan:** no TBD/TODO; every edit names exact files + line ranges + shows resulting content for non-trivial changes; test ports name exact test-function names.
- **Type/name consistency:** `from_spec`, `InitConfig`, `team_env_params`, `_team_cfg_from`, `make_team_env`, `run_rllib_battery`, `_classify_terminal`, `TERMINAL_BUCKETS` are used consistently with the verified source.
- **Two open decisions** (resume stub vs wire `Tuner.restore`; deleting `upload_legacy_models.py`) are flagged at top + in the handoff.
