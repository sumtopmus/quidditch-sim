# Team-resume — Design

**Goal.** Add `--resume` to `scripts/train_team_ppo.py` so a paused or finished team-play training run can be continued from a checkpoint, with a configurable learning rate.

**Why now.** `ppo_hoop_blue_1/20260507_194423` finished its 10M-step config-default budget against `frozen:models/ppo_hoop_red_1_20260506_103058/best_model`. We want to extend training with `lr=1e-4` (lowered from 3e-4) for another ~20M steps. The single-agent `train_ppo.py` already supports `--resume`; the team script does not, and `make resume` is hardcoded to the single-agent script.

**Non-goals.**
- Re-warm-starting from a single-agent checkpoint with input-layer surgery (that's `--warm-start`, already exists, untouched here).
- Changing TB log continuity behavior — we keep the existing single-agent pattern of "new trial dir per resume" so lineage tools work.
- Resuming with a different opponent or a different `--learner`. Possible in principle (the policy's I/O contract doesn't depend on opponent identity), but out of scope; both should be re-specified on the CLI/config so the resumed trial's `info.toml` records them honestly.

## Design Decisions

### 1. New trial dir on resume (match single-agent)

`train_ppo.py` resume creates a *new* timestamped trial dir under the same `run_name` and writes a `[resume]` block in its `info.toml` pointing to the source checkpoint. This means TB curves split across `runs/<run_name>/<trial_a>/PPO_1` and `runs/<run_name>/<trial_b>/PPO_1`, but the `num_timesteps` X-axis is preserved across the split so the curves visually continue when TB is pointed at the `run_name` parent.

**Decision:** match this pattern exactly. Trade-off accepted: minor visual seam in TB; benefit: lineage walker (`scripts/lineage.py`) works without changes, and `evaluations.npz` per-trial isolation is preserved.

### 2. Hyperparameter override on resume — `learning_rate` only

`train_ppo.py:373-379` passes the checkpoint through `PPO.load(...)` without overriding any hyperparameters — the saved `learning_rate` schedule wins. This is arguably a latent bug (the user has reported wanting to lower lr on resume and finding it doesn't take effect).

**Decision:** override `learning_rate` from config on resume. Leave the other PPO hyperparameters (`n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`) at their checkpoint values.

**Why only lr.** Changing `gamma` mid-training breaks the value function's discount assumption. Changing `n_steps` mid-training reallocates the rollout buffer, which works mechanically but invalidates any previously-tuned KL/clip behavior. `learning_rate` is the conventional "decay-on-resume" knob and is the documented user need. Other knobs require a fresh `--warm-start`-style trial.

**CLI surface:** `--lr` (already present in `parse_args`) wins over the config value when passed. So the precedence is: `--lr` (CLI) > `[training.ppo].lr` (config). This mirrors the existing CLI hierarchy.

### 3. `--resume` and `--warm-start` are mutually exclusive

`train_ppo.py` enforces this implicitly via an `if/elif/else` chain. We do the same here, but make it explicit with `argparse`'s mutual-exclusion guard so a user passing both gets a clear error rather than silent priority.

### 4. `total_timesteps` is the absolute target

SB3 PPO's `model.learn(total_timesteps=N, reset_num_timesteps=False)` trains until `model.num_timesteps == N`, not "for N more steps". This matches `train_ppo.py:464-469`. We keep this convention so:

- `make resume-team RUN_NAME=ppo_hoop_blue_1` continues to whatever `total_timesteps` is set in the current `config/training.toml`.
- The user's mental model "I bumped `total_timesteps` to 30M, that's the new target" works.
- `ResumeProgressCallback` (already exists at `scripts/callbacks.py:15-42`) renders the bar correctly because it reads `self.model.num_timesteps` and uses `total_timesteps` as the bar max.

### 5. `info.toml` `[resume]` block

Mirror the single-agent `info.toml` shape:

```toml
[resume]
checkpoint  = "runs/ppo_hoop_blue_1/20260507_194423/checkpoints/ppo_10000000_steps.zip"
resumed_at  = 10002432
```

The team training script writes via `scripts/_train_common.write_run_info`, which currently accepts `extra=` (flat key/value emit under `[extra]`). We extend it with an optional `resume=` parameter that emits a structured `[resume]` block, matching the single-agent format.

### 6. Makefile target — `resume-team`

Mirror `make resume` exactly, but call `train_team_ppo.py`. Reuse the existing `_TRIAL_DIR`, `_LATEST_CKPT`, and `_RESUME_RUN` Make variables — they're already defined for the single-agent case.

```makefile
resume-team: ## ▶️  Resume team-play training  RUN_NAME=... [TRIAL=...] [CHECKPOINT=...]
    @ckpt="$(or $(CHECKPOINT),$(_LATEST_CKPT))"; \
     test -n "$$ckpt" || { echo "ERROR: no checkpoint found in $(_TRIAL_DIR)/checkpoints/"; exit 1; }; \
     $(PYTHON) scripts/train_team_ppo.py --learner $(LEARNER) --opponent "$(OPPONENT)" \
       --resume "$$ckpt" --run-name "$(_RESUME_RUN)"
```

The team scripts need `--learner` and `--opponent` to construct the env, but those *aren't* recoverable from the SB3 checkpoint alone — they're env-level, not policy-level. We have two options:

**Option A (chosen):** read them from the resumed trial's `config_snapshot.toml` + `info.toml [extra]` block, defaulting to whatever the caller supplied on the Make CLI.

**Option B:** require `LEARNER=...` and `OPPONENT=...` on every `make resume-team` invocation.

Option A is friendlier for the common case (no flags = same setup as the parent trial) and falls through to the user's CLI for overrides. Implementation: `train_team_ppo.py --resume <ckpt>` reads the parent trial's `info.toml [extra]` to fill `learner` and `opponent_spec` if `--learner` / `--opponent` are not given on the CLI. The Makefile target leaves them optional.

### 7. ResumeProgressCallback

Single-agent `train_ppo.py` swaps SB3's built-in progress bar for `ResumeProgressCallback` when resuming, so the bar shows absolute steps (e.g. `10.0M / 30.0M`) instead of restarting from 0. The team script doesn't currently use it. We wire it in for resume mode only — same conditional as `train_ppo.py:459-463`.

## File Inventory

| Action | Path | Why |
|---|---|---|
| Modify | `scripts/train_team_ppo.py` | add `--resume` argparse + branch in `main()` |
| Modify | `scripts/_train_common.py` | extend `write_run_info` with optional `resume=` |
| Create | `tests/unit/test_team_resume_args.py` | argparse mutual-exclusion + parsing |
| Create | `tests/integration/test_team_resume.py` | end-to-end: train tiny → checkpoint → resume |
| Modify | `Makefile` | add `resume-team` target + `.PHONY` |

No changes to `envs/quidditch/`, `core/`, or other training infrastructure. The change is strictly additive on top of the existing `train_team_ppo.py` flow.

## Failure Modes & Tests

| Failure | Test |
|---|---|
| User passes `--resume` and `--warm-start` together | unit: argparse exits with non-zero |
| User passes `--resume <bogus_path>` | integration: raises `FileNotFoundError` from `PPO.load` (SB3 default) |
| Resume preserves `num_timesteps` | integration: assert `model.num_timesteps > checkpoint_steps` after a brief `learn()` |
| Resume writes `[resume]` block | integration: parse `info.toml`, assert `data["resume"]["checkpoint"]` matches |
| `--lr` override takes effect | integration: assert `model.learning_rate(progress_remaining=1.0) == new_lr` (SB3 stores lr as a callable schedule) |
| Existing canaries still pass | run `make test` after change; `tests/integration/test_team_env_canary.py` and `tests/integration/test_warm_start.py` must stay green |

## Out of Scope (deliberately)

- TB log merging (would require copying `PPO_1/` events between trial dirs; deferred — TB renders both as siblings under `run_name/`).
- Auto-promoting the resumed `best_model.zip` (manual step, same as today).
- Updating `info.toml` `[pretrain]` chain — resume is a *continuation*, not a new lineage; `[resume]` block is the right semantic.
