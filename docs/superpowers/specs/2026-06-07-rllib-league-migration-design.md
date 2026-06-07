# SB3 → RLlib Migration + Dual-Population Self-Play League

**Date:** 2026-06-07
**Branch:** `feature/rllib-league` (worktree off develop)
**Supersedes:** the single-agent-with-frozen-opponent training model (SB3 PPO + `OpponentControlledEnv`).
**Resolves:** the "RL library: SB3 vs RLlib" open design question (brain `index.md` §Open Design Questions) and the Blue-defender learning-signal wall (brain `index.md` §Active Priorities #1).

## Context

The 1v1 stack trains one policy at a time against a *frozen* opponent. `QuidditchTeamEnv` is already a PettingZoo `ParallelEnv` (both `red_0` and `blue_0` step simultaneously), but training collapses it to single-agent via `OpponentControlledEnv`, which drives the non-learner from a frozen/scripted `Opponent` and hands SB3 PPO only the learner's slice. Self-play is therefore *sequential and manual*: train Red vs scripted Blue, freeze it, train Blue vs frozen Red, freeze, repeat — the hand-managed ladder `R1s→R2s→R3t→R4t`.

This model hit a wall. The Blue defender stalls at a **~25% learning-signal wall** against the frozen `R4t` Red — not because R4t is evasive (it was trained only to score vs a *hovering* Blue, so it is non-reactive) but because pure-R4t episodes are almost all losses, yielding no gradient. A hand-built `beeline+R4t` mixture bridge partially fixed it (de-noised ~32% honest prevention), but it is a manual approximation of what a self-play curriculum produces for free: a stream of progressively harder, *reactive* opponents.

The deeper limitation is structural. SB3 PPO is single-agent; it cannot co-adapt two policies. True joint co-adaptation — and a principled cure for the cold-opponent problem — requires either a custom multi-agent loop or a multi-agent framework. We choose **RLlib** (new API stack: RLModule + Learner) driven by **Ray Tune**, and rebuild training as a **dual-population PFSP self-play league**.

The migration is more tractable than "rewrite the env," because the env is *already* multi-agent. The SB3 coupling is concentrated in the training/eval/callbacks/CLI layer; the simulator core, the env, the obs system, and the reward stack are framework-agnostic and carry over.

### Decisions settled in brainstorming (2026-06-07)

| Axis | Decision |
|------|----------|
| Framework | RLlib **new API stack** (RLModule + Learner), PPO, multi-agent |
| Orchestration | **Ray Tune** owns the training loop; native `WandbLoggerCallback` |
| Migration extent | **Full SB3 retirement**; retain Hydra `conf/` tree, W&B artifacts/`:prod` aliases, and `dsim inventory/lineage/promote` |
| Bootstrap | **Pure-scratch** league (no SB3→RLlib weight bridge); scripted opponents seed the population; dense→sparse reward anneal handles cold-start |
| Policies | Two **independent** trainable policies (`main_red`, `main_blue`) — asymmetric roles, obs goal-targets, and rewards |
| League | **PFSP dual-population** (Red population + Blue population), forward-compatible with AlphaStar-lite exploiters |

## Goals

- Retire SB3 entirely; all training runs on RLlib new API stack under Ray Tune.
- Train both teams as a **dual-population self-play league**: a Red population and a Blue population, each `main` policy training against opponents sampled (PFSP) from the *other* population.
- Solve the Blue-wall **structurally** — the league auto-generates the diverse, reactive Red curriculum that the manual `beeline+R4t` mixture was hand-approximating.
- Preserve the infra investment: Hydra config composition, W&B artifacts with `:prod` aliases, and the `dsim` inventory/lineage/promote conventions all survive (re-pointed at RLlib checkpoint dirs).
- Keep the framework-agnostic core (`core/`, `team_env.py`, `obs_spec.py`, `rewards/`, scoring/tagging/crash/scene) untouched.
- De-risk with a **walking-skeleton-first** build sequence: prove the env/obs/reward/PID/Ray plumbing with a single policy before any league machinery.

## Non-goals

- **No SB3→RLlib weight bridge.** Pure-scratch league; existing checkpoints (`R4t`, `blue_v7`, `blue_v4`) are *not* imported. They remain in `backup/` as historical references only.
- **No AlphaStar-lite exploiters in this milestone.** The design is forward-compatible (the population machinery is the same), but dedicated main-/league-exploiter policies are deferred.
- **No solo (single-agent) training path.** The milestone is reached; all training is team-play. `QuidditchSimpleEnv` and `eval_solo` survive only as diagnostics / canaries, not as a training entrypoint.
- **No `core/policies/warm_start.py` port.** Cross-obs-spec input-layer surgery has no role in a pure-scratch league. If RLModule→RLModule transfer is ever needed, it is a separate future spec.
- **No GPU/MPS optimization in this milestone.** Correctness first on CPU learners + CPU env-runners; Apple-Silicon acceleration is a follow-up.
- **No bit-deterministic RLlib canary as a hard gate** (see Open Question OQ-3) — a statistical smoke test replaces the exact-fingerprint team canary if determinism proves impractical.

## Architecture

### Layer map: what carries over, what dies, what's new

**Reused untouched (framework-agnostic):**
- `core/world.py`, `core/quadrotor.py`, `core/position_controller.py`, `core/mjcf/*`, `core/drone/cf2x.py` — simulator + PID. No RL framework knowledge.
- `envs/quidditch/team_env.py` `QuidditchTeamEnv` — already a correct PettingZoo `ParallelEnv`. The per-agent obs composition (`learner_spec`/`build_spec_from_block_names`) and the `step({red_0, blue_0}) → (obs, rew, term, trunc, info) dicts` contract are exactly what RLlib's multi-agent API consumes.
- `envs/quidditch/obs_spec.py` — `ObsBlock`/`ObsSpec`, `BLOCK_BY_NAME`, `build_spec_from_block_names`. Obs composition is env-internal; unchanged.
- `envs/quidditch/rewards/*` — `RewardStack` already returns **per-agent dicts** (`{red_0: r, blue_0: r}`). The old wrapper discarded the opponent's slice; RLlib consumes both. This is the single biggest reason the migration is cheap.
- `envs/quidditch/{scene,scoring,tagging,crash}.py`, `constants.py`.
- Scripted `Opponent` classes (`ZeroOpponent`, `BeelineRed`, `BeelineBlue`, `IntercepterBlue`) — repurposed via a thin adapter as **frozen, non-learning league seeds** (see B2).

**Retired / deleted:**
- `OpponentControlledEnv` (`team_env.py`) — the single-agent collapse. RLlib consumes the `ParallelEnv` directly.
- SB3 PPO construction, `DummyVecEnv`/`SubprocVecEnv`/`VecFrameStack`, the SB3 callback set (`EvalCallback`, `CheckpointCallback`, `VideoRecorderCallback`, `WandbCallback`), and the manual `FrameStackWrapper`.
- The SB3 training path in `scripts/train.py`.
- `dsim sweep {create,agent,agents}` — superseded by Tune.
- SB3 `.zip` checkpoint assumptions throughout `promote`/`lineage`/`eval`.
- `core/policies/warm_start.py` and `scripts/migrate_legacy_models.py` (no warm-start, no legacy import).

**New components (the migration surface):**

1. **Env adapter** — register `QuidditchTeamEnv` with RLlib. First choice: RLlib's `ParallelPettingZooEnv` (`ray.rllib.env.wrappers.pettingzoo_env`); fallback: a thin `MultiAgentEnv` subclass if the wrapper's space introspection fights the per-agent specs. Registered via `tune.register_env("quidditch_team", env_creator)`. Frame-stacking (was `VecFrameStack`) moves to a new-stack **ConnectorV2** in the env-to-module pipeline, or into env-internal obs history — decided in the skeleton phase (see OQ-2).

2. **Config bridge** (`conf/` Hydra → `PPOConfig`) — a new `_target_`-instantiated builder maps Hydra fields onto the `AlgorithmConfig` builder chain (`.training()/.env_runners()/.learners()/.multi_agent()/.rl_module()`). New config groups:
   - `algo/ppo.yaml` (replaces `trainer/ppo.yaml`) — PPO hyperparameters.
   - `multiagent/` — `policies` dict, `policy_mapping_fn` selection, `policies_to_train`.
   - `league/` — population caps, snapshot win-rate threshold + eval N, PFSP exponent `p` and uniform floor, pruning policy.
   - `tune/` — `RunConfig` (stop conditions, `CheckpointConfig` cadence, `WandbLoggerCallback`).
   The existing groups (`env`, `obs`, `reward`, `curriculum`, `wandb`) are reused; `opponent/` is repurposed to describe *seed* scripted opponents and `init/` is simplified (scratch-only for now).

3. **League callback** (`RLlibCallback.on_train_result`) — the heart of Part B. Each iteration: evaluate each `main` against its opposite population, freeze + `add_module` a snapshot when the win-rate threshold is cleared, and update the PFSP sampling weights consumed by `policy_mapping_fn`. Owns dual-population bookkeeping (membership, per-opponent win-rate estimates, prune decisions). State must be checkpointed alongside the algorithm so Tune restore is faithful.

4. **RLModule spec** — `MultiRLModuleSpec` with trainable `main_red`, `main_blue` (MLP matched to the current net width from `trainer/ppo.yaml`) plus frozen snapshot modules added dynamically via `Algorithm.add_module(...)`. Scripted opponents are wrapped as deterministic, non-learning `RLModule`s (a `forward_inference` that calls the scripted `act`). `policies_to_train=["main_red","main_blue"]` ensures gradients flow only to mains.

5. **Checkpoint / lineage adapter** — RLlib checkpoints are **directories**, not `.zip`. Re-point `dsim promote`/`lineage`/`inventory` at RLlib checkpoint dirs; keep the per-run `.hydra/{config,meta}.yaml` convention and the W&B-artifact + `:prod`-alias model registry. Lineage edges become snapshot ancestry within the league rather than a linear ladder.

6. **Eval** — port `eval_team` to load RLModule checkpoints and run head-to-head rollouts on the `ParallelEnv` (no opponent-controlled wrapper). Reimplement `eval/success_rate` (honest Blue prevention — the load-bearing, length-unconfounded metric) as a custom evaluation metric via RLlib's `evaluation()` config + a metrics callback. `eval_solo` is retained only as a "main vs scripted" diagnostic.

7. **Tune harness** — `tune.Tuner(PPO, param_space=<hydra-built config>, run_config=RunConfig(callbacks=[WandbLoggerCallback(...)], stop=..., checkpoint_config=...))` replaces `model.learn()` and `dsim sweep`. Hydra composes; the builder emits the `param_space`; Tune owns the loop, checkpointing, restore, and HPO.

### macOS / Ray correctness constraints (carried from prior hard-won lessons)

- **`KMP_DUPLICATE_LIB_OK=TRUE` must reach Ray env-runner subprocesses.** Today's `conftest.py`/`train.py` module-level set does *not* propagate across Ray's process boundary. Set it via `ray.init(runtime_env={"env_vars": {"KMP_DUPLICATE_LIB_OK": "TRUE"}})` (and the equivalent for learner/eval workers).
- **`==` not `is` across the Ray serialization boundary.** Ray pickles env config + modules to workers; the SubprocVecEnv lesson applies verbatim. Audit any module-level reward-term / obs-spec singletons that participate in identity checks.
- **The env creator receives a config dict.** The env factory must read *every* knob from that dict — no closure capture across the Ray boundary. (Generalizes the "thread new kwargs through ALL construction sites" lesson: Ray's `env_creator` is the new must-audit construction site.)
- **CPU-first.** Design for CPU learners + CPU env-runners; RLlib's MPS support on Apple Silicon is unreliable. Correctness before throughput.
- **Keep rendering out of env-runners.** Video recording stays in a dedicated eval/render worker (the CGL-invalid-connection failure mode in headless subprocesses is unchanged). `KMP`/MuJoCo render caveats from the brain still hold.

## Part B — Training methodology (the league)

### B1 · Policy topology

- Trainable: `main_red` (attacker), `main_blue` (defender).
- Frozen populations: `red_pop`, `blue_pop`. Each is seeded with scripted opponents and grows by snapshotting the corresponding `main`.
- `policy_mapping_fn(agent_id, episode, ...)`: per episode, pair one `main` against one frozen member of the *opposite* population (with a tunable fraction of `main_red` vs `main_blue` "live" episodes for direct co-adaptation). Concretely: with prob `q`, `red_0→main_red` and `blue_0→` a sampled `blue_pop` member (Blue is the frozen field, Red is exploited); with prob `q`, the symmetric case; with prob `1-2q`, `main` vs `main`.
- `policies_to_train=["main_red","main_blue"]` — frozen members never update.

### B2 · Cold-start (the Blue-wall cure, generalized)

- **Phase 0:** populations contain *only* scripted opponents (`beeline_red`/`beeline_blue`/`intercepter_blue`/`zero`). `main_red` learns to score vs scripted Blue; `main_blue` learns to defend vs scripted Red. **Dense distance shaping ON** (Red→hoop, Blue→midpoint) — the existing reward terms.
- **Reward anneal:** dense distance shaping → sparse outcome reward on a step schedule (a schedule hook over the existing `RewardStack` magnitudes). This is the same dense→sparse pattern the manual campaign used, now automated and league-wide.
- **Judge by `eval/success_rate`, never reward** — the length-confounded `best_model`-by-reward failure mode (corr 0.88 with episode length; stalling + suicide hacks) is a standing hazard. Promotion and snapshot gating both key off honest prevention / score rates, not return.

### B3 · PFSP progression

- **Snapshot rule:** when `main_X`'s win-rate vs the current opposite population ≥ `threshold` (e.g. 0.7 over N eval episodes), freeze a copy into `X_pop`.
- **PFSP sampling:** opponent drawn with probability ∝ `(1 − winrate_vs_that_opponent)^p` — concentrate training on opponents the main currently struggles against — with a uniform floor to retain coverage of already-beaten opponents (anti-forgetting).
- **Population management:** a cap per population plus a prune-dominated policy (drop snapshots that no longer contribute distinct difficulty), to bound memory and per-iteration eval cost.

### B4 · Evaluation & promotion

- Periodic eval battery (reuse `eval_team` metrics): `eval/success_rate` (Blue honest prevention), Red score-rate, takedown-rate, terminal-cause histogram.
- Promote a `main_*` to W&B artifact `:prod` when it dominates the prior prod across the battery. Lineage records snapshot ancestry.

### B5 · Curriculum knobs

- `red_start_x_max` and `red_action_scale` (already added + TDD'd in the retraining campaign) become **league difficulty levers** — anneal Red's start spread and action authority as the league matures.
- The standing open question — *is 100% prevention of a competent scorer geometrically feasible, or is the target a score-rate reduction?* — gets answered **empirically** by Blue's asymptotic prevention rate against a co-adapting (not frozen, non-reactive) Red.

## Build sequence (walking skeleton first)

1. **Skeleton.** New-stack PPO, a single trainable `main_red` vs a *scripted* frozen Blue on the wrapped `ParallelEnv`, Tune + W&B wired → reproduce "Red learns to score." Proves the env adapter, obs composition, reward stack, PID interface, Ray plumbing, and every macOS/Ray gotcha — before any league code exists.
2. **Two-policy simultaneous.** Add `main_blue`; naive self-play (single live/lagged opponent). Proves dual-policy gradient flow + `policy_mapping_fn`.
3. **Population + snapshotting.** Add frozen snapshot populations + the threshold snapshot callback (uniform sampling first).
4. **PFSP.** Add prioritized sampling + population pruning.
5. **Curriculum + promotion.** Reward anneal schedule, difficulty levers, eval battery + lineage/promote adapter.
6. **Retire SB3.** Delete `OpponentControlledEnv` + the SB3 train path once 1–5 are green.

Each step is independently verifiable and leaves a runnable system; the league is only switched on after the plumbing is proven.

## Testing strategy

- **Env-adapter contract** (unit): obs/action spaces per agent match the persisted specs; `reset`/`step` dict shapes survive the RLlib wrapper.
- **`policy_mapping_fn`** (unit): pairings respect the `q`/population logic; frozen members never appear in `policies_to_train`.
- **PFSP sampling** (unit): empirical sample distribution matches `(1−winrate)^p` + uniform floor within tolerance.
- **Snapshot freeze** (unit): a snapshotted module's weights do not change across subsequent training iterations.
- **Smoke run** (integration): a few-iteration Tune run completes, writes a checkpoint, and restores faithfully (including league callback state).
- **Canary** (see OQ-3): either port the team canary to a deterministic RLlib rollout fingerprint, or — if RLlib rollouts resist bit-determinism — replace it with a statistical smoke test (mean/variance of a fixed scripted-vs-scripted rollout within tolerance).
- **macOS render tests** remain GUI-shell-only (the CGL-in-headless caveat is environmental, unchanged).

## Open Questions

- **OQ-1 · `warm_start.py` retirement.** Confirmed retired for this milestone (pure-scratch). The surgery *concept* may return as RLModule→RLModule transfer in a future spec; not in scope here.
- **OQ-2 · Frame-stacking location.** ConnectorV2 in the env-to-module pipeline vs env-internal obs history. Decide empirically in the skeleton phase; both are viable, ConnectorV2 is the more idiomatic new-stack choice.
- **OQ-3 · Canary determinism.** RLlib rollouts are harder to make bit-deterministic than SB3's. If exact fingerprinting proves impractical, fall back to the statistical smoke test above rather than block the migration on it.
- **OQ-4 · Cold-start insurance.** Phase-0 scripted-opponent seeding + dense shaping is the primary cold-start defense. Whether to additionally gate league opening behind a dense-reward "scoring canary" (Red must reliably score vs scripted Blue before snapshotting begins) is a tuning decision deferred to implementation.

## Risks

- **Ray on macOS Apple Silicon** is less battle-tested than on Linux; the env-var propagation + CPU-first constraints are mitigations, but expect plumbing friction in the skeleton phase. This is precisely why the skeleton is step 1.
- **MARL non-stationarity / cycling** — the league (PFSP + populations + anti-forgetting uniform floor) is the designed mitigation, but co-adaptation can still produce degenerate equilibria. AlphaStar-lite exploiters are the escalation path if it does.
- **Infra re-pointing scope creep** — `promote`/`lineage`/`inventory` assume `.zip`; the directory-checkpoint adapter touches more of `dsim` than it first appears. Scoped explicitly as new-component #5, not folded into eval.
