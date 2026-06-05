# Oracle-Privileged Critic + World-Frame Observable Redesign — Design

- **Date:** 2026-06-04
- **Branch / worktree:** `exp/env` (`worktrees/exp/env/`, off `develop`)
- **Status:** Approved design, ready for implementation plan
- **Slice:** Observables + privileged critic (one slice of a larger asymmetric-training rethink; the other slices — co-training/self-play, reward taxonomy, partial-observability sensors, richer game — are explicitly out, see §13)

---

## 1. Context & Motivation

The project is chronically blocked on a **weak Blue defender** (brain `index.md` → "Active Priorities"). Two independent sources point at the same root causes:

1. **The value function is starved.** Training uses SB3's default shared actor-critic — the critic sees exactly the same partial observation as the actor. In an asymmetric 1v1 game, value estimation is high-variance because the critic can't use train-time-only ground truth it is allowed to see.
2. **The horizon is too short.** `gamma=0.99` at 120 Hz control ⇒ ~100-step (<1 s) effective horizon over ~3600-step episodes. The agent cannot reason about an approach-and-score or defend-and-takedown sequence.

A user-supplied review of 6 papers on asymmetric competitive games (distributed-scale systems with a richer game) recommends, among heavier machinery that does **not** transfer to a single-CPU SB3 setup, two principles that do: **privileged "Invisible Information" critics** and **longer strategic horizons**.

This slice redesigns the **environments and observables** to fix #1 directly and #2 cheaply, per the decisions in §3. The intended outcome: a Blue (and later Red) that learns decisively because its critic is well-informed and credit assignment spans seconds, not milliseconds.

Everything is **opt-in, additive config**. Existing flat-obs checkpoints, the `team_v2`/`single_agent` paths, and both canaries (`SCORED at step 434 / total reward 7.3837`; team-env canary) stay **byte-identical**.

---

## 2. Current State (what we're changing)

- **Obs** are a single flat `ObsSpec` (`obs_spec.py`) of named `ObsBlock`s, composed from a YAML `blocks:` list, frame-stacked ×3 via `VecFrameStack`. Both actor and critic consume the identical vector. The team env builds a universal feature dict (`team_env._build_agent_features`) and `pack()`s the subset a spec requests.
- **Frame mismatch.** The actor obs has straddled world-frame (`duel_v2_world`, `blue_v4`) and body-ego (`duel_v3_body_ego`, `blue_v7/v8`). **The action is applied in the world frame** (`simple_env.py` / `team_env.py`: `self._setpoint += action * ACTION_SCALE`, with `x,y` clipped to `±ARENA_RADIUS` in world coords). A body-ego actor therefore must internally learn a yaw rotation to connect perception to control — undercutting the very yaw-invariance that motivated body-ego.
- **Training shape.** Single SB3 learner vs a scripted/frozen opponent via `OpponentControlledEnv` (OCE). This slice keeps that shape.

---

## 3. Design Decisions (locked via brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Stay on SB3 PPO.** Privileged critic ships as a custom `ActorCriticPolicy` with a `Dict` obs space. | Switching to RLlib/MAPPO would discard the Hydra/W&B/obs-spec/warm-start investment for a 1v1. |
| D2 | **Privilege = oracle/future**, not partial-obs sensors. | Infra-light: the env holds full physics state and (this slice) controls the opponent, so it can hand the critic genuine lookahead with no new env mechanics. |
| D3 | **Actor obs = world frame, fully consistent + pruned.** `lin_vel`, relative geometry, opp velocity all world; attitude (`ang_vel`,`ang_pos`) stays body; drop the body-ego and `unit_to_goal` blocks. | Obs frame matches the world-frame action (§2); removes the `midpoint_alpha`-in-obs coupling. |
| D4 | **Plumbing = Dict obs + custom policy** (`{actor, critic}`), `share_features_extractor=False`. | SB3-idiomatic at the forward-pass level; clean gradient isolation; Dict supports per-key frame-stacking. |
| D5 | **Keep single-learner-vs-frozen/scripted shape.** | Co-training/self-play is a separate, deferred slice. |
| D6 | **Clean break on the obs schema; CTDE runs are `init=scratch` only.** | `models/` is already cleared for retrain; no flat→dict warm-start. |
| D7 | **Oracle via kinematic extrapolation** (const-velocity, horizon `k`), not true sim-rollout. | Effectively free; reuses the existing `InterceptShaping` pattern. True rollout costs ~k× physics × n_envs. |
| D8 | **Incorporated from the alt plan:** actor feature **normalization**, a **`time_remaining`** feature, **enrich the critic** with cheap ground-truth blocks, **Phase-0 de-risk** of the custom policy, **flat↔dict incompatibility** in obs-compat. | Free optimization/representation wins; riskiest-piece-first engineering. |
| D9 | **Incorporated from the alt plan:** raise **gamma 0.99→0.9995, gae_lambda 0.95→0.98** via the experiment YAML only. | ~16 s effective horizon; cheap, low-risk, high-value. Likely a core reason Blue can't learn decisive multi-second behavior. |
| D10 | **Deferred from the alt plan:** PBRS reward shaping, action-repeat=4. | PBRS belongs to the deferred reward slice; action-repeat changes control rate and risks slower takedown reactions. Reward stack reused unchanged (`team_v3_intercept`). |

---

## 4. Architecture Overview

**Opt-in dual-view obs.** A run's obs config declares `obs_mode: flat | dict`.

- `flat` (default, today's behavior): single `ObsSpec`, flat vector, standard SB3 `MlpPolicy`/`MultiInputPolicy`. Canaries byte-identical.
- `dict`: the **learner's** observation becomes `Dict({"actor": Box, "critic": Box})`; the **opponent's stays flat** (a frozen/scripted policy expecting its trained shape). Training uses the new `AsymmetricActorCriticPolicy`.

Only the learner side is touched — this propagates automatically through OCE, which forwards `obs[learner_id]` to SB3 and `obs[opponent_id]` to the opponent.

---

## 5. Observation Design

### 5.1 Actor view — `CTDE_V1.actor` (world-frame, normalized), `n_stack=3`

| block | dim | frame | normalization | change vs `duel_v2_world` |
|---|---|---|---|---|
| `ang_vel` | 3 | body | ÷ 10 rad/s | unchanged |
| `ang_pos` | 3 | body | ÷ π | unchanged |
| `lin_vel_world` | 3 | **world** | ÷ 3 m/s | **NEW block** (was body) |
| `lin_pos` | 3 | world | ÷ `ARENA_RADIUS` | unchanged |
| `vec_to_hoop` | 3 | world | ÷ `ARENA_RADIUS` | unchanged |
| `opp_pos_rel` | 3 | world | ÷ `ARENA_RADIUS` | unchanged |
| `opp_vel_rel_world` | 3 | world | ÷ 3 m/s | unchanged |
| `closing_rate` | 1 | — | ÷ 3 m/s | unchanged |
| `time_remaining` | 1 | — | frac ∈ [0,1] | **NEW block** |
| ~~`unit_to_goal`~~ | — | — | — | **dropped** (midpoint coupling) |

Total **23-d** (×3 = 69-d). Role-symmetric for Red and Blue; attack/defend asymmetry lives in reward, not obs.

### 5.2 Critic extras — privileged, world-frame, **unstacked**

The critic sees `concat(stacked actor view 69, critic extras 28) = 97-d`. The extras are everything the actor cannot see:

**Part A — oracle/future (kinematic, horizon `k = ORACLE_HORIZON_S = 0.5 s`, cap `T_cap = 3.0 s`):**

| block | dim | formula |
|---|---|---|
| `opp_next_action` | 4 | the `[dx,dy,dyaw,dz]` the opponent applies this step (injected by OCE, §6.1) |
| `self_future_disp` | 3 | `k·v_self_world / ARENA_RADIUS` |
| `opp_future_rel` | 3 | `((opp_pos + k·v_opp) − self_pos) / ARENA_RADIUS` |
| `score_pred` | 3 | `[time_to_hoop_plane/T_cap, lateral_miss_from_ring_center/ARENA_RADIUS, approach_align∈[−1,1]]` for the attacker (red_0) |
| `takedown_pred` | 3 | `[time_to_CPA/T_cap, predicted_min_sep/ARENA_RADIUS, imminent_crash_flag∈{0,1}]` |

**Part B — ground-truth state (cheap "Invisible Information" the egocentric actor lacks):**

| block | dim | formula |
|---|---|---|
| `red_pos_abs` | 3 | `red_pos / ARENA_RADIUS` |
| `blue_pos_abs` | 3 | `blue_pos / ARENA_RADIUS` |
| `tag_state_onehot` | 2 | `[tag_during, tag_cooldown_active]` |
| `terminal_margins` | 4 | `[red_wall, blue_wall, red_floor, blue_floor]` margins, normalized (1 = safe, 0 = at boundary) |

Total extras **28-d**. All blocks normalized and **NaN-guarded**: degenerate kinematics fall back to bounded sentinels (`v_normal≈0 ⇒ time_to_plane=T_cap, lateral_miss=ARENA_RADIUS, align=0`; `‖v_rel‖≈0 ⇒ time_to_cpa=0, min_sep=‖r‖, flag=0`). A test asserts the entire obs is finite.

### 5.3 Normalization without breaking the flat path

Normalization is applied **at pack-time for CTDE specs**, via a per-block scale map (`NORM_BY_BLOCK`), **not** inside the raw feature dict. The flat path calls plain `pack(spec, features)` and is byte-identical; the CTDE path calls `pack_normalized(spec, features, NORM_BY_BLOCK)`. Block *identity* `(name, dim, frame)` is unchanged; the YAML documents the scales.

### 5.4 Frame-stacking

A thin `SelectiveDictFrameStack` stacks only the `actor` key (×3); the `critic` key passes through (×1) — the oracle already encodes temporal/future content, so re-stacking it is wasted width. (SB3 2.8.0 `VecFrameStack` stacks Dict spaces per-key uniformly; we need *selective* stacking, hence the thin wrapper.) `n_stack` is configurable; default 3 (matches the validated `blue_v4`/`blue_v7` runs).

---

## 6. Data Flow

### 6.1 Opponent-action timing (the one subtle bit)

`opp_next_action` must equal the action the opponent *applies* leaving `s_t`, embedded in `obs_t` so the critic's `V(s_t)` sees it. Opponents are deterministic given state (scripted `f(state)`; frozen `policy.predict(opp_obs, deterministic=True)`), so OCE computes it at obs-build and **caches it for the apply step** — guaranteeing embedded == applied even for a stochastic opponent:

```
OCE.reset():        env.reset → s0
                    opp_a0 = opponent(s0); cache opp_a0
                    return obs0 = {actor: f_act(s0), critic: f_crit(s0, opp_a0)}

OCE.step(a_learner): env.step({learner: a_learner, opp: cached opp_a0}) → s1, r0
                    opp_a1 = opponent(s1); cache opp_a1
                    return obs1 = {actor: f_act(s1), critic: f_crit(s1, opp_a1)}, r0, …
```

Only `opp_next_action` is injected by OCE; the other 24 critic-extra dims are pure functions of physics state, computed inside the env. Velocities come from `data.qvel[dofadr:dofadr+3]` (world-frame, same source as `opp_vel_rel_world` today).

### 6.2 Custom policy & gradient isolation

`AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy)`, `share_features_extractor=False`. An overridden `_build()` constructs:
- `pi_features_extractor` reading only `obs["actor"]`;
- `vf_features_extractor` reading `concat(obs["actor"], obs["critic"])`;
- independent actor/critic MLP heads (different input dims).

`predict()` / rollout uses only `obs["actor"]`; `evaluate_actions()` uses both. Oracle has **no computational path to the action distribution** — a unit test asserts `∂(action logits)/∂(critic) == 0` and that perturbing the critic key *does* move `V`.

### 6.3 Training vs eval

The trained actor consumes `obs["actor"]` only, so eval needs no critic info. `eval_team`/`eval_solo` request an actor-only view (env returns the actor Box, or a Dict with a zeroed critic key the actor ignores). Oracle/ground-truth computation is **training-only**.

---

## 7. Obs-Spec System Changes

- **New blocks** in `obs_spec.py` (11): `LIN_VEL_WORLD`, `TIME_REMAINING`, `OPP_NEXT_ACTION`, `SELF_FUTURE_DISP`, `OPP_FUTURE_REL`, `SCORE_PRED`, `TAKEDOWN_PRED`, `RED_POS_ABS`, `BLUE_POS_ABS`, `TAG_STATE_ONEHOT`, `TERMINAL_MARGINS`. `BLOCK_BY_NAME` auto-registers them.
- **`build_ctde_specs_from_yaml(stem) -> (actor_spec, critic_spec)`** alongside the existing flat `build_spec_from_block_names`. `ObsSpec` itself is unchanged.
- **YAML schema** gains `obs_mode`, `actor_blocks`, `critic_blocks`; legacy `blocks:` still parses (→ flat). New `conf/obs/ctde_v1.yaml`.
- **`config_schema.py`**: extend `ObsConfig` (`obs_mode`, `actor_blocks`, `critic_blocks`); add `PolicyConfig` (`policy_class`, `net_arch`, `share_features_extractor`) + `conf/policy/{mlp,asymmetric}.yaml`; add `- policy: mlp` to `conf/config.yaml` defaults.
- **Persistence**: `.hydra/config.yaml` + `meta.yaml [obs]` record both block lists + `obs_mode`; `MODEL.md` renderer and `dsim obs-specs`/`obs-preflight` learn the dual layout.
- **Compat** (`core/obs_compat.py::preflight`): **flat↔dict is incompatible** (CTDE = scratch-only). For dict↔dict pretrain (future), compat is judged on the **actor spec only**; a critic-spec mismatch re-inits the value extractor with a **warning, not a raise** (the critic isn't behavior-bearing at deployment). `resume` still requires exact match on both.

---

## 8. Discounting / Horizon (config-only)

In the experiment YAML override **only** (leaving `conf/trainer/ppo.yaml` at 0.99 so canary defaults are untouched):
- `gamma: 0.9995` (~16 s effective horizon)
- `gae_lambda: 0.98`

No action-repeat (deferred, D10).

---

## 9. Reward

**Unchanged.** Reward terms read `StepState`, not obs, so the existing `team_v3_intercept` stack drops in as-is. PBRS is deferred to the reward slice (D10). The actor dropping `unit_to_goal` does not affect reward (the env still computes the midpoint for `HoopDistancePenalty` from state).

---

## 10. Edge Cases / Error Handling

- **Degenerate kinematics** → bounded sentinels (§5.2); whole-obs finite check in tests.
- **Opponent-action consistency** via the cache-at-build pattern (§6.1) — covers scripted / frozen / mixture uniformly.
- **Cross-process safety**: all specs are frozen (picklable) dataclasses; any spec-identity check uses `==`, never `is` (memory `feedback_eq_not_is_across_processes` — bit `blue_v7` under `SubprocVecEnv`).
- **Thread the CTDE flag through *every* construction site**: the factory `_make_thunk`, the **inline `eval_env_fn`**, and `video_env_fn` in `scripts/train.py` (memory `feedback_thread_kwarg_through_all_sites` — the `bd4e8b2` eval-env crash). The smoke test must actually trigger an eval.
- **Aftermath window** (`crash_aftermath_seconds>0`, eval-only): critic extras still compute but are unused (training runs at `0.0`).
- **simple_env + canaries**: untouched. Default `conf/obs` stays `duel_v2_world` (flat); only the new experiment opts into the dict path ⇒ canaries byte-identical.

---

## 11. Testing Strategy

- **Unit**: each oracle/ground-truth formula with hand-computed values incl. sentinel cases; `build_ctde_specs_from_yaml` + `pack_normalized` dims & scales; legacy `blocks:` still parses; **gradient isolation** (critic→action-logits grad zero, critic→value non-zero); `SelectiveDictFrameStack` stacks actor only.
- **Integration**: short dict-mode training smoke (`WANDB_MODE=disabled`) that **triggers eval**, shapes line up, obs finite; actor-only eval loads a dict-trained model and runs; `SubprocVecEnv(n_envs>1)` smoke for pickling.
- **Canary**: assert team + single-agent fingerprints byte-identical with `obs_mode=flat`.

---

## 12. Build Sequence (phased, TDD; each phase leaves `make test` green)

**Phase 0 — Asymmetric policy spike (de-risk first).** `core/policies/asymmetric.py` (`ActorExtractor`, `CriticExtractor`, `AsymmetricActorCriticPolicy`) + export. Unit-test against a **mock `spaces.Dict` env** (no quidditch dependency): actor extractor ignores the critic key; critic concatenates both; `PPO(...)` constructs, `learn(512)` runs, **`save()`→`load()` round-trips** (`_get_constructor_parameters()`). *If serialization is intractable in SB3 2.8.0, surface immediately — fallback: a single `CombinedExtractor` over the full Dict with the actor MLP masking the critic slice.*

**Phase 1 — Obs-spec dual plumbing (pure).** 11 new blocks; `build_ctde_specs_from_yaml`; `NORM_BY_BLOCK` + `pack_normalized`; `conf/obs/ctde_v1.yaml`. Tests only; no env behavior change; flat path byte-identical.

**Phase 2 — team_env feature computation.** Add `time_remaining`, world-frame actor features, oracle features (kinematic + sentinels), ground-truth features to the feature dict — all behind a `ctde_mode` kwarg branch so the flat/`duel_v2` path is byte-identical.

**Phase 3 — CTDE Dict obs + OCE opp-action cache.** `QuidditchTeamEnv(ctde_mode=False)`; in dict mode the learner's `observation_spaces` entry becomes `spaces.Dict` and `_build_agent_obs` returns `{actor, critic}`; opponent stays flat. OCE caches the opponent action at obs-build and injects `opp_next_action` (§6.1).

**Phase 4 — SelectiveDictFrameStack + factory wiring.** `TeamEnvFactory` gains `ctde_mode`; applied/threaded through `_make_thunk`; `FrameStackWrapper` (`opponents.py`) gains Dict support.

**Phase 5 — train.py wiring + first experiment.** Use `cfg.policy.policy_class`, build `policy_kwargs`; when `cfg.obs.obs_mode=="dict"` call `build_ctde_specs_from_yaml` and pass both specs to the factory; **thread `ctde_mode` through the inline `eval_env_fn` and `video_env_fn`**; teach `obs_compat.preflight` the flat↔dict rule; `conf/experiment/blue_oracle_v1.yaml` (env=team, obs=ctde_v1, reward=team_v3_intercept, policy=asymmetric, **opponent=beeline_red** — scripted, always on disk, deterministic, mirrors `blue_v4`; `frozen:red_v1` is the next ladder rung once `models/` is restored, init=scratch, curriculum=random_start, gamma=0.9995, gae_lambda=0.98). Smoke test `tests/scripts/test_ctde_train_smoke.py`.

---

## 13. Scope

**In:** team_env CTDE dual-view obs (opt-in); `CTDE_V1` actor + critic specs; the 28-d oracle/ground-truth critic extras (kinematic); `AsymmetricActorCriticPolicy`; `SelectiveDictFrameStack`; dual obs-spec + YAML + persistence + compat; OCE opp-action cache; all-construction-sites threading; actor-only eval; gamma/λ override; one experiment YAML proving it end-to-end.

**Out (each a later slice):** simultaneous co-training · self-play pool / pFSP · reward taxonomy + **PBRS** + anti-farming · partial-observability sensors (range/FOV/noise) · **action-repeat** / control-rate change · true sim-rollout oracle · simple_env oracle · growing the game (multi-hoop, obstacles, extra roles) · body-ego actor frame.

---

## 14. Files

**Create:** `core/policies/asymmetric.py`; `conf/policy/{mlp,asymmetric}.yaml`; `conf/obs/ctde_v1.yaml`; `conf/experiment/blue_oracle_v1.yaml`; tests `tests/core/policies/test_asymmetric_policy.py`, `tests/envs/quidditch/{test_ctde_features,test_ctde_obs,test_obs_normalization}.py`, `tests/scripts/test_ctde_train_smoke.py`.

**Modify:** `envs/quidditch/obs_spec.py` (11 blocks + `build_ctde_specs_from_yaml` + `pack_normalized`/`NORM_BY_BLOCK`); `envs/quidditch/team_env.py` (`ctde_mode`, feature computation, time_remaining, Dict obs space); `envs/quidditch/opponents.py` (OCE opp-action cache + injection; `FrameStackWrapper` Dict support; `SelectiveDictFrameStack`); `envs/quidditch/env_factories.py` (`ctde_mode`); `config_schema.py` (`PolicyConfig`, `ObsConfig` extension); `conf/config.yaml` (`- policy: mlp`); `scripts/train.py` (policy_class/policy_kwargs + CTDE obs path in `_build_or_load_model` **and** inline `eval_env_fn`/`video_env_fn`); `core/obs_compat.py` (flat↔dict incompat); `core/policies/__init__.py`.

**Must NOT change (stability anchors):** `conf/reward/{single_agent,team_v2}.yaml`; `conf/experiment/{canary_single,canary_team}.yaml`; `tests/.../test_scoring_canary.py`, `tests/.../test_team_env_canary.py`; `conf/trainer/ppo.yaml` gamma (override per-experiment).

**Reuse:** `obs_spec.{pack, world_to_body}` + `BLOCK_BY_NAME` auto-registry; the `_prev_dist_to_opp` caching pattern in `team_env`; `RewardStack`/`team_v3_intercept`; `resolve_parent`/`load_run_context`; SB3 `VecFrameStack` Dict support.

---

## 15. Risks & Open Items

- **Custom SB3 policy serialization** (Phase 0) is the gating risk; validated before anything depends on it. Fallback documented (CombinedExtractor + masking).
- **CTDE is scratch-only** — no flat→dict warm-start. Acceptable: the goal is a fresh, well-informed run, not extending `blue_v4`.
- **Kinematic oracle is approximate** (const-velocity). Still strictly more than the actor knows; true rollout is a later upgrade if the critic underperforms.
- **n_stack choice** (3 vs 1) is a knob; default 3. If the actor's velocity + time features prove sufficient, dropping to 1 simplifies the frame-stack wrapper.

---

## 16. Verification & First Experiment

1. `make test-fast` green after every phase; `make test` green after Phases 2, 4, 5 (canaries byte-identical — `obs_mode=flat`).
2. Phase 0: standalone `pytest tests/core/policies/test_asymmetric_policy.py` — construct, `learn(512)`, save/load round-trip.
3. Phase 5 smoke: `WANDB_MODE=disabled uv run python -m scripts.train +experiment=blue_oracle_v1 trainer.total_timesteps=2048` runs and produces a Dict-obs rollout that triggers an eval.
4. Real run (user-driven, after merge): `make train EXP=blue_oracle_v1` — Blue vs `beeline_red`, scratch (the `frozen:red_v1` rung follows once `models/` is restored). Watch in W&B for (a) higher critic explained-variance / lower value-loss than the flat-critic baseline (privileged critic working), (b) rising score-prevention / takedown rate, (c) decisive multi-second behavior under the longer horizon. Per memory `feedback_verify_before_commit`, qualitative viewer confirmation (`make eval-team … GUI=1`) is the user's call before promotion.
5. GitFlow finish: `--no-ff` merge `exp/env` → `develop`; update `brain/` (`index.md`, `changelog.md`, `decisions.md` for the CTDE/oracle-critic + horizon ADRs, `models.md` if a run is promoted).
