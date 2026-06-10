# Defender-Aware Reward + Step-3 Snapshot-Population League

**Date:** 2026-06-10
**Branch:** `feature/rllib-league-step-3` (worktree off `develop`)
**Implements:** Step 3 of the RLlib migration ("Population + snapshotting") from
`docs/superpowers/specs/2026-06-07-rllib-league-migration-design.md`, plus its
flagged prerequisite (a defender-aware env refactor).
**Builds on:** Step 2 (two-policy self-play), merged to `develop` at `c977e11`.

## Context

The RLlib migration is a 6-step build sequence (see the 2026-06-07 design doc,
"Build sequence"). Steps 1 (walking skeleton) and 2 (two-policy self-play) are
green and merged to `develop`. Step 2 proved dual-policy gradient flow:
`main_red` and `main_blue` are both trainable PPO policies, routed by a static
`policy_mapping_fn`, training live against each other under the competitive
`team_selfplay_v1` reward.

This spec covers **Step 3 — "Population + snapshotting"**: add frozen snapshot
populations plus a threshold-triggered snapshot callback (uniform opponent
sampling; PFSP is deferred to Step 4). It also covers the **prerequisite the
brain flagged after Step 2**: a *defender-aware env refactor* so
`InterceptShaping` can return a non-zero reward for `blue_0` in two-policy
self-play. The two ship together on one branch (per the user's scope decision).

### Why the prerequisite exists

`InterceptShaping` is the dense potential-based reward that pulls the defender
toward the attacker's *predicted* future position (the geometrically correct
intercept signal — distinct from `HoopAnchor`, which only anchors Blue near the
hoop). Its `StepState` inputs (`dist_def_to_future_red`,
`dist_def_to_future_red_prev`) are computed in `team_env.py` only when a single
`learner_id` is configured (the SB3-era one-learner assumption). Two-policy
self-play sets `learner_id=None`, so those fields stay `0.0` and the term
no-ops. `conf/reward/team_selfplay_v1.yaml` therefore omits `InterceptShaping`
today, and `tests/.../test_reward_stack.py::test_team_selfplay_v1_stack_composition`
asserts its absence. Re-enabling it requires populating those `StepState`
fields independently of `learner_id`.

### Decisions settled in brainstorming (2026-06-10)

| # | Axis | Decision |
|---|------|----------|
| Q1 | Intercept symmetry | **Asymmetric, Blue-only.** The intercept signal is defender→attacker by construction; Red's objective is the hoop (already covered by `HoopApproachShaping`). No symmetric Red intercept term. Only one direction (`blue_pos`→`future_red`) is ever needed. |
| Q2 | Obs path | **Untouched.** The `closing_rate` obs gate stays as-is. This work re-enables a *reward* term only; no obs-dimension or checkpoint-compat change. |
| Q3 | Snapshot trigger metric | **Reuse in-training metrics.** Read `red_score_rate` / `blue_prevention_rate` off `on_train_result`'s `result` dict (already logged by Step-2's `ScoreMetricsCallback`). No dedicated eval pass in Step 3 — that is the Step-4/5 upgrade. |
| Q4 | Cold-start / seeding | **Empty populations, snapshot-driven growth.** Training starts as main-vs-main (Step-2 behavior, proven to bootstrap from scratch). No scripted Phase-0 seeds; populations grow only via snapshots. |
| Q5 | Episode pairing | **`live_fraction` knob (default 0.5).** prob `live_fraction` → main-vs-main; remainder split 50/50 between "Red exploits frozen Blue" and "Blue exploits frozen Red"; uniform draw within a population; empty-pop ⇒ fall back to live. |

## Goals

- Re-enable `InterceptShaping` for `blue_0` under two-policy self-play by making
  the env's intercept-shaping `StepState` inputs `learner_id`-independent.
- Add frozen snapshot populations (`red_pop`, `blue_pop`) that grow when a main
  clears a win-rate threshold, with **uniform** opponent sampling.
- Keep restore checkpoint-faithful with **no separate league-state file** —
  population membership is derived from the restored module-id set.
- Leave a runnable, independently-verifiable system after Part A and after
  Part B.

## Non-goals

- **No PFSP / prioritized sampling** — Step 4. Step 3 is uniform sampling.
- **No population pruning / eviction** — Step 4. When a population hits its cap,
  Step 3 simply stops snapshotting that side.
- **No dedicated eval battery** for the snapshot trigger — Step 4/5 (the
  in-training metrics are the Step-3 trigger).
- **No obs changes** — the `closing_rate` obs gate is untouched (Q2).
- **No symmetric Red intercept term** (Q1).
- **No scripted Phase-0 population seeds** (Q4) — `ScriptedRLModule` remains
  available for a future step if cold-start ever regresses.
- **No AlphaStar-lite exploiters** — forward-compatible but out of scope (as in
  the parent migration spec).

## Architecture

### Part A — Defender-aware env refactor

**Files touched:** `envs/quidditch/team_env.py`,
`conf/reward/team_selfplay_v1.yaml`,
`tests/envs/quidditch/test_team_env.py` (new defender-aware tests),
`tests/envs/quidditch/rewards/test_reward_stack.py` (flip the composition
assertion).

**The change.** Today `team_env.py` gates the intercept-shaping computation in
two places behind `if self._learner_id is not None`, and selects the defender
from the learner:

- `reset()` (~lines 305-326): initializes `self._dist_def_to_future_red` /
  `self._dist_def_to_future_red_prev` only when a learner is set.
- `step()` (~lines 442-453): updates those caches only when a learner is set,
  with `defender_pos = blue_pos if learner_id == blue_id else red_pos`.

Refactor: extract a small helper that computes the defender→future-red distance
**unconditionally**, with the defender fixed to `blue_0`:

```python
def _dist_blue_to_future_red(self) -> float:
    red_vel_world = self._world.data.qvel[
        self._red_dofadr : self._red_dofadr + 3
    ].copy()
    future_red = self._red_pos() + REWARD_LOOKAHEAD_S * red_vel_world
    return float(np.linalg.norm(self._blue_pos() - future_red))
```

`reset()` seeds both caches from this helper (no `learner_id` branch). `step()`
rolls `prev = curr` then `curr = self._dist_blue_to_future_red()` (no
`learner_id` branch). `StepState` is then always constructed with real values
in two-policy mode. The `learner_id` instance var and its **obs-spec** uses
(`_spec_for_agent`) and the **`closing_rate` obs gate** are left exactly as they
are (Q2) — only the intercept-shaping reward inputs are de-gated.

Backward compatibility: the single-learner and no-learner-canary paths produce
the same `StepState` values they did before (the helper computes the same
quantity the old `defender = blue` branch did; the old `defender = red` branch
was only reachable with `learner_id == red_0`, a config that never used
`InterceptShaping`, so dropping it changes no live behavior — covered by a
back-compat test).

`conf/reward/team_selfplay_v1.yaml` re-adds `InterceptShaping` for the Blue
side (defender `blue_0`), restoring the term the comment said was "deferred to
Step 3."

### Part B — Step-3 snapshot-population league

**New files:** `rllib/league.py` (the `LeagueCallback` + the league
`policy_mapping_fn` factory + membership helpers),
`conf/league/default.yaml`, `conf/multiagent/red_blue_league.yaml`,
`conf/experiment/rllib_league.yaml`, `tests/rllib/test_league.py`.
**Modified:** `rllib/config_builder.py` (wire `LeagueCallback` + league mapping
fn when `cfg.league.enabled`).

**B1 · Module naming = source of truth.** Frozen snapshots are named
`red_pop_v{N}` / `blue_pop_v{N}` (N starting at 1). The set of module ids in the
`MultiRLModule` *is* the population membership — no separate registry. Helpers:

```python
RED_POP_RE  = re.compile(r"^red_pop_v(\d+)$")
BLUE_POP_RE = re.compile(r"^blue_pop_v(\d+)$")

def population_members(module_ids, regex):  # -> list[str] sorted by version
def next_version(module_ids, regex):        # -> int (max existing + 1, else 1)
```

**B2 · `LeagueCallback(RLlibCallback)`.**

- `on_algorithm_init(algorithm, ...)` — fires on fresh start **and** restore.
  Reads `algorithm.config.league` knobs, inspects current module ids, and
  installs the league mapping fn so it reflects whatever population was restored.
- `on_train_result(algorithm, result, ...)` — the snapshot driver. For each
  side X in {red, blue}:
  1. read `metric_X` from `result` (`red_score_rate` for red,
     `blue_prevention_rate` for blue — see B5 for the exact key path);
  2. if `metric_X >= snapshot_threshold` **and**
     `iters_since_last_snapshot_X >= min_iters_between_snapshots` **and**
     `len(X_pop) < population_cap`: take a snapshot (B3) and record the current
     training iteration as the last-snapshot iter for X.

Per-side last-snapshot iteration is held as callback instance state and reset to
the current iteration in `on_algorithm_init` (cooldown restarts on restore —
acceptable; it only delays the next snapshot by at most the cooldown).

**B3 · Snapshot mechanism (`add_module` + weight copy).**

```python
main_id = "main_red"                      # or "main_blue"
new_id  = f"red_pop_v{next_version(...)}"  # or blue
spec    = MultiRLModuleSpec.from_module(algorithm.get_module()) \
              .rl_module_specs[main_id]    # spec matching the live main
algorithm.add_module(
    module_id=new_id,
    module_spec=spec,
    new_should_module_be_updated=["main_red", "main_blue"],  # frozen excluded
    new_agent_to_module_mapping_fn=make_league_mapping_fn(
        algorithm.get_module_ids() | {new_id}, cfg.league
    ),
    add_to_learners=True,
    add_to_env_runners=True,
    add_to_eval_env_runners=True,
)
# Copy the live main's weights into the fresh frozen module:
main_state = algorithm.get_module(main_id).get_state()
algorithm.set_state({
    "learner_group": {"learner": {"rl_module": {new_id: main_state}}}
})
```

`add_module` creates a freshly-*initialized* module; the `set_state` step copies
the live main's current weights into it so the snapshot captures the policy at
threshold-crossing time. `new_should_module_be_updated=["main_red","main_blue"]`
keeps every frozen snapshot out of the gradient update (RLlib's
league-example pattern). The exact `set_state` nesting is verified empirically
in the implementation (the "snapshot equals main at snapshot time, then frozen"
test is the gate); if the nested-dict form is rejected by this Ray version, the
fallback is `algorithm.get_module(new_id).set_state(main_state)` broadcast to
env runners via `algorithm.env_runner_group.foreach_env_runner(...)`.

**B4 · League `policy_mapping_fn`.** A factory closing over the current
membership lists + `live_fraction`, returning a fn whose per-episode choice is
seeded on `episode.id_` so both agents in an episode get a coherent matchup:

```python
def make_league_mapping_fn(module_ids, league_cfg):
    red_pop  = population_members(module_ids, RED_POP_RE)
    blue_pop = population_members(module_ids, BLUE_POP_RE)
    live_fraction = league_cfg.live_fraction

    def league_mapping_fn(agent_id, episode, **kw):
        rng = random.Random(hash(episode.id_))   # same draw for both agents
        roll = rng.random()
        if roll < live_fraction or (not red_pop and not blue_pop):
            mode = "live"
        elif rng.random() < 0.5:
            mode = "red_exploits" if blue_pop else "live"   # red vs frozen blue
        else:
            mode = "blue_exploits" if red_pop else "live"   # frozen red vs blue
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        if mode == "red_exploits":
            return "main_red" if agent_id == "red_0" else rng.choice(blue_pop)
        # blue_exploits
        return rng.choice(red_pop) if agent_id == "red_0" else "main_blue"
    return league_mapping_fn
```

(The mapping fn must be picklable to cross the Ray boundary — it closes over
plain lists/floats only, no algorithm refs.)

**B5 · Metric key paths.** `ScoreMetricsCallback` logs `red_score_rate` /
`blue_prevention_rate` via `metrics_logger.log_value(..., reduce="mean")`, which
surfaces them under the env-runner results subtree of the `on_train_result`
`result` dict. The implementation resolves the exact key path against a real
one-iteration `result` (the first plan task prints the dict) rather than
guessing; a small `read_metric(result, name)` helper isolates that lookup so the
path lives in one place and is unit-testable.

**B6 · Config.**

`conf/league/default.yaml`:
```yaml
enabled: true
snapshot_threshold: 0.7
min_iters_between_snapshots: 20
population_cap: 5
live_fraction: 0.5
```

`conf/multiagent/red_blue_league.yaml` — same two trainable mains as
`red_blue_selfplay.yaml`, but the league mapping fn + `LeagueCallback` are
selected by `config_builder` when `cfg.league.enabled` (the static mapping dict
is the fallback when league is disabled, so Step-2 configs are unaffected).

`conf/experiment/rllib_league.yaml` — composes `obs=duel_v1_body_n1`,
`reward=team_selfplay_v1` (now with `InterceptShaping`),
`curriculum=fixed_red_start`, `multiagent=red_blue_league`, `league=default`.

`config_builder.py`: when `cfg.get("league")` is present and
`cfg.league.enabled`, append `LeagueCallback` to the callbacks and pass the
league mapping fn into `.multi_agent(...)`; otherwise keep the existing static
mapping + `ScoreMetricsCallback`-only behavior unchanged.

### macOS / Ray constraints (unchanged, carried from prior steps)

`KMP_DUPLICATE_LIB_OK=TRUE` + `MUJOCO_GL` propagation via `ray_init_for_project`
(`rllib/runtime.py`) already handle the env-runner subprocess boundary. The
league mapping fn and callback add no new cross-process state beyond the
picklable mapping-fn closure. `==`-not-`is` discipline holds (no new identity
checks introduced).

## Data flow

```
train_rllib.py
  └─ build_ppo_config(cfg, reward_stack)          # config_builder.py
       ├─ .multi_agent(policy_mapping_fn = league fn if cfg.league.enabled
       │                                   else static dict)
       ├─ .callbacks([ScoreMetricsCallback, LeagueCallback?])
       └─ .rl_module(MultiRLModuleSpec{main_red, main_blue})   # pops start empty

Tune loop, each iteration:
  env runners roll out episodes
     └─ league_mapping_fn(agent_id, episode)  # per-episode matchup, seeded on id
          └─ QuidditchTeamEnv.step → RewardStack.compute_step(StepState)
               └─ InterceptShaping(blue_0)  ← now non-zero (Part A)
  ScoreMetricsCallback aggregates red_score_rate / blue_prevention_rate
  LeagueCallback.on_train_result(result)
     └─ if metric ≥ threshold & cooldown elapsed & pop not full:
          add_module(red_pop_vN / blue_pop_vN) + copy weights + reinstall mapping fn

checkpoint: RLlib persists ALL modules (mains + frozen snapshots)
restore: LeagueCallback.on_algorithm_init derives membership from module ids,
         reinstalls the league mapping fn  →  faithful, no separate state file
```

## Testing strategy

**Part A (unit, fast):**
- `test_intercept_inputs_populated_without_learner_id`: build `QuidditchTeamEnv`
  with `learner_id=None`; step with Red carrying world-frame velocity toward the
  hoop and Blue positioned to close; assert `StepState.dist_def_to_future_red`
  is non-zero and `dist_def_to_future_red_prev` differs from it across two steps.
- `test_intercept_shaping_rewards_blue_in_two_policy`: with the two-policy
  `StepState` from above and Red inside `activation_dist`, assert
  `InterceptShaping.compute(state)["blue_0"] > 0` and `["red_0"] == 0`.
- `test_single_learner_intercept_unchanged`: with `learner_id="blue_0"`, the
  produced `dist_def_to_future_red` matches the pre-refactor value (back-compat).
- `test_team_selfplay_v1_stack_composition` (modify): assert the term list now
  **includes** `InterceptShaping` and that it targets `blue_0`.

**Part B (unit, fast):**
- `test_population_helpers`: `population_members` / `next_version` over a mixed
  module-id set (sorted by version; next = max+1; empty ⇒ 1).
- `test_league_mapping_live_only_when_pops_empty`: empty pops ⇒ always live.
- `test_league_mapping_respects_live_fraction`: over many episode ids, the
  empirical live-vs-exploit split matches `live_fraction` within tolerance; both
  agents in one episode get a coherent matchup; exploit modes draw uniformly.
- `test_league_mapping_empty_pop_falls_back_to_live`: one pop empty ⇒ that
  exploit mode degrades to live.
- `test_snapshot_trigger_threshold_and_cooldown`: a pure-logic harness over the
  trigger predicate (fires at/above threshold; suppressed during cooldown;
  suppressed at cap).
- `test_read_metric_key_path`: `read_metric` extracts `red_score_rate` /
  `blue_prevention_rate` from a representative `result` dict.

**Part B (integration, `@pytest.mark.slow`):**
- `test_league_snapshots_and_restores`: a few-iteration league run with the
  threshold lowered so a snapshot fires; assert a `*_pop_v1` module appears,
  its weights equal the main's at snapshot time and do **not** change across the
  next iteration, the run checkpoints, and a fresh `Algorithm.from_checkpoint`
  restores with membership reconstructed (mapping fn routes to the frozen
  module).

**macOS render tests** remain GUI-shell-only (unchanged, environmental).

## Open questions

- **OQ-1 · `set_state` nesting for the weight copy.** The exact dict path for
  `algorithm.set_state` to write a single module's weights is version-sensitive;
  resolved empirically in implementation (the freeze test is the gate), with the
  per-runner `set_state` broadcast as the documented fallback. Not a design risk.
- **OQ-2 · Metric key path.** Resolved empirically against a real `result` dict
  in the first Part-B task (isolated in `read_metric`).

## Risks

- **Mid-training `add_module` on the new API stack** is the least-exercised
  RLlib path here; the integration smoke test is the gate, and the design keeps
  the frozen modules out of `should_be_updated` exactly as RLlib's own
  league example does.
- **Cooldown reset on restore** can delay the next snapshot by up to
  `min_iters_between_snapshots` after a resume; accepted (bounded, harmless).
- **Population growth unbounded by quality** in Step 3 (cap + threshold only, no
  pruning) — by design; PFSP + pruning in Step 4 address opponent quality.
