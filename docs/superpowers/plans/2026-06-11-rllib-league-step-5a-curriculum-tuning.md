# RLlib League Step 5a — Curriculum & League Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three league-training improvements that the migration spec (§B2, §B5) and the Step-4 validation run call for: (1) **asymmetric per-side snapshot thresholds** (cure "Red never snapshots"), (2) a **dense→sparse reward anneal** schedule, and (3) re-add the **difficulty levers** `red_action_scale` and `red_start_r_max` and anneal them with a step schedule.

**Architecture:** A new `rllib/curriculum.py` holds a pure `scheduled_value(schedule, t)` linear-interpolator and a thin `CurriculumCallback(RLlibCallback)` that, each `on_train_result`, computes the scheduled dense-reward scale and difficulty-lever values from the lifetime step count and pushes them onto every env-runner's live `QuidditchTeamEnv` (reward stack + `TeamConfig`). `RewardStack` gains a mutable `dense_scale` that multiplies only the shaping terms (leaving the sparse outcome rewards intact). `TeamConfig` gains `red_action_scale` (throttles Red's per-step setpoint delta) and `red_start_r_max` (caps the random-start disc), both read *live* in `step`/`_sample_red_start` so a runtime push takes effect on the next step/episode. The snapshot gate reads a per-side threshold.

**Tech Stack:** Python 3.13, Ray RLlib new API stack, Hydra `conf/` tree + `@dataclass` schemas, MuJoCo env, pytest (`@pytest.mark.slow` for Ray/MuJoCo). Run via `uv run` (sandbox disabled).

**Sequencing — execute AFTER Step 5b.** 5b re-points the snapshot gate at the clean `eval_*` metric (preferring `eval_<metric>` over the windowed metric). 5a's asymmetric threshold is applied at the same `should_snapshot(threshold=…)` call, so it composes cleanly: per-side threshold against the per-side clean metric. The reward anneal and difficulty levers are independent of 5b. If 5b has not landed, 5a still works (the gate falls back to the windowed metric) — only the *quality* of the gating signal differs.

---

## Orientation for the implementer

Read these before starting:

- [rllib/league.py](../../../rllib/league.py) — `LeagueCallback.on_train_result`. The snapshot trigger (lines 374–381) reads a single shared `self._cfg["snapshot_threshold"]` (line 376) for both sides via the `_SIDES` loop (lines 290–293, 364). Task 1 makes this per-side.
- [conf/league/default.yaml](../../../conf/league/default.yaml) — `snapshot_threshold: 0.7` (line 23) is shared. The callback reads the league dict from `algorithm.config.env_config["league"]`, so any new key here propagates to the callback with no builder change (it accesses via `self._cfg.get(...)`).
- [envs/quidditch/rewards/stack.py](../../../envs/quidditch/rewards/stack.py) — `RewardStack` is a plain `@dataclass(terms: list)`; `compute_step` (lines 82–87) sums `term.compute(state)` per agent. No scaling, no mutation today. `StepState` is `frozen=True`. Task 2 adds a mutable `dense_scale`.
- [envs/quidditch/rewards/terms.py](../../../envs/quidditch/rewards/terms.py) — every term is a stateless `@dataclass`. Dense shaping terms: `HoopApproachShaping`, `HoopDistancePenalty`, `HoopAnchor`, `ZeroSumDistMirror`, `InterceptShaping`, `GoalSideCone`, `ProximityGradedTag`, `ClosingVelInTagZone`, `TagEntryPulse`. Sparse outcome terms (never annealed): `ScoreEvent`, `TakeDown`, `CrashEvent`.
- [envs/quidditch/rewards/__init__.py](../../../envs/quidditch/rewards/__init__.py) — `load_reward_stack` is `@lru_cache`'d, so one `RewardStack` instance is shared per process. With RLlib's default 1 env per runner, each runner process has its own instance — runtime mutation is safe per-runner.
- [envs/quidditch/team_env.py](../../../envs/quidditch/team_env.py) — `TeamConfig` (lines 83–104). `ACTION_SCALE` constant (line 67). The step action loop applies `delta = action * ACTION_SCALE` at line 339 (and aftermath at 543). `_sample_red_start` (lines 601–611) draws `r` from `[0, START_SAMPLE_RADIUS]` (= 2.9) with no cap. `START_SAMPLE_RADIUS` is defined at line 69.
- [rllib/config_builder.py](../../../rllib/config_builder.py) — `_team_cfg_from(cfg)` (lines 29–59) folds `cfg.curriculum` into `env_config["team_cfg"]`. `build_ppo_config` assembles `env_config` at lines 114–122 and the callback list at lines 99/106. **Note: `cfg.curriculum` is NOT currently passed to the callback** — Task 5 adds an `env_config["curriculum"]` block carrying the schedules.
- [config_schema.py](../../../config_schema.py) — `CurriculumConfig` (lines 85–92): `randomise_start`, `episode_seconds`, `red_start_pos`, `red_start_yaw`. Registered at line 155. Hydra struct-mode validates curriculum YAMLs against it, so new YAML keys require new dataclass fields. (`league`/`algo`/`multiagent`/`tune` have NO dataclass schema — free-form.)
- [conf/curriculum/fixed_red_start.yaml](../../../conf/curriculum/fixed_red_start.yaml) — the league's current curriculum (fixed airborne Red start at `[0.5, 0, 2]`, `randomise_start: false`).
- [tests/rllib/test_league.py](../../../tests/rllib/test_league.py) — `_FakeAlgo` (lines 233–258), `_LEAGUE_CFG` (lines 261–264), and the callback test patterns. Task 1 extends these.

**Test/run commands** (repo root, sandbox disabled):
- `uv run python -m pytest tests/rllib/test_curriculum.py -v`
- `uv run python -m pytest -m "not slow"` (fast) / `-m slow` (integration)

---

## File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `rllib/league.py` | Per-side snapshot threshold in the gate | **Modify** (line 376) |
| `conf/league/default.yaml` | `snapshot_threshold_red` / `snapshot_threshold_blue` | **Modify** |
| `envs/quidditch/rewards/stack.py` | `RewardStack.dense_scale` + `set_dense_scale` + dense-only scaling | **Modify** |
| `rllib/curriculum.py` | Pure `scheduled_value` + `CurriculumCallback` + env-push seam | **Create** |
| `envs/quidditch/team_env.py` | `TeamConfig.red_action_scale` / `red_start_r_max`, applied live | **Modify** (TeamConfig, step, `_sample_red_start`) |
| `config_schema.py` | `CurriculumConfig` static levers + schedule fields | **Modify** |
| `rllib/config_builder.py` | Thread levers + schedules; register `CurriculumCallback` | **Modify** (`_team_cfg_from`, env_config, callbacks) |
| `conf/curriculum/league_anneal.yaml` | Curriculum with anneal schedules | **Create** |
| `conf/experiment/rllib_league_step5.yaml` | Step-5 league experiment composing the above | **Create** |
| `tests/rllib/test_curriculum.py` | Unit tests for scheduler + callback + RewardStack scaling | **Create** |
| `tests/rllib/test_league.py` | Asymmetric-threshold tests | **Modify** |
| `tests/envs/quidditch/test_team_env_features.py` | Difficulty-lever env contract | **Modify** |
| `tests/rllib/test_smoke_train.py` | Slow integration: anneal reaches the env | **Modify** |

---

## Task 1: Asymmetric per-side snapshot thresholds

Replace the single shared `snapshot_threshold` read in the snapshot gate with a per-side lookup `snapshot_threshold_<side>`, falling back to the shared value. Cures the Step-4 finding that Red never snapshots (its score-rate vs a competent Blue rarely holds ≥ 0.7).

**Files:**
- Modify: [rllib/league.py](../../../rllib/league.py) (line 376)
- Modify: [conf/league/default.yaml](../../../conf/league/default.yaml)
- Test: [tests/rllib/test_league.py](../../../tests/rllib/test_league.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_league.py`:

```python
def _asym_cfg(**over):
    return {**_LEAGUE_CFG, "snapshot_threshold_red": 0.4,
            "snapshot_threshold_blue": 0.7, **over}


def test_per_side_snapshot_threshold(monkeypatch):
    """Red uses snapshot_threshold_red (0.4); Blue uses snapshot_threshold_blue
    (0.7). A metric of 0.5 snapshots Red but not Blue."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_asym_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "red_score_rate": 0.5, "blue_prevention_rate": 0.5}})
    assert "red_pop_v1" in algo.added        # 0.5 >= 0.4 (red threshold)
    assert "blue_pop_v1" not in algo.added   # 0.5 < 0.7 (blue threshold)


def test_per_side_threshold_falls_back_to_shared(monkeypatch):
    """Without per-side keys, both sides use the shared snapshot_threshold."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "red_score_rate": 0.75, "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added        # 0.75 >= shared 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_league.py::test_per_side_snapshot_threshold -v`
Expected: FAIL — currently both sides use the shared 0.7, so Red does NOT snapshot at 0.5.

- [ ] **Step 3: Read the per-side threshold in the gate**

In `rllib/league.py`, `on_train_result`, change the `should_snapshot(...)` call's `threshold=` argument (currently line 376):

```python
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg.get(
                    f"snapshot_threshold_{side}", self._cfg["snapshot_threshold"])),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(pop) - (1 if victim else 0),
                cap=int(self._cfg["population_cap"]),
            ):
```

- [ ] **Step 4: Document the knob in the league config**

Append to `conf/league/default.yaml`:

```yaml
# Per-side snapshot thresholds (Step 5a). Asymmetric roles need asymmetric bars:
# the Step-4 run showed Red's score-rate rarely holds >= 0.7 against a competent
# Blue, so Red never snapshotted. Absent keys fall back to snapshot_threshold.
snapshot_threshold_red: 0.7
snapshot_threshold_blue: 0.7
```

(The Step-5 experiment YAML in Task 8 overrides `snapshot_threshold_red` to a lower value; the default stays at the shared 0.7 so existing runs are unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_league.py -v`
Expected: PASS (both new tests + all existing league tests — fallback preserves Step-4 behavior).

- [ ] **Step 6: Commit**

```bash
git add rllib/league.py conf/league/default.yaml tests/rllib/test_league.py
git commit -m "feat(rllib): per-side asymmetric snapshot thresholds"
```

---

## Task 2: `RewardStack.dense_scale` — anneal only the shaping terms

Give `RewardStack` a mutable `dense_scale` (default 1.0) that multiplies only the dense **shaping** terms in `compute_step`, leaving the sparse outcome terms (`ScoreEvent`, `TakeDown`, `CrashEvent`) at full magnitude. `dense_scale=1.0` is byte-identical to today (canary preserved).

**Files:**
- Modify: [envs/quidditch/rewards/stack.py](../../../envs/quidditch/rewards/stack.py)
- Test: [tests/rllib/test_curriculum.py](../../../tests/rllib/test_curriculum.py)

- [ ] **Step 1: Write the failing test**

Create `tests/rllib/test_curriculum.py`:

```python
"""Step-5a curriculum: reward-anneal scaling, scheduler, and CurriculumCallback."""
from __future__ import annotations

from envs.quidditch.rewards.stack import RewardStack, StepState
from envs.quidditch.rewards.terms import HoopApproachShaping, ScoreEvent


def _state(scored: bool, prev: float, cur: float) -> StepState:
    return StepState(agent_ids=("red_0", "blue_0"), scored=scored,
                     dist_red_to_hoop=cur, dist_red_to_hoop_prev=prev)


def test_dense_scale_scales_shaping_not_outcomes():
    stack = RewardStack(terms=[
        HoopApproachShaping(scale=2.0, agent="red_0"),   # dense -> scaled
        ScoreEvent(magnitude=10.0, scorer="red_0", zero_sum_opponent="blue_0"),
    ])
    # progress 1.0 -> shaping = 2.0; score -> +10 red / -10 blue.
    full = stack.compute_step(_state(scored=True, prev=2.0, cur=1.0))
    assert abs(full["red_0"] - (2.0 + 10.0)) < 1e-9

    stack.set_dense_scale(0.5)
    half = stack.compute_step(_state(scored=True, prev=2.0, cur=1.0))
    # shaping halved (1.0); score untouched (+10) -> 11.0 red
    assert abs(half["red_0"] - (1.0 + 10.0)) < 1e-9
    # Blue's zero-sum score mirror (-10) is a sparse outcome, NOT scaled.
    assert abs(half["blue_0"] - (-10.0)) < 1e-9


def test_dense_scale_zero_removes_all_shaping():
    stack = RewardStack(terms=[HoopApproachShaping(scale=2.0, agent="red_0")])
    stack.set_dense_scale(0.0)
    out = stack.compute_step(_state(scored=False, prev=2.0, cur=1.0))
    assert out["red_0"] == 0.0


def test_default_dense_scale_is_identity():
    stack = RewardStack(terms=[HoopApproachShaping(scale=2.0, agent="red_0")])
    assert stack.dense_scale == 1.0
    out = stack.compute_step(_state(scored=False, prev=2.0, cur=1.0))
    assert abs(out["red_0"] - 2.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py -v`
Expected: FAIL with `AttributeError: 'RewardStack' object has no attribute 'set_dense_scale'`.

- [ ] **Step 3: Add dense scaling to `RewardStack`**

In `envs/quidditch/rewards/stack.py`, replace the `RewardStack` dataclass (lines 77–87) with:

```python
# Dense SHAPING terms (annealed by dense_scale). The sparse OUTCOME terms
# (ScoreEvent, TakeDown, CrashEvent) are never annealed — they encode the true
# objective and must keep full magnitude through the dense->sparse transition.
_DENSE_TERM_TYPES = frozenset({
    "HoopApproachShaping", "HoopDistancePenalty", "HoopAnchor",
    "ZeroSumDistMirror", "InterceptShaping", "GoalSideCone",
    "ProximityGradedTag", "ClosingVelInTagZone", "TagEntryPulse",
})


@dataclass
class RewardStack:
    """Ordered reward terms; accumulates per-agent rewards per step.

    `dense_scale` (default 1.0, identity) multiplies the dense shaping terms
    only — the Step-5a dense->sparse anneal. CurriculumCallback mutates it at
    runtime via set_dense_scale. Sparse outcome terms are unaffected.
    """
    terms: list[Any]
    dense_scale: float = 1.0

    def set_dense_scale(self, scale: float) -> None:
        self.dense_scale = float(scale)

    def compute_step(self, state: StepState) -> dict[str, float]:
        totals: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for term in self.terms:
            scale = (self.dense_scale
                     if type(term).__name__ in _DENSE_TERM_TYPES else 1.0)
            for agent, r in term.compute(state).items():
                totals[agent] += scale * r
        return totals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Confirm reward canaries still hold (dense_scale defaults to 1.0)**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_canary.py tests/envs/quidditch/test_scoring_canary.py -v`
Expected: PASS — identity scaling leaves rewards unchanged.

- [ ] **Step 6: Commit**

```bash
git add envs/quidditch/rewards/stack.py tests/rllib/test_curriculum.py
git commit -m "feat(rewards): dense_scale anneals shaping terms, not outcomes"
```

---

## Task 3: Pure `scheduled_value` interpolator

Add a pure linear-interpolation scheduler to a new `rllib/curriculum.py`. Same `[[t, v], …]` shape RLlib uses for `entropy_coeff_schedule`. Clamps below the first knot and above the last.

**Files:**
- Create: [rllib/curriculum.py](../../../rllib/curriculum.py)
- Test: [tests/rllib/test_curriculum.py](../../../tests/rllib/test_curriculum.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_curriculum.py`:

```python
import rllib.curriculum as C


def test_scheduled_value_linear_interpolation():
    sched = [[0, 1.0], [100, 0.0]]
    assert C.scheduled_value(sched, 0) == 1.0
    assert abs(C.scheduled_value(sched, 50) - 0.5) < 1e-9
    assert C.scheduled_value(sched, 100) == 0.0


def test_scheduled_value_clamps_outside_range():
    sched = [[10, 0.5], [20, 1.0]]
    assert C.scheduled_value(sched, 0) == 0.5     # before first knot -> first value
    assert C.scheduled_value(sched, 999) == 1.0   # past last knot -> last value


def test_scheduled_value_none_and_constant():
    assert C.scheduled_value(None, 5) is None          # no schedule -> no override
    assert C.scheduled_value([[0, 0.7]], 999) == 0.7   # single knot -> constant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py::test_scheduled_value_linear_interpolation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rllib.curriculum'`.

- [ ] **Step 3: Implement the scheduler**

Create `rllib/curriculum.py`:

```python
"""Step-5a league curriculum: dense->sparse reward anneal + difficulty levers.

A pure scheduled_value interpolator + a thin CurriculumCallback that pushes the
scheduled reward dense-scale and difficulty-lever values onto every env-runner's
live QuidditchTeamEnv each on_train_result. The env reads red_action_scale /
red_start_r_max live (in step / _sample_red_start) and dense_scale live (in
RewardStack.compute_step), so a runtime push takes effect on the next step.
"""
from __future__ import annotations

from typing import Optional

from ray.rllib.callbacks.callbacks import RLlibCallback


def scheduled_value(schedule, t) -> Optional[float]:
    """Piecewise-linear interpolation of a [[t0, v0], [t1, v1], ...] schedule.

    Clamps to the first value below t0 and the last value above the final knot.
    Returns None when `schedule` is falsy (no override configured), so callers
    can distinguish "leave the field alone" from "set it to 0".
    """
    if not schedule:
        return None
    knots = [(float(t_i), float(v_i)) for t_i, v_i in schedule]
    knots.sort()
    t = float(t)
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    for (t0, v0), (t1, v1) in zip(knots, knots[1:]):
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return v0 + frac * (v1 - v0)
    return knots[-1][1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py -k scheduled_value -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/curriculum.py tests/rllib/test_curriculum.py
git commit -m "feat(rllib): pure scheduled_value interpolator for curriculum"
```

---

## Task 4: Difficulty levers in the env — `red_action_scale` + `red_start_r_max`

Add the two levers to `TeamConfig` and read them **live** in the step loop and start sampler so a runtime push takes effect immediately. `red_action_scale` (default 1.0) multiplies Red's per-step setpoint delta — throttling Red's action authority. `red_start_r_max` (default `START_SAMPLE_RADIUS`) caps the random-start disc radius. Both default to no behavior change.

**Files:**
- Modify: [envs/quidditch/team_env.py](../../../envs/quidditch/team_env.py) (`TeamConfig`, step loop line 339, `_sample_red_start` line 607)
- Test: [tests/envs/quidditch/test_team_env_features.py](../../../tests/envs/quidditch/test_team_env_features.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/envs/quidditch/test_team_env_features.py`:

```python
def test_red_action_scale_throttles_red_setpoint_delta():
    """red_action_scale halves Red's setpoint movement but leaves Blue's full."""
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig, ACTION_SCALE

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False,
                                          red_action_scale=0.5))
    env.reset(seed=0)
    sp_red0 = env._setpoint_red.copy()
    sp_blue0 = env._setpoint_blue.copy()
    a = np.ones(4, dtype=np.float32)
    env.step({"red_0": a, "blue_0": a})
    # Red moved by 0.5 * ACTION_SCALE; Blue by full ACTION_SCALE (x/y unclamped here).
    assert np.allclose(env._setpoint_red[:2] - sp_red0[:2], 0.5 * ACTION_SCALE[:2])
    assert np.allclose(env._setpoint_blue[:2] - sp_blue0[:2], ACTION_SCALE[:2])
    env.close()


def test_red_start_r_max_caps_random_disc():
    """With randomise_red_start and a small r_max, every sampled start sits
    within the cap (not the full 2.9 m disc)."""
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=True,
                                          red_start_r_max=0.5))
    for seed in range(20):
        pos, _ = env._sample_red_start() if False else (None, None)  # see note
        env._np_random = np.random.default_rng(seed)
        pos, _ = env._sample_red_start()
        assert float(np.linalg.norm(pos[:2])) <= 0.5 + 1e-9
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_features.py::test_red_action_scale_throttles_red_setpoint_delta -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'red_action_scale'`.

- [ ] **Step 3: Add the fields to `TeamConfig`**

In `envs/quidditch/team_env.py`, add to the `TeamConfig` dataclass (after `red_start_yaw`, line 97):

```python
    red_start_yaw: float = 0.0
    # Difficulty levers (Step 5a, annealable via CurriculumCallback):
    #   red_action_scale: multiplies Red's per-step setpoint delta (action
    #     authority). 1.0 = full; lower = a slower/weaker attacker.
    #   red_start_r_max: caps the random-start disc radius (only consulted when
    #     randomise_red_start). None -> full START_SAMPLE_RADIUS (no cap).
    red_action_scale: float = 1.0
    red_start_r_max: float | None = None
    episode_seconds: float = EPISODE_SECONDS_DEFAULT
```

- [ ] **Step 4: Apply `red_action_scale` in the step loop**

In `step` (the action loop at lines 338–353), scale Red's delta by the live `red_action_scale`:

```python
        for agent_id, action in actions.items():
            delta = np.asarray(action, dtype=np.float32) * ACTION_SCALE
            if agent_id == self._red_id:
                delta = delta * self.cfg.red_action_scale
                self._setpoint_red += delta
                self._setpoint_red[0] = np.clip(self._setpoint_red[0], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_red[1] = np.clip(self._setpoint_red[1], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_red[2] = (self._setpoint_red[2] + np.pi) % (2 * np.pi) - np.pi
                self._setpoint_red[3] = np.clip(self._setpoint_red[3], 0.01, 4.0)
                self._red.set_setpoint(self._setpoint_red)
            else:
                self._setpoint_blue += delta
                self._setpoint_blue[0] = np.clip(self._setpoint_blue[0], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_blue[1] = np.clip(self._setpoint_blue[1], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_blue[2] = (self._setpoint_blue[2] + np.pi) % (2 * np.pi) - np.pi
                self._setpoint_blue[3] = np.clip(self._setpoint_blue[3], 0.01, 4.0)
                self._blue.set_setpoint(self._setpoint_blue)
```

- [ ] **Step 5: Apply `red_start_r_max` in the start sampler**

In `_sample_red_start` (lines 601–611), cap the radius:

```python
    def _sample_red_start(self) -> tuple[np.ndarray, float]:
        if not self.cfg.randomise_red_start:
            fixed = self.cfg.red_start_pos
            pos = np.array(fixed if fixed is not None else (0.0, 0.0, 0.0),
                           dtype=np.float64)
            return pos, float(self.cfg.red_start_yaw)
        r_max = START_SAMPLE_RADIUS
        if self.cfg.red_start_r_max is not None:
            r_max = min(r_max, float(self.cfg.red_start_r_max))
        r = r_max * float(np.sqrt(self._np_random.uniform(0.0, 1.0)))
        theta = float(self._np_random.uniform(0.0, 2.0 * np.pi))
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=np.float64)
        yaw = float(self._np_random.uniform(-np.pi, np.pi))
        return pos, yaw
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_features.py -k "red_action_scale or red_start_r_max" -v`
Expected: PASS (both tests).

- [ ] **Step 7: Confirm team canary still holds (defaults are no-ops)**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_canary.py -v`
Expected: PASS (`red_action_scale=1.0`, `red_start_r_max=None` → unchanged dynamics).

- [ ] **Step 8: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_team_env_features.py
git commit -m "feat(team-env): red_action_scale + red_start_r_max difficulty levers"
```

---

## Task 5: Thread levers + schedules through Hydra → env_config

Add the static levers and the three schedules to `CurriculumConfig`, thread the static levers into `env_config["team_cfg"]` via `_team_cfg_from`, and add an `env_config["curriculum"]` block carrying the schedules so `CurriculumCallback` can read them off `algorithm.config.env_config`.

**Files:**
- Modify: [config_schema.py](../../../config_schema.py) (`CurriculumConfig`)
- Modify: [rllib/config_builder.py](../../../rllib/config_builder.py) (`_team_cfg_from`, env_config)
- Test: [tests/rllib/test_config_builder.py](../../../tests/rllib/test_config_builder.py), [tests/test_config_schema.py](../../../tests/test_config_schema.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_config_builder.py`:

```python
def test_curriculum_levers_and_schedules_thread_into_env_config():
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config

    cfg = _league_cfg()  # existing league-enabled inline cfg factory in this file
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"curriculum": {
        "randomise_start": False, "episode_seconds": 30.0,
        "red_start_pos": [0.5, 0.0, 2.0], "red_start_yaw": 0.0,
        "red_action_scale": 0.6, "red_start_r_max": None,
        "dense_scale_schedule": [[0, 1.0], [1000, 0.0]],
        "red_action_scale_schedule": [[0, 0.6], [1000, 1.0]],
        "red_start_r_max_schedule": None,
    }}))
    config = build_ppo_config(cfg)
    ec = config.env_config
    # Static lever lands in team_cfg (env construction reads it).
    assert ec["team_cfg"]["red_action_scale"] == 0.6
    # Schedules land in a dedicated curriculum block (the callback reads it).
    assert ec["curriculum"]["dense_scale_schedule"] == [[0, 1.0], [1000, 0.0]]
    assert ec["curriculum"]["red_action_scale_schedule"] == [[0, 0.6], [1000, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py::test_curriculum_levers_and_schedules_thread_into_env_config -v`
Expected: FAIL — `team_cfg` has no `red_action_scale`, and `env_config` has no `curriculum` key.

- [ ] **Step 3: Extend `CurriculumConfig`**

In `config_schema.py`, replace the `CurriculumConfig` dataclass (lines 85–92):

```python
@dataclass
class CurriculumConfig:
    randomise_start: bool = True
    episode_seconds: float = 30.0
    # Fixed-start lever (only read when randomise_start is False): explicit Red
    # spawn [x, y, z] and yaw.  None → origin (legacy fixed-start behavior).
    red_start_pos: list[float] | None = None
    red_start_yaw: float = 0.0
    # Difficulty levers (Step 5a) — static initial values; schedules below anneal
    # them at runtime. red_start_r_max=None → full random-start disc.
    red_action_scale: float = 1.0
    red_start_r_max: float | None = None
    # Anneal schedules as [[timestep, value], ...] (RLlib's schedule shape).
    # None → no anneal (the static value above holds for the whole run).
    dense_scale_schedule: list[list[float]] | None = None
    red_action_scale_schedule: list[list[float]] | None = None
    red_start_r_max_schedule: list[list[float]] | None = None
```

- [ ] **Step 4: Thread the static lever + add the curriculum block in the builder**

In `rllib/config_builder.py`, extend `_team_cfg_from` (the `cur is not None` branch, after line 58):

```python
    cur = cfg.get("curriculum")
    if cur is not None:
        out["randomise_red_start"] = bool(cur.randomise_start)
        out["episode_seconds"] = float(cur.episode_seconds)
        rsp = cur.get("red_start_pos")
        if rsp is not None:
            out["red_start_pos"] = [float(v) for v in rsp]
        out["red_start_yaw"] = float(cur.get("red_start_yaw") or 0.0)
        out["red_action_scale"] = float(cur.get("red_action_scale") or 1.0)
        rrm = cur.get("red_start_r_max")
        out["red_start_r_max"] = float(rrm) if rrm is not None else None
    return out
```

Add a `_curriculum_dict_from` helper near `_team_cfg_from` (after line 59):

```python
def _curriculum_dict_from(cfg: DictConfig) -> dict:
    """Anneal schedules for CurriculumCallback. Plain Python (crosses the Ray
    boundary into env_config). Empty when no curriculum group is composed."""
    cur = cfg.get("curriculum")
    if cur is None:
        return {}
    out: dict = {}
    for key in ("dense_scale_schedule", "red_action_scale_schedule",
                "red_start_r_max_schedule"):
        sched = cur.get(key)
        if sched is not None:
            out[key] = [[float(t), float(v)] for t, v in sched]
    return out
```

In `build_ppo_config`, add the curriculum block to `env_config` (lines 116–122):

```python
        .environment(
            _ENV_NAME,
            env_config={
                "learner_id": ma.learner_id,
                "obs_blocks": obs_blocks,
                "team_cfg": _team_cfg_from(cfg),
                "reward_stack": reward_stack,
                "league": league_dict,
                "curriculum": _curriculum_dict_from(cfg),
            },
        )
```

- [ ] **Step 5: Add a schema test for the new fields**

Append to `tests/test_config_schema.py`:

```python
def test_curriculum_schema_has_difficulty_levers_and_schedules():
    from config_schema import CurriculumConfig
    c = CurriculumConfig()
    assert c.red_action_scale == 1.0
    assert c.red_start_r_max is None
    assert c.dense_scale_schedule is None
    assert c.red_action_scale_schedule is None
    assert c.red_start_r_max_schedule is None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py tests/test_config_schema.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add config_schema.py rllib/config_builder.py tests/rllib/test_config_builder.py tests/test_config_schema.py
git commit -m "feat(conf): thread difficulty levers + anneal schedules into env_config"
```

---

## Task 6: `CurriculumCallback` — push annealed values onto live envs

Add `CurriculumCallback(RLlibCallback)` to `rllib/curriculum.py`. Each `on_train_result`, it reads the lifetime step count, computes each scheduled value, and pushes them onto every env-runner's live `QuidditchTeamEnv` (reward stack `dense_scale` + `TeamConfig.red_action_scale`/`red_start_r_max`). The env navigation is isolated behind a small, fast-tested seam.

**Files:**
- Modify: [rllib/curriculum.py](../../../rllib/curriculum.py)
- Test: [tests/rllib/test_curriculum.py](../../../tests/rllib/test_curriculum.py)

- [ ] **Step 1: Write the failing test (fakes, no MuJoCo)**

Append to `tests/rllib/test_curriculum.py`:

```python
import types


class _FakeRewardStack:
    def __init__(self):
        self.dense_scale = 1.0
    def set_dense_scale(self, s):
        self.dense_scale = s


class _FakeTeamEnv:
    def __init__(self):
        self._reward_stack = _FakeRewardStack()
        self.cfg = types.SimpleNamespace(red_action_scale=1.0, red_start_r_max=None)


class _FakeEnvRunnerGroup:
    def __init__(self, inner):
        # runner.env._inner is the QuidditchTeamEnv
        self._runner = types.SimpleNamespace(
            env=types.SimpleNamespace(_inner=inner))
    def foreach_env_runner(self, fn, **kw):
        return [fn(self._runner)]


def _curr_algo(curriculum, lifetime_steps, inner):
    return types.SimpleNamespace(
        iteration=1,
        config=types.SimpleNamespace(env_config={"curriculum": curriculum}),
        env_runner_group=_FakeEnvRunnerGroup(inner),
        env_runner=types.SimpleNamespace(),
        # result carries the lifetime step count (read by the callback)
        _lifetime=lifetime_steps,
    )


def test_curriculum_callback_pushes_scheduled_values():
    inner = _FakeTeamEnv()
    algo = _curr_algo({
        "dense_scale_schedule": [[0, 1.0], [1000, 0.0]],
        "red_action_scale_schedule": [[0, 0.5], [1000, 1.0]],
    }, lifetime_steps=500, inner=inner)
    cb = C.CurriculumCallback()
    cb.on_train_result(algorithm=algo,
                       result={"num_env_steps_sampled_lifetime": 500})
    assert abs(inner._reward_stack.dense_scale - 0.5) < 1e-9   # halfway anneal
    assert abs(inner.cfg.red_action_scale - 0.75) < 1e-9       # halfway 0.5->1.0


def test_curriculum_callback_noop_without_schedules():
    inner = _FakeTeamEnv()
    algo = _curr_algo({}, lifetime_steps=500, inner=inner)
    cb = C.CurriculumCallback()
    cb.on_train_result(algorithm=algo,
                       result={"num_env_steps_sampled_lifetime": 500})
    assert inner._reward_stack.dense_scale == 1.0   # untouched
    assert inner.cfg.red_action_scale == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py::test_curriculum_callback_pushes_scheduled_values -v`
Expected: FAIL with `AttributeError: ... no attribute 'CurriculumCallback'`.

- [ ] **Step 3: Implement the callback + push seam**

Append to `rllib/curriculum.py`:

```python
def _read_lifetime_steps(result: dict) -> float:
    """Lifetime env-steps sampled, the schedule's time axis. Recursive so it
    works wherever RLlib nests the counter."""
    key = "num_env_steps_sampled_lifetime"
    if key in result and isinstance(result[key], (int, float)):
        return float(result[key])
    for v in result.values():
        if isinstance(v, dict):
            found = _read_lifetime_steps(v)
            if found is not None:
                return found
    return 0.0


def _inner_team_env(runner):
    """Reach the QuidditchTeamEnv from an env runner. RLlib's env runner holds
    the QuidditchMultiAgentEnv at runner.env (single env per runner); ._inner is
    the wrapped QuidditchTeamEnv. Returns None if the shape is unexpected."""
    env = getattr(runner, "env", None)
    env = getattr(env, "unwrapped", env)
    return getattr(env, "_inner", None)


def _apply_to_team_envs(algorithm, fn) -> None:
    def _set(runner, _fn=fn):
        inner = _inner_team_env(runner)
        if inner is not None:
            _fn(inner)
    algorithm.env_runner_group.foreach_env_runner(_set, local_env_runner=True)


class CurriculumCallback(RLlibCallback):
    """Pushes scheduled reward dense-scale + difficulty-lever values onto every
    env-runner's live QuidditchTeamEnv each iteration. No-op for any lever
    without a schedule (its static value holds)."""

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        cur = dict(algorithm.config.env_config.get("curriculum", {}))
        if not cur:
            return
        t = _read_lifetime_steps(result)
        dense = scheduled_value(cur.get("dense_scale_schedule"), t)
        ras = scheduled_value(cur.get("red_action_scale_schedule"), t)
        rsm = scheduled_value(cur.get("red_start_r_max_schedule"), t)
        if dense is None and ras is None and rsm is None:
            return

        def _push(inner, _d=dense, _a=ras, _r=rsm):
            if _d is not None:
                inner._reward_stack.set_dense_scale(_d)
            if _a is not None:
                inner.cfg.red_action_scale = _a
            if _r is not None:
                inner.cfg.red_start_r_max = _r

        _apply_to_team_envs(algorithm, _push)
        # Surface the live values in the result for W&B visibility.
        cd = result.setdefault("curriculum", {})
        if dense is not None:
            cd["dense_scale"] = dense
        if ras is not None:
            cd["red_action_scale"] = ras
        if rsm is not None:
            cd["red_start_r_max"] = rsm
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_curriculum.py -v`
Expected: PASS (all curriculum unit tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/curriculum.py tests/rllib/test_curriculum.py
git commit -m "feat(rllib): CurriculumCallback pushes annealed values to live envs"
```

---

## Task 7: Register `CurriculumCallback` in the config builder

Register `CurriculumCallback` in the callback list when the league is on and any curriculum schedule is configured. Place it before `LeagueCallback` (it mutates the reward/env that the next iteration samples; ordering vs the gate is not load-bearing, but keep it adjacent to the other league callbacks).

**Files:**
- Modify: [rllib/config_builder.py](../../../rllib/config_builder.py)
- Test: [tests/rllib/test_config_builder.py](../../../tests/rllib/test_config_builder.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_config_builder.py`:

```python
def test_curriculum_callback_registered_when_schedule_present():
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.curriculum import CurriculumCallback

    cfg = _league_cfg()
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"curriculum": {
        "randomise_start": False, "episode_seconds": 30.0,
        "dense_scale_schedule": [[0, 1.0], [1000, 0.0]]}}))
    cbs = build_ppo_config(cfg).callbacks_class
    cbs = list(cbs) if isinstance(cbs, (list, tuple)) else [cbs]
    assert CurriculumCallback in cbs


def test_curriculum_callback_absent_without_schedule():
    from rllib.config_builder import build_ppo_config
    from rllib.curriculum import CurriculumCallback

    cfg = _league_cfg()  # no curriculum schedules
    cbs = build_ppo_config(cfg).callbacks_class
    cbs = list(cbs) if isinstance(cbs, (list, tuple)) else [cbs]
    assert CurriculumCallback not in cbs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py::test_curriculum_callback_registered_when_schedule_present -v`
Expected: FAIL — `CurriculumCallback` is not registered.

- [ ] **Step 3: Register the callback conditionally**

In `rllib/config_builder.py`, add the import (near line 20):

```python
from rllib.curriculum import CurriculumCallback
```

In the `league_on` branch (where `callbacks` is assembled — this builds on the 5b callback list), append `CurriculumCallback` when any schedule is present:

```python
    if league_on:
        league_dict = OmegaConf.to_container(league_cfg, resolve=True)
        policy_mapping_fn = make_league_mapping_fn(set(ma.modules.keys()), league_dict)
        if league_dict.get("eval_enabled"):
            callbacks = [ScoreMetricsCallback, EvalBatteryCallback, LeagueCallback]
        else:
            callbacks = [ScoreMetricsCallback, LeagueCallback]
        if _curriculum_dict_from(cfg):
            # Insert before LeagueCallback so curriculum + gating live together.
            callbacks.insert(len(callbacks) - 1, CurriculumCallback)
```

> If executing 5a **before** 5b, the `eval_enabled` branch won't exist yet — in that case use the plain `callbacks = [ScoreMetricsCallback, LeagueCallback]` and the same `insert`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py -v`
Expected: PASS (both new tests + existing).

- [ ] **Step 5: Commit**

```bash
git add rllib/config_builder.py tests/rllib/test_config_builder.py
git commit -m "feat(rllib): register CurriculumCallback when an anneal schedule is set"
```

---

## Task 8: Curriculum + experiment YAMLs

Create a `league_anneal` curriculum (fixed airborne start + the anneal schedules) and a `rllib_league_step5` experiment composing 5b's eval battery, the asymmetric Red threshold, and the curriculum.

**Files:**
- Create: [conf/curriculum/league_anneal.yaml](../../../conf/curriculum/league_anneal.yaml)
- Create: [conf/experiment/rllib_league_step5.yaml](../../../conf/experiment/rllib_league_step5.yaml)
- Test: [tests/test_config_loading.py](../../../tests/test_config_loading.py)

- [ ] **Step 1: Write the curriculum YAML**

Create `conf/curriculum/league_anneal.yaml`:

```yaml
# Step-5a league curriculum: the fixed airborne Red start from fixed_red_start,
# plus dense->sparse reward anneal and a Red action-authority ramp.
#   dense_scale_schedule: shaping is full early (bootstraps scoring/defending),
#     annealed to zero by 3M steps so the learned policy holds on the true
#     (sparse) objective — the same dense->sparse pattern the manual campaign used.
#   red_action_scale_schedule: Red starts throttled (0.6) and ramps to full (1.0)
#     by 3M steps, so Blue faces a progressively more agile attacker as the
#     league matures (spec B5).
# red_start_r_max_schedule left null: the league trains from the fixed airborne
# start; switching to a randomised ground disc is a separate regime (see note).
randomise_start: false
episode_seconds: 30.0
red_start_pos: [0.5, 0.0, 2.0]
red_start_yaw: 0.0
red_action_scale: 0.6
red_start_r_max: null
dense_scale_schedule: [[0, 1.0], [3_000_000, 0.0]]
red_action_scale_schedule: [[0, 0.6], [3_000_000, 1.0]]
red_start_r_max_schedule: null
```

> **Regime note (red_start_r_max):** the random-start disc (`randomise_start: true`) spawns Red on the **ground** (`z=0`), whereas the league trains from a **fixed airborne** start (`z=2`). Annealing start *spread* therefore means switching regimes, not just widening a disc — left off by default. The lever is plumbed and tested; enable it only with a deliberate randomised-start experiment.

- [ ] **Step 2: Write the experiment YAML**

Create `conf/experiment/rllib_league_step5.yaml`:

```yaml
# @package _global_
# RLlib migration Step 5: league + curriculum + eval battery.
#   - eval battery (5b) on: snapshots gate on clean eval_* metrics.
#   - asymmetric snapshot thresholds: Red's bar lowered (it rarely held >=0.7).
#   - dense->sparse reward anneal + Red action-authority ramp (league_anneal).
run_name: rllib_league_step5

defaults:
  - override /obs: duel_v1_body_n1
  - override /reward: team_selfplay_v1
  - override /curriculum: league_anneal
  - override /multiagent: red_blue_league
  - override /league: default

league:
  snapshot_threshold_red: 0.5    # Red scores less often than Blue prevents
  eval_enabled: true

algo:
  total_timesteps: 5_000_000
seed: 42
```

- [ ] **Step 3: Write a config-loading test**

Append to `tests/test_config_loading.py` (follow the file's existing `hydra_compose`/`compose` pattern):

```python
def test_rllib_league_step5_experiment_composes():
    from tests.conftest import hydra_compose
    with hydra_compose(overrides=["+experiment=rllib_league_step5"]) as cfg:
        assert cfg.curriculum.dense_scale_schedule == [[0, 1.0], [3_000_000, 0.0]]
        assert cfg.curriculum.red_action_scale == 0.6
        assert cfg.league.snapshot_threshold_red == 0.5
        assert cfg.league.eval_enabled is True
        assert cfg.reward._target_.endswith("RewardStack")
```

> If `tests/conftest.py` exposes the compose helper under a different name, use the same call the existing experiment-composition tests in `tests/test_config_loading.py` use.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m pytest tests/test_config_loading.py::test_rllib_league_step5_experiment_composes -v`
Expected: PASS. (If it fails on a struct-mode "key not in schema" error for a curriculum field, recheck Task 5 Step 3 added every field to `CurriculumConfig`.)

- [ ] **Step 5: Commit**

```bash
git add conf/curriculum/league_anneal.yaml conf/experiment/rllib_league_step5.yaml tests/test_config_loading.py
git commit -m "feat(conf): league_anneal curriculum + rllib_league_step5 experiment"
```

---

## Task 9: Slow integration — anneal reaches the env over a real run

A `@pytest.mark.slow` test proving `CurriculumCallback` actually mutates the live env's `dense_scale` and `red_action_scale` during a tiny league run. Mirrors the existing smoke-test shape.

**Files:**
- Modify: [tests/rllib/test_smoke_train.py](../../../tests/rllib/test_smoke_train.py)

- [ ] **Step 1: Write the slow integration test**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_curriculum_anneal_reaches_env(tmp_path):
    """CurriculumCallback pushes the scheduled dense_scale + red_action_scale
    onto the live env. With a schedule that hits 0 almost immediately, after one
    iteration the local env runner's reward stack reports the annealed value."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        # dense_scale drops to 0 by step 1; red_action_scale ramps 0.5 -> 1.0.
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0,
                       "red_action_scale": 0.5,
                       "dense_scale_schedule": [[0, 1.0], [1, 0.0]],
                       "red_action_scale_schedule": [[0, 0.5], [10_000, 1.0]]},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()  # one iteration -> CurriculumCallback.on_train_result fired
        inner = algo.env_runner.env._inner   # QuidditchTeamEnv on the local runner
        assert inner._reward_stack.dense_scale == 0.0     # annealed to zero
        assert inner.cfg.red_action_scale > 0.5           # ramped up off the floor
    finally:
        algo.stop()
```

> This run uses `team_selfplay_v1`'s reward stack via `build_ppo_config`'s default? No — the inline cfg passes `reward_stack: None`, so the env builds its own `default_team_stack()` (team_v2). `dense_scale` still applies (it's a `RewardStack` field). The assertion only needs the live `dense_scale` attribute, which every `RewardStack` has after Task 2.

- [ ] **Step 2: Run the slow test**

Run: `uv run python -m pytest tests/rllib/test_smoke_train.py::test_curriculum_anneal_reaches_env -v -m slow`
Expected: PASS. If `algo.env_runner.env._inner` raises `AttributeError`, the env-runner env attribute differs in the installed Ray version — adjust `_inner_team_env` (Task 6) to match, then re-run (this is the env-navigation seam called out in Task 6).

- [ ] **Step 3: Commit**

```bash
git add tests/rllib/test_smoke_train.py
git commit -m "test(rllib): slow integration — curriculum anneal reaches the env"
```

---

## Task 10: Full-suite verification

- [ ] **Step 1: Fast suite**

Run: `uv run python -m pytest -m "not slow"`
Expected: PASS — all prior fast tests plus the new `test_curriculum.py`, the league/team-env/config-builder/schema/config-loading additions.

- [ ] **Step 2: Slow suite**

Run: `uv run python -m pytest -m slow`
Expected: PASS — existing RLlib smoke tests plus `test_curriculum_anneal_reaches_env` (and 5b's battery smoke if landed). (5 macOS-render tests may fail in a headless shell — environmental.)

- [ ] **Step 3: Stop for review**

Ready to merge into develop (`--no-ff`) **after** a user-confirmed real run of `+experiment=rllib_league_step5` showing: Red takes snapshots (per-side threshold working), and W&B `curriculum/dense_scale` / `curriculum/red_action_scale` panels track the schedules. Do not merge before that confirmation.

---

## Self-review checklist (run before handing off)

1. **Spec coverage:**
   - §B2 reward anneal (dense→sparse on a step schedule, over the `RewardStack` magnitudes) → Tasks 2, 3, 6, 8 (`dense_scale` + `scheduled_value` + `CurriculumCallback` + `dense_scale_schedule`). ✅
   - §B5 difficulty levers `red_start_x_max` + `red_action_scale` as league levers → Tasks 4–8 (re-added as `red_start_r_max` (radius-honest name for the spec's `red_start_x_max` disc cap) + `red_action_scale`, both annealable; `red_action_scale` demonstrated in the experiment, `red_start_r_max` plumbed with the documented airborne/ground regime caveat). ✅
   - "Judge by `eval/success_rate`, never reward" / asymmetric thresholds watch-item → Task 1 (per-side thresholds, gating on 5b's clean metric). ✅
2. **Type consistency:** `scheduled_value(schedule, t) -> float | None`; `RewardStack.dense_scale` / `set_dense_scale`; `TeamConfig.red_action_scale` / `red_start_r_max`; `_curriculum_dict_from(cfg)` keys (`dense_scale_schedule`, `red_action_scale_schedule`, `red_start_r_max_schedule`) match `CurriculumConfig` fields and the callback's `cur.get(...)` reads. `_inner_team_env(runner) -> QuidditchTeamEnv`. ✅
3. **No placeholders:** every code step shows complete code; commands have expected output. The two env-runner navigation seams (`_inner_team_env`, the smoke test's `algo.env_runner.env._inner`) are flagged as the spots to adjust against the installed Ray version. ✅
4. **Canary safety:** `dense_scale=1.0`, `red_action_scale=1.0`, `red_start_r_max=None` are all no-ops; Task 2 Step 5 and Task 4 Step 7 explicitly re-run the canaries. ✅
```
