# RLlib League Step 5b — Eval Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the RLlib league a dedicated, deterministic eval battery that produces clean, length-unconfounded metrics (honest Blue prevention `eval_blue_prevention_rate`, `eval_red_score_rate`, `eval_takedown_rate`, terminal-cause histogram) and feeds them to the league's snapshot gating — replacing the NaN-prone windowed in-training metrics that currently drive snapshotting.

**Architecture:** A new `rllib/eval_battery.py` module splits into (a) pure aggregation functions, (b) a generic deterministic head-to-head rollout helper `rollout_battery(env, act_red, act_blue, …)`, and (c) an RLModule→action adapter. A thin `EvalBatteryCallback(RLlibCallback)` runs the battery on a cadence inside `on_train_result` (on the driver, in-process — the same way the slow smoke tests build & step an env), caches the result, and writes flat `eval_*` keys into the train result every iteration. `LeagueCallback.on_train_result` is taught to *prefer* the `eval_*` metric for each side, falling back to the existing windowed metric when the battery is disabled (fully backward-compatible with Step 4).

**Tech Stack:** Python 3.13, Ray RLlib new API stack (RLModule + Learner), PettingZoo/MuJoCo env, Hydra config, pytest (`@pytest.mark.slow` for Ray/MuJoCo integration). Run everything via `uv run` (sandbox disabled — `uv`/`make` need `~/.cache/uv`).

**Why this is Step 5b (sequencing):** The brain calls the dedicated eval battery "the Step-5 fix" for snapshot quality. Step 4 judges snapshots by RLlib's windowed `red_score_rate`/`blue_prevention_rate`, which emit NaN when a window drains and are confounded by the PFSP matchup mix. This plan provides the clean signal that **5a** (asymmetric per-side snapshot thresholds) and **5c** (promotion "dominates prior prod across the battery") both build on. The reusable `rollout_battery` defined here is consumed by **5c**'s `eval_team` RLlib port.

---

## Orientation for the implementer

You are extending an RLlib self-play league. Key existing files (read before starting):

- [rllib/metrics.py](../../../rllib/metrics.py) — `ScoreMetricsCallback`: the *pattern to follow*. Pure aggregation functions + a thin `RLlibCallback` that drives them from `on_episode_*` hooks and logs via `metrics_logger.log_value(...)`. The windowed `red_score_rate` / `blue_prevention_rate` it logs are exactly the confounded metrics we are replacing for snapshot gating.
- [rllib/league.py](../../../rllib/league.py) — `LeagueCallback`. The `_SIDES` tuple (lines 290–293) + `on_train_result` (lines 357–394) read the per-side metric via `read_metric(result, metric_name)` (lines 106–128, NaN-guarded). This is the single place we re-point at the eval metric.
- [rllib/config_builder.py](../../../rllib/config_builder.py) — `build_ppo_config`. Registers callbacks at lines 99/106 (`callbacks = [ScoreMetricsCallback, LeagueCallback]` when league on). Assembles `env_config` at lines 114–122 (`learner_id`, `obs_blocks`, `team_cfg`, `reward_stack`, `league`). The eval battery needs these same `env_config` values to rebuild an env on the driver.
- [envs/quidditch/rllib_env.py](../../../envs/quidditch/rllib_env.py) — `make_team_env(env_config)` builds a `QuidditchMultiAgentEnv` (RLlib `MultiAgentEnv`) wrapping `QuidditchTeamEnv`. `step(action_dict)` returns `(obs, rew, term, trunc, infos)` with `term["__all__"]`/`trunc["__all__"]` whole-episode flags. We rebuild the eval env with this exact creator so the obs/reward/team config matches training.
- [core/eval_core.py](../../../core/eval_core.py) — `TERMINAL_BUCKETS` (lines 32–39) and `_classify_terminal(red_info, blue_info)` (lines 249–261): a pure function mapping a terminal step's two info dicts to one of nine buckets. **Reuse both** — they are import-clean (the SB3 imports in `eval_core` are lazy, inside `run_scenario`). Note `take_down_fired` is read by `eval_core` but **never set by the env** (Task 1 fixes this).
- [envs/quidditch/team_env.py](../../../envs/quidditch/team_env.py) — `QuidditchTeamEnv.step` builds the per-agent info dicts at lines 490–501 (normal) and 564–569 (aftermath). `scored` is already present in both agents' info; `take_down_fired` is missing.
- [tests/rllib/test_smoke_train.py](../../../tests/rllib/test_smoke_train.py) — the `@pytest.mark.slow` pattern: build an inline `OmegaConf` cfg, `build_ppo_config(cfg).build_algo()`, `algo.train()`, inspect `algo.env_runner.module.keys()`. Our integration test follows this shape.

**Test/run commands** (from repo root, sandbox disabled):
- Single test: `uv run python -m pytest tests/rllib/test_eval_battery.py -v`
- Fast suite: `uv run python -m pytest -m "not slow"` (or `make test-fast`)
- Slow suite: `uv run python -m pytest -m slow`

**macOS/Ray note:** the eval battery builds and steps a MuJoCo env **on the driver process** inside `on_train_result`. The slow smoke tests already do exactly this in-process and pass, so `KMP_DUPLICATE_LIB_OK=TRUE` (set in `tests/conftest.py` and by `train_rllib`) covers it. Build the eval env with `render_mode=None` — never render in the eval path (the CGL-invalid-connection failure mode).

---

## File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `rllib/eval_battery.py` | Pure battery aggregation + `rollout_battery` + RLModule action adapter + `EvalBatteryCallback` | **Create** |
| `rllib/league.py` | Prefer `eval_<metric>` over windowed metric in the snapshot gate | **Modify** (`_SIDES`/`on_train_result`) |
| `rllib/config_builder.py` | Register `EvalBatteryCallback` when league + eval enabled | **Modify** (lines 99, 116–122) |
| `envs/quidditch/team_env.py` | Set `take_down_fired` in both info dicts (normal + aftermath) | **Modify** (lines 490–501, 564–569) |
| `conf/league/default.yaml` | Eval-battery knobs (`eval_enabled`, `eval_interval_iters`, `eval_episodes`, `eval_seed`) | **Modify** |
| `tests/rllib/test_eval_battery.py` | Unit tests for aggregation + rollout + callback wiring | **Create** |
| `tests/rllib/test_smoke_train.py` | Slow integration: battery runs, writes `eval_*`, snapshot keys off it | **Modify** (append one test) |
| `tests/envs/quidditch/test_team_env_features.py` | `take_down_fired` info-key contract | **Modify** (append one test) |

---

## Task 1: Set `take_down_fired` in the env info dicts

`core/eval_core._classify_terminal` and the takedown-rate metric both read `red_info["take_down_fired"]` / `blue_info["take_down_fired"]`, but `QuidditchTeamEnv.step` never sets that key — so takedown-rate is silently always 0. A take-down *is* a `drone_drone_crash` that crossed the velocity threshold (the same flag `TakeDown` fires on). Set `take_down_fired = drone_drone_crash` in both agents' info, on both the normal and aftermath paths.

**Files:**
- Modify: [envs/quidditch/team_env.py](../../../envs/quidditch/team_env.py) (normal-step info at 490–501; aftermath info at 564–569)
- Test: [tests/envs/quidditch/test_team_env_features.py](../../../tests/envs/quidditch/test_team_env_features.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/envs/quidditch/test_team_env_features.py`:

```python
def test_take_down_fired_set_on_drone_drone_crash():
    """take_down_fired mirrors drone_drone_crash in BOTH agents' info dicts.

    eval_core._classify_terminal and the eval battery's takedown-rate read this
    key; before this fix the env never set it, so takedown-rate was always 0.
    """
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False))
    env.reset(seed=0)
    # One ordinary step: no crash -> take_down_fired present and False.
    zero = np.zeros(4, dtype=np.float32)
    _, _, _, _, infos = env.step({"red_0": zero, "blue_0": zero})
    assert infos["red_0"]["take_down_fired"] is False
    assert infos["blue_0"]["take_down_fired"] is False
    # Force a drone-drone crash flag and confirm both infos mirror it.
    env._aftermath_steps_left = 0
    env.reset(seed=0)
    env.cfg.crash_aftermath_seconds = 0.0
    # Drive the detector path directly: monkeypatch events() to report a ram.
    import types
    fake = types.SimpleNamespace(
        solo_floor={"red_0": False, "blue_0": False},
        wall={"red_0": 0.0, "blue_0": 0.0},
        drone_drone=(0.0, 0.0, 99.0),  # rel speed >> crash_vel_thr
    )
    env._crash_detector.events = lambda f=fake: f  # type: ignore[assignment]
    _, _, _, _, infos = env.step({"red_0": zero, "blue_0": zero})
    assert infos["red_0"]["take_down_fired"] is True
    assert infos["blue_0"]["take_down_fired"] is True
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_features.py::test_take_down_fired_set_on_drone_drone_crash -v`
Expected: FAIL with `KeyError: 'take_down_fired'`.

- [ ] **Step 3: Add the key to the normal-step info dicts**

In `envs/quidditch/team_env.py`, the normal-step info update (currently lines 490–501) — add `take_down_fired` to both agents:

```python
        infos[self._red_id].update({
            "scored": scored, "drone_drone_crash": drone_drone_crash,
            "take_down_fired": drone_drone_crash,
            "red_floor": red_floor, "red_wall_crash": red_wall_crash,
            "red_oob": red_oob, "step": self._step_count,
            "dist_red_to_hoop": dist_red,
        })
        infos[self._blue_id].update({
            "scored": scored, "drone_drone_crash": drone_drone_crash,
            "take_down_fired": drone_drone_crash,
            "blue_floor": blue_floor, "blue_wall_crash": blue_wall_crash,
            "blue_oob": blue_oob, "step": self._step_count,
            "dist_b2r": dist_b2r,
        })
```

- [ ] **Step 4: Add the key to the aftermath info dicts**

In `_step_aftermath` (currently lines 564–569), add `take_down_fired: True` (aftermath only ever fires from a drone-drone ram — see `_enter_aftermath` caller):

```python
        infos: dict[str, dict[str, Any]] = {
            self._red_id:  {"aftermath": True, "drone_drone_crash": True,
                            "take_down_fired": True, "step": self._step_count},
            self._blue_id: {"aftermath": True, "drone_drone_crash": True,
                            "take_down_fired": True, "step": self._step_count},
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_features.py::test_take_down_fired_set_on_drone_drone_crash -v`
Expected: PASS.

- [ ] **Step 6: Confirm the team canary still holds (info-dict additions are additive)**

Run: `uv run python -m pytest tests/envs/quidditch/test_team_env_canary.py -v`
Expected: PASS (adding keys does not change rewards/dynamics).

- [ ] **Step 7: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_team_env_features.py
git commit -m "feat(team-env): set take_down_fired in info dicts (eval-battery prereq)"
```

---

## Task 2: Pure battery-metrics aggregation

Create `rllib/eval_battery.py` with pure functions mirroring `rllib/metrics.py`'s split: a fresh accumulator, a per-episode fold, and a finalizer that emits the flat `eval_*` metric keys the league gate will read. "Honest prevention" = fraction of episodes Red did **not** score — a binary per-episode question, independent of episode length (the load-bearing, length-unconfounded property).

**Files:**
- Create: [rllib/eval_battery.py](../../../rllib/eval_battery.py)
- Test: [tests/rllib/test_eval_battery.py](../../../tests/rllib/test_eval_battery.py)

- [ ] **Step 1: Write the failing test**

Create `tests/rllib/test_eval_battery.py`:

```python
"""Step-5b eval battery: pure aggregation + rollout + callback wiring."""
from __future__ import annotations

import rllib.eval_battery as EB


def test_battery_metrics_honest_prevention_is_length_independent():
    acc = EB.init_battery_acc()
    # 4 episodes: 1 score, 3 prevented. Episode lengths vary wildly; prevention
    # must depend ONLY on the binary scored flag, not on length.
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",    length=900)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout",  length=10)
    EB.fold_episode(acc, scored=False, take_down=True,  bucket="drone_drone_crash", length=50)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="red_oob",  length=5)
    m = EB.battery_metrics(acc)
    assert m["eval_red_score_rate"] == 0.25
    assert m["eval_blue_prevention_rate"] == 0.75
    assert m["eval_takedown_rate"] == 0.25
    assert abs(m["eval_mean_ep_len"] - (900 + 10 + 50 + 5) / 4) < 1e-9


def test_battery_metrics_terminal_histogram():
    acc = EB.init_battery_acc()
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",   length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    m = EB.battery_metrics(acc)
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 2
    # every bucket is represented (zeros included) so W&B panels are stable
    assert m["eval_terminal_red_oob"] == 0


def test_battery_metrics_empty_is_safe():
    m = EB.battery_metrics(EB.init_battery_acc())
    assert m["eval_red_score_rate"] == 0.0
    assert m["eval_blue_prevention_rate"] == 0.0
    assert m["eval_mean_ep_len"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rllib.eval_battery'`.

- [ ] **Step 3: Write the pure aggregation (minimal module)**

Create `rllib/eval_battery.py`:

```python
"""Dedicated, deterministic eval battery for the RLlib league (Step 5b).

Three layers, mirroring rllib/metrics.py:
  1. Pure aggregation (init/fold/finalize) — unit-tested, no RLlib deps.
  2. rollout_battery(env, act_red, act_blue, ...) — deterministic head-to-head
     rollouts on a QuidditchMultiAgentEnv; takes two obs->action callables.
  3. EvalBatteryCallback — runs the battery on a cadence inside on_train_result
     and writes flat eval_* keys into the train result.

Why a dedicated battery: Step 4 gates snapshots on RLlib's WINDOWED
red_score_rate / blue_prevention_rate, which emit NaN when a window drains and
are confounded by the PFSP matchup mix. The battery plays a fixed number of
deterministic main_red-vs-main_blue episodes and reports clean, length-
unconfounded metrics. "Honest prevention" = fraction of episodes Red did NOT
score — a binary per-episode question, independent of episode length.
"""
from __future__ import annotations

from core.eval_core import TERMINAL_BUCKETS, _classify_terminal


# ── Pure aggregation ────────────────────────────────────────────────────────
def init_battery_acc() -> dict:
    """Fresh battery accumulator."""
    return {
        "n": 0,
        "scored": 0,
        "take_down": 0,
        "len_sum": 0,
        "buckets": {b: 0 for b in TERMINAL_BUCKETS},
    }


def fold_episode(
    acc: dict, *, scored: bool, take_down: bool, bucket: str, length: int
) -> None:
    """Fold one finished episode's outcome into the accumulator (in place)."""
    acc["n"] += 1
    acc["scored"] += 1 if scored else 0
    acc["take_down"] += 1 if take_down else 0
    acc["len_sum"] += int(length)
    if bucket in acc["buckets"]:
        acc["buckets"][bucket] += 1


def battery_metrics(acc: dict) -> dict:
    """Flat eval_* metrics. Red score-rate, honest Blue prevention (1 - score-
    rate), takedown-rate, per-bucket terminal histogram, mean episode length."""
    n = acc["n"]
    score_rate = acc["scored"] / n if n else 0.0
    out: dict[str, float] = {
        "eval_red_score_rate": score_rate,
        "eval_blue_prevention_rate": (1.0 - score_rate) if n else 0.0,
        "eval_takedown_rate": (acc["take_down"] / n) if n else 0.0,
        "eval_mean_ep_len": (acc["len_sum"] / n) if n else 0.0,
        "eval_episodes": float(n),
    }
    for bucket, count in acc["buckets"].items():
        out[f"eval_terminal_{bucket}"] = count
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/eval_battery.py tests/rllib/test_eval_battery.py
git commit -m "feat(rllib): pure eval-battery aggregation (honest prevention)"
```

---

## Task 3: Deterministic head-to-head rollout helper

Add `rollout_battery(env, act_red, act_blue, *, n_episodes, seed)` to `rllib/eval_battery.py`. It plays `n_episodes` deterministic episodes on a `QuidditchMultiAgentEnv`, driving each agent from its `obs -> action` callable, classifies each terminal via `_classify_terminal`, and returns `battery_metrics(...)`. Tested fast with a stub env (no MuJoCo) — the MuJoCo path is covered by the slow integration test in Task 7.

**Files:**
- Modify: [rllib/eval_battery.py](../../../rllib/eval_battery.py)
- Test: [tests/rllib/test_eval_battery.py](../../../tests/rllib/test_eval_battery.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_eval_battery.py`:

```python
class _StubEnv:
    """Scripted multi-agent env: each episode runs `script` steps, then ends
    with the given terminal infos. Mimics QuidditchMultiAgentEnv's step shape
    (obs/rew/term/trunc/infos dicts with __all__) without MuJoCo."""

    def __init__(self, episodes):
        # episodes: list of (n_steps, scored, take_down, terminal_cause)
        self._episodes = episodes
        self._idx = -1

    def reset(self, *, seed=None, options=None):
        self._idx += 1
        self._step = 0
        return {"red_0": [0.0], "blue_0": [0.0]}, {}

    def step(self, action_dict):
        n_steps, scored, take_down, cause = self._episodes[self._idx]
        self._step += 1
        done = self._step >= n_steps
        obs = {"red_0": [0.0], "blue_0": [0.0]}
        rew = {"red_0": 0.0, "blue_0": 0.0}
        term = {"red_0": done, "blue_0": done, "__all__": done}
        trunc = {"red_0": False, "blue_0": False, "__all__": False}
        infos = {"red_0": {}, "blue_0": {}}
        if done:
            # Stamp the terminal info so _classify_terminal returns `cause` and
            # `scored`/`take_down_fired` read correctly.
            if cause == "score":
                infos["red_0"]["scored"] = True
                infos["blue_0"]["scored"] = True
            elif cause == "drone_drone_crash":
                infos["red_0"]["drone_drone_crash"] = True
                infos["blue_0"]["drone_drone_crash"] = True
            elif cause == "red_oob":
                infos["red_0"]["red_oob"] = True
            infos["red_0"]["take_down_fired"] = take_down
            infos["blue_0"]["take_down_fired"] = take_down
        return obs, rew, term, trunc, infos


def test_rollout_battery_aggregates_outcomes():
    env = _StubEnv([
        (3, True,  False, "score"),
        (5, False, False, "timeout"),
        (2, False, True,  "drone_drone_crash"),
    ])
    # Action callables are never inspected by the stub; identity is fine.
    m = EB.rollout_battery(env, lambda o: o, lambda o: o, n_episodes=3, seed=0)
    assert m["eval_red_score_rate"] == 1 / 3
    assert m["eval_blue_prevention_rate"] == 2 / 3
    assert m["eval_takedown_rate"] == 1 / 3
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 1
    assert m["eval_terminal_drone_drone_crash"] == 1
    assert abs(m["eval_mean_ep_len"] - (3 + 5 + 2) / 3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py::test_rollout_battery_aggregates_outcomes -v`
Expected: FAIL with `AttributeError: module 'rllib.eval_battery' has no attribute 'rollout_battery'`.

- [ ] **Step 3: Implement `rollout_battery`**

Append to `rllib/eval_battery.py`:

```python
import numpy as np


# ── Deterministic head-to-head rollouts ─────────────────────────────────────
def rollout_battery(env, act_red, act_blue, *, n_episodes: int, seed: int) -> dict:
    """Play `n_episodes` deterministic episodes; aggregate into eval_* metrics.

    `env` is a QuidditchMultiAgentEnv (RLlib MultiAgentEnv): step returns
    (obs, rew, term, trunc, infos) dicts with an "__all__" whole-episode flag.
    `act_red` / `act_blue` map an agent's obs array -> action array. A per-
    episode seed derived from `seed` keeps the battery reproducible run-to-run.
    """
    acc = init_battery_acc()
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        length = 0
        red_info: dict = {}
        blue_info: dict = {}
        while True:
            actions = {}
            if "red_0" in obs:
                actions["red_0"] = act_red(obs["red_0"])
            if "blue_0" in obs:
                actions["blue_0"] = act_blue(obs["blue_0"])
            obs, _, term, trunc, infos = env.step(actions)
            length += 1
            red_info = infos.get("red_0", red_info) or red_info
            blue_info = infos.get("blue_0", blue_info) or blue_info
            if term.get("__all__") or trunc.get("__all__"):
                break
        scored = bool(red_info.get("scored") or blue_info.get("scored"))
        take_down = bool(
            red_info.get("take_down_fired") or blue_info.get("take_down_fired")
        )
        bucket = _classify_terminal(red_info, blue_info)
        fold_episode(acc, scored=scored, take_down=take_down,
                     bucket=bucket, length=length)
    return battery_metrics(acc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py::test_rollout_battery_aggregates_outcomes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rllib/eval_battery.py tests/rllib/test_eval_battery.py
git commit -m "feat(rllib): deterministic head-to-head rollout_battery"
```

---

## Task 4: RLModule → deterministic-action adapter

Add `module_action_fn(module)` that wraps a live RLModule into the `obs -> action` callable `rollout_battery` expects, taking the **deterministic** action (the policy mean) from `forward_inference`. PPO's default Torch RLModule emits `action_dist_inputs` = `[mean (4), log_std (4)]` for the `Box(4)` action; the deterministic action is the mean (`[:4]`). The env clips/scales it (`ACTION_SCALE` + `np.clip`), so an unsquashed mean is fine.

**Files:**
- Modify: [rllib/eval_battery.py](../../../rllib/eval_battery.py)
- Test: [tests/rllib/test_eval_battery.py](../../../tests/rllib/test_eval_battery.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_eval_battery.py`:

```python
def test_module_action_fn_returns_deterministic_mean():
    import numpy as np
    import torch
    from ray.rllib.core.columns import Columns

    class _FakeModule:
        """Emits action_dist_inputs = [mean(4), log_std(4)] for a 1-row batch."""
        def forward_inference(self, batch):
            n = batch[Columns.OBS].shape[0]
            mean = torch.arange(4, dtype=torch.float32).repeat(n, 1)  # [0,1,2,3]
            log_std = torch.zeros(n, 4)
            return {Columns.ACTION_DIST_INPUTS: torch.cat([mean, log_std], dim=1)}

    fn = EB.module_action_fn(_FakeModule())
    a = fn(np.zeros(8, dtype=np.float32))
    assert isinstance(a, np.ndarray)
    assert a.shape == (4,)
    assert np.allclose(a, [0.0, 1.0, 2.0, 3.0])   # the mean, not a sample
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py::test_module_action_fn_returns_deterministic_mean -v`
Expected: FAIL with `AttributeError: ... no attribute 'module_action_fn'`.

- [ ] **Step 3: Implement the adapter**

Append to `rllib/eval_battery.py`:

```python
def module_action_fn(module):
    """Wrap an RLModule into a deterministic obs->action callable.

    Uses forward_inference and takes the action distribution's mean (the first
    half of action_dist_inputs for a DiagGaussian over the Box(4) action) — the
    greedy/deterministic action. torch is imported lazily so the pure-aggregation
    layer stays dependency-free.
    """
    import torch
    from ray.rllib.core.columns import Columns

    def _act(obs):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_t})
        dist_inputs = out[Columns.ACTION_DIST_INPUTS][0]
        action_dim = dist_inputs.shape[-1] // 2   # [mean, log_std]
        mean = dist_inputs[:action_dim]
        return mean.cpu().numpy().astype(np.float32)

    return _act
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py::test_module_action_fn_returns_deterministic_mean -v`
Expected: PASS.

> **Implementer note:** `action_dist_inputs` layout is RLlib-version-specific. If the slow integration test in Task 7 produces non-finite or constant metrics, verify the inputs shape (`print(out[Columns.ACTION_DIST_INPUTS].shape)`) — a squashed/uneven split would need adjusting here. This is the one spot most likely to need a tweak against the installed Ray version.

- [ ] **Step 5: Commit**

```bash
git add rllib/eval_battery.py tests/rllib/test_eval_battery.py
git commit -m "feat(rllib): RLModule deterministic-action adapter for the battery"
```

---

## Task 5: `EvalBatteryCallback` — run the battery on a cadence

Add `EvalBatteryCallback(RLlibCallback)` to `rllib/eval_battery.py`. On `on_train_result`, every `eval_interval_iters` (and on iteration 1), it rebuilds a local eval env from the algorithm's `env_config`, wraps the live `main_red`/`main_blue` modules with `module_action_fn`, runs `rollout_battery`, caches the result, and writes the cached `eval_*` keys into `result["eval"]` (flat keys, so `read_metric` finds them). It writes the **cached** metrics every iteration so the league gate always sees the most recent eval.

**Files:**
- Modify: [rllib/eval_battery.py](../../../rllib/eval_battery.py)
- Test: [tests/rllib/test_eval_battery.py](../../../tests/rllib/test_eval_battery.py)

- [ ] **Step 1: Write the failing test (cadence + caching, no MuJoCo)**

Append to `tests/rllib/test_eval_battery.py`:

```python
import types


def _eval_algo(league_cfg, modules=("main_red", "main_blue")):
    """Minimal Algorithm stand-in for EvalBatteryCallback: exposes env_config,
    iteration, and get_module."""
    return types.SimpleNamespace(
        iteration=1,
        config=types.SimpleNamespace(env_config={
            "learner_id": "red_0",
            "obs_blocks": ["ANG_VEL"],
            "team_cfg": {},
            "reward_stack": None,
            "league": league_cfg,
        }),
        get_module=lambda mid=None: object(),
    )


def test_eval_callback_runs_on_cadence_and_caches(monkeypatch):
    calls = {"n": 0}

    def fake_battery(env, act_red, act_blue, *, n_episodes, seed):
        calls["n"] += 1
        return {"eval_red_score_rate": 0.3, "eval_blue_prevention_rate": 0.7}

    # Stub the env build + rollout so the test needs no MuJoCo.
    monkeypatch.setattr(EB, "_build_eval_env", lambda env_config: object())
    monkeypatch.setattr(EB, "rollout_battery", fake_battery)
    monkeypatch.setattr(EB, "module_action_fn", lambda m: (lambda o: o))

    cb = EB.EvalBatteryCallback()
    algo = _eval_algo({"eval_enabled": True, "eval_interval_iters": 5,
                       "eval_episodes": 4, "eval_seed": 0})

    # iter 1: always evaluates.
    r1 = {}
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result=r1)
    assert calls["n"] == 1
    assert EB.read_eval(r1, "eval_red_score_rate") == 0.3

    # iter 3: not on cadence -> no new battery, but cached metrics still written.
    r3 = {}
    algo.iteration = 3
    cb.on_train_result(algorithm=algo, result=r3)
    assert calls["n"] == 1                                   # not re-run
    assert EB.read_eval(r3, "eval_blue_prevention_rate") == 0.7  # cached

    # iter 6: 6 % 5 hits a cadence boundary -> re-run.
    r6 = {}
    algo.iteration = 6
    cb.on_train_result(algorithm=algo, result=r6)
    assert calls["n"] == 2


def test_eval_callback_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(EB, "_build_eval_env",
                        lambda env_config: (_ for _ in ()).throw(AssertionError))
    cb = EB.EvalBatteryCallback()
    algo = _eval_algo({"eval_enabled": False})
    result = {}
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result=result)   # must not build an env
    assert "eval" not in result
```

`read_eval` is a tiny helper (added below) that mirrors `league.read_metric`'s recursive search, used here to assert against the nested `result["eval"]` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py::test_eval_callback_runs_on_cadence_and_caches -v`
Expected: FAIL with `AttributeError: ... no attribute 'EvalBatteryCallback'`.

- [ ] **Step 3: Implement the callback + env builder + read helper**

Append to `rllib/eval_battery.py`:

```python
from ray.rllib.callbacks.callbacks import RLlibCallback

_MAIN_RED = "main_red"
_MAIN_BLUE = "main_blue"


def read_eval(result: dict, name: str):
    """Recursively find flat eval key `name` in the (nested) result dict.

    The league gate reads eval_* metrics this way (same recursive shape as
    league.read_metric), tolerating wherever the eval block is nested.
    """
    if name in result and isinstance(result[name], (int, float)):
        return result[name]
    for v in result.values():
        if isinstance(v, dict):
            found = read_eval(v, name)
            if found is not None:
                return found
    return None


def _build_eval_env(env_config: dict):
    """Build a no-render eval env from the live env_config. Imported lazily so
    the pure layers don't pull MuJoCo. Mirrors training's env exactly (same obs
    blocks, team_cfg, reward stack) so eval is on-distribution."""
    from envs.quidditch.rllib_env import make_team_env
    return make_team_env(dict(env_config))


class EvalBatteryCallback(RLlibCallback):
    """Runs the dedicated eval battery on a cadence and writes clean eval_*
    metrics into the train result.

    Built and stepped on the driver (same in-process pattern as the slow smoke
    tests). The battery plays deterministic main_red-vs-main_blue episodes — a
    controlled head-to-head, NOT the PFSP matchup mix the windowed metrics see —
    so the snapshot/promotion gate keys off a stable, length-unconfounded signal.
    Cached metrics are re-written every iteration so the gate always sees the
    latest eval even between cadence boundaries.
    """

    def __init__(self):
        super().__init__()
        self._last_eval: dict = {}

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        cfg = dict(algorithm.config.env_config.get("league", {}))
        if not cfg.get("eval_enabled", False):
            return
        it = int(algorithm.iteration)
        interval = int(cfg.get("eval_interval_iters", 20))
        if it <= 1 or (interval > 0 and it % interval == 0) or not self._last_eval:
            self._last_eval = self._run_battery(algorithm, cfg)
        result.setdefault("eval", {}).update(self._last_eval)

    def _run_battery(self, algorithm, cfg: dict) -> dict:
        env = _build_eval_env(algorithm.config.env_config)
        try:
            act_red = module_action_fn(algorithm.get_module(_MAIN_RED))
            act_blue = module_action_fn(algorithm.get_module(_MAIN_BLUE))
            return rollout_battery(
                env, act_red, act_blue,
                n_episodes=int(cfg.get("eval_episodes", 20)),
                seed=int(cfg.get("eval_seed", 12345)),
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_eval_battery.py -v`
Expected: PASS (all eval-battery unit tests, including the two callback tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/eval_battery.py tests/rllib/test_eval_battery.py
git commit -m "feat(rllib): EvalBatteryCallback runs the battery on a cadence"
```

---

## Task 6: Gate snapshots on the eval metric (prefer `eval_*`, fall back to windowed)

Teach `LeagueCallback.on_train_result` to read the **clean** eval metric for each side when present, falling back to the existing windowed metric (Step-4 behavior) when the eval battery is disabled. This is the single line that re-points snapshot gating at the battery. Fully backward-compatible: with `eval_enabled: false`, nothing changes.

**Files:**
- Modify: [rllib/league.py](../../../rllib/league.py) (`on_train_result`, the `_SIDES` loop ~line 364)
- Test: [tests/rllib/test_league.py](../../../tests/rllib/test_league.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_league.py`:

```python
def test_snapshot_prefers_eval_metric_over_windowed(monkeypatch):
    """When the eval battery has written eval_red_score_rate, the gate uses it
    (not the confounded windowed red_score_rate)."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    # Windowed red_score_rate is high (0.9) but the clean eval says 0.2 -> below
    # the 0.7 threshold -> NO red snapshot. The eval metric wins.
    cb.on_train_result(algorithm=algo, result={
        "env_runners": {"red_score_rate": 0.9, "blue_prevention_rate": 0.1},
        "eval": {"eval_red_score_rate": 0.2, "eval_blue_prevention_rate": 0.1},
    })
    assert "red_pop_v1" not in algo.added


def test_snapshot_falls_back_to_windowed_without_eval(monkeypatch):
    """With no eval block (battery disabled), Step-4 windowed gating is intact."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={
        "env_runners": {"red_score_rate": 0.8, "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_league.py::test_snapshot_prefers_eval_metric_over_windowed -v`
Expected: FAIL — currently the windowed 0.9 clears the 0.7 threshold and `red_pop_v1` IS added.

- [ ] **Step 3: Prefer the eval metric in the gate**

In `rllib/league.py`, inside `on_train_result`, change the per-side metric read (currently `metric = read_metric(result, metric_name)` at line 365) to prefer the `eval_`-prefixed key:

```python
        for side, metric_name, main_id, regex in _SIDES:
            # Prefer the dedicated eval-battery metric (clean, length-unconfounded,
            # never NaN); fall back to the windowed in-training metric when the
            # battery is disabled (Step-4 behavior, fully backward-compatible).
            metric = read_metric(result, f"eval_{metric_name}")
            if metric is None:
                metric = read_metric(result, metric_name)
            if metric is None:
                continue
```

No change to `_SIDES` itself — `metric_name` stays `red_score_rate` / `blue_prevention_rate`, and the eval battery writes `eval_red_score_rate` / `eval_blue_prevention_rate` (matching `eval_` + `metric_name`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_league.py -v`
Expected: PASS (the two new tests plus all existing league tests — the fallback path keeps Step-4 tests green).

- [ ] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): gate snapshots on the eval metric, fall back to windowed"
```

---

## Task 7: Wire `EvalBatteryCallback` into the config builder + league config

Register `EvalBatteryCallback` in the callback list (before `LeagueCallback`, so the `eval` block is written into `result` before the gate reads it) when the league is on, and add the eval-battery knobs to `conf/league/default.yaml`. Verify the keys thread through to `env_config["league"]`.

**Files:**
- Modify: [rllib/config_builder.py](../../../rllib/config_builder.py) (callback list, line 99)
- Modify: [conf/league/default.yaml](../../../conf/league/default.yaml)
- Test: [tests/rllib/test_config_builder.py](../../../tests/rllib/test_config_builder.py)

- [ ] **Step 1: Add the eval knobs to the league config**

Append to `conf/league/default.yaml`:

```yaml
# Step-5b dedicated eval battery. When eval_enabled, EvalBatteryCallback runs a
# deterministic main_red-vs-main_blue battery every eval_interval_iters and
# writes clean eval_* metrics; the snapshot gate prefers them over the windowed
# (NaN-prone, PFSP-confounded) in-training metrics.
#   eval_interval_iters: cadence; keep <= min_iters_between_snapshots so a fresh
#                        eval exists each time a snapshot could fire.
eval_enabled: true
eval_interval_iters: 10
eval_episodes: 20
eval_seed: 12345
```

- [ ] **Step 2: Write the failing test (callback registration order + config threading)**

Append to `tests/rllib/test_config_builder.py` (follow the existing `_league_cfg()` inline-config pattern in that file):

```python
def test_eval_battery_callback_registered_before_league_when_enabled():
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.metrics import ScoreMetricsCallback
    from rllib.eval_battery import EvalBatteryCallback
    from rllib.league import LeagueCallback

    cfg = _league_cfg()  # the file's existing league-enabled inline cfg factory
    cfg = OmegaConf.merge(cfg, OmegaConf.create(
        {"league": {"eval_enabled": True, "eval_interval_iters": 10,
                    "eval_episodes": 4, "eval_seed": 0}}))
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    assert isinstance(cbs, (list, tuple))
    names = [c.__name__ for c in cbs]
    # Order matters: EvalBatteryCallback writes result["eval"] before the league
    # gate reads it in the same on_train_result sweep.
    assert names == ["ScoreMetricsCallback", "EvalBatteryCallback", "LeagueCallback"]
    assert EvalBatteryCallback in cbs and LeagueCallback in cbs
    assert ScoreMetricsCallback in cbs


def test_eval_battery_callback_absent_when_disabled():
    from rllib.config_builder import build_ppo_config
    from rllib.eval_battery import EvalBatteryCallback

    cfg = _league_cfg()
    cfg.league.eval_enabled = False
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    cbs = list(cbs) if isinstance(cbs, (list, tuple)) else [cbs]
    assert EvalBatteryCallback not in cbs
```

> If the file has no `_league_cfg()` helper, copy the inline league cfg dict from `tests/rllib/test_config_builder.py`'s existing league test (it builds an `OmegaConf` with `obs`/`algo`/`multiagent`/`league` keys). `config.callbacks_class` is RLlib's accessor for the configured callbacks; if the installed Ray version exposes them differently, read them off the built config object the same way the existing config-builder tests do.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py::test_eval_battery_callback_registered_before_league_when_enabled -v`
Expected: FAIL — current code registers `[ScoreMetricsCallback, LeagueCallback]`, missing `EvalBatteryCallback`.

- [ ] **Step 4: Register the callback conditionally**

In `rllib/config_builder.py`, add the import near the existing league import (line 20):

```python
from rllib.league import LeagueCallback, make_league_mapping_fn
from rllib.eval_battery import EvalBatteryCallback
```

Then in the `league_on` branch (currently line 99, `callbacks = [ScoreMetricsCallback, LeagueCallback]`), insert the eval callback between them when eval is enabled:

```python
    if league_on:
        league_dict = OmegaConf.to_container(league_cfg, resolve=True)
        # Populations start empty: only the two mains exist at build time.
        policy_mapping_fn = make_league_mapping_fn(set(ma.modules.keys()), league_dict)
        if league_dict.get("eval_enabled"):
            # Order: EvalBatteryCallback writes result["eval"] BEFORE LeagueCallback
            # reads it in the same on_train_result sweep.
            callbacks = [ScoreMetricsCallback, EvalBatteryCallback, LeagueCallback]
        else:
            callbacks = [ScoreMetricsCallback, LeagueCallback]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py -v`
Expected: PASS (both new tests + existing config-builder tests).

- [ ] **Step 6: Commit**

```bash
git add rllib/config_builder.py conf/league/default.yaml tests/rllib/test_config_builder.py
git commit -m "feat(rllib): wire EvalBatteryCallback + eval knobs into the league config"
```

---

## Task 8: Slow integration test — battery runs end-to-end and drives a snapshot

A `@pytest.mark.slow` test proving the whole path on a real (tiny) league run: the battery builds a MuJoCo env on the driver, writes `eval_*` into the train result, and a snapshot keys off it. Mirrors the existing `test_league_snapshots_freezes_and_restores` shape.

**Files:**
- Modify: [tests/rllib/test_smoke_train.py](../../../tests/rllib/test_smoke_train.py)

- [ ] **Step 1: Write the slow integration test**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_eval_battery_runs_and_writes_metrics(tmp_path):
    """A league run with the eval battery enabled writes clean eval_* metrics
    into the train result, and the snapshot gate consumes them (threshold 0)."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.eval_battery import read_eval
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
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5,
                   "eval_enabled": True, "eval_interval_iters": 1,
                   "eval_episodes": 3, "eval_seed": 0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        result = algo.train()   # iter 1: battery runs, writes eval_* metrics
        srate = read_eval(result, "eval_red_score_rate")
        prev = read_eval(result, "eval_blue_prevention_rate")
        assert srate is not None and 0.0 <= float(srate) <= 1.0
        assert prev is not None and abs((srate + prev) - 1.0) < 1e-6
        assert read_eval(result, "eval_episodes") == 3.0
    finally:
        algo.stop()
```

- [ ] **Step 2: Run the slow test**

Run: `uv run python -m pytest tests/rllib/test_smoke_train.py::test_eval_battery_runs_and_writes_metrics -v -m slow`
Expected: PASS. If the metrics are non-finite or `srate + prev != 1`, revisit the `module_action_fn` action extraction (Task 4 implementer note) before anything else.

- [ ] **Step 3: Commit**

```bash
git add tests/rllib/test_smoke_train.py
git commit -m "test(rllib): slow integration — eval battery writes clean metrics"
```

---

## Task 9: Full-suite verification

- [ ] **Step 1: Run the fast suite**

Run: `uv run python -m pytest -m "not slow"`
Expected: PASS — all previously-green fast tests plus the new `test_eval_battery.py`, `test_team_env_features.py`, `test_league.py`, and `test_config_builder.py` additions. (Step-4 baseline was 441 fast green; expect ~+8.)

- [ ] **Step 2: Run the slow suite**

Run: `uv run python -m pytest -m slow`
Expected: PASS — the four existing RLlib smoke tests plus `test_eval_battery_runs_and_writes_metrics`. (5 macOS-render tests may fail in a non-GUI shell — environmental, not a regression; run from a real Terminal window if needed.)

- [ ] **Step 3: Commit any final fixups, then stop for review**

The branch is now ready to merge into develop (`--no-ff`) after user confirmation of a real league run showing `eval_*` panels in W&B. **Do not merge without that confirmation** (verify behavior before committing the merge — GUI/training-run verification is the user's call).

---

## Self-review checklist (run before handing off)

1. **Spec coverage (component #6 / B4 eval battery):**
   - Honest `eval/success_rate` (Blue prevention) → `eval_blue_prevention_rate` (Task 2/3). ✅
   - Red score-rate → `eval_red_score_rate`. ✅
   - Takedown-rate → `eval_takedown_rate` (Task 1 unblocks the data; Task 2/3 aggregate). ✅
   - Terminal-cause histogram → `eval_terminal_<bucket>` (Task 2/3, reusing `TERMINAL_BUCKETS`). ✅
   - "Dedicated eval battery is the Step-5 fix" for snapshot quality → Task 6 re-points the gate. ✅
   - RLlib-native (not SB3) → `rollout_battery` + `module_action_fn` on RLModules; no `PPO.load`. ✅
   - **Deferred to 5c:** the standalone `scripts/eval_team.py` / `eval_battery.py` RLlib port (needs the checkpoint-dir RLModule loader). It consumes `rollout_battery` from this plan. Noted, not a gap in 5b.
2. **Type consistency:** `rollout_battery(env, act_red, act_blue, *, n_episodes, seed)` defined in Task 3 and consumed unchanged by the callback (Task 5) and by 5c. `module_action_fn(module) -> (obs->action)`. `battery_metrics` keys (`eval_red_score_rate`, `eval_blue_prevention_rate`, `eval_takedown_rate`, `eval_mean_ep_len`, `eval_episodes`, `eval_terminal_<bucket>`) match the league gate's `eval_` + `metric_name` lookup. ✅
3. **No placeholders:** every code step shows complete code; commands have expected output. ✅
```
