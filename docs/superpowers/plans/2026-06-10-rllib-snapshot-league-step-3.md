# Defender-Aware Reward + Step-3 Snapshot-Population League — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable `InterceptShaping` for `blue_0` under two-policy self-play (defender-aware env refactor), then add a frozen snapshot-population league (threshold-triggered snapshots, uniform opponent sampling, checkpoint-faithful restore).

**Architecture:** Part A de-gates the intercept-shaping `StepState` inputs in `team_env.py` so the defender→future-Red distance is always computed from `blue_0` (independent of `learner_id`). Part B adds `rllib/league.py` (a `LeagueCallback` + a league `policy_mapping_fn` factory + pure helpers); population membership is derived from module-id naming (`red_pop_v{N}`/`blue_pop_v{N}`), so RLlib's own module checkpointing makes restore faithful with no separate state file.

**Tech Stack:** Python 3.13 (uv), Ray/RLlib 2.55.1 new API stack (RLModule + Learner), Hydra config, PettingZoo `ParallelEnv` (`QuidditchTeamEnv`), pytest.

**Conventions for every task:**
- Run all commands from the worktree root: `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/rllib-league-step-3`.
- `uv` commands need the command sandbox disabled (cache write to `~/.cache/uv`).
- Tests run via `uv run pytest`. `WANDB_MODE=disabled` is set in `tests/conftest.py`. Slow/integration tests are gated `@pytest.mark.slow`; the default suite skips them, so run a slow test explicitly with `-m slow`.
- Commits are GPG-signed (`git commit -S`) and end with the `Co-Authored-By:` trailer.

---

## Part A — Defender-aware env refactor

Today `team_env.py` computes the intercept-shaping inputs (`dist_def_to_future_red*`) only inside `if self._learner_id is not None`, selecting the defender from the learner (`defender = blue if learner==blue else red`). The live Step-2 config sets `learner_id: red_0`, so the defender resolves to **red** — `dist_def_to_future_red` becomes ‖red − future_red‖ (meaningless for Blue), and `team_selfplay_v1.yaml` omits `InterceptShaping` entirely. Part A makes the defender always `blue_0` and computes the distance unconditionally.

### Task A1: De-gate intercept-shaping inputs (defender = blue_0, learner_id-independent)

**Files:**
- Modify: `envs/quidditch/team_env.py` (add helper near `_red_pos`/`_blue_pos` ~line 591; replace the reset block at lines 305-326; replace the step block at lines 442-453)
- Test: `tests/envs/quidditch/test_team_env_intercept.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/envs/quidditch/test_team_env_intercept.py`:

```python
"""Defender-aware intercept-shaping inputs: dist_def_to_future_red is computed
from blue_0 (the defender) regardless of learner_id, so InterceptShaping works
in two-policy self-play (RLlib migration Step 3 prerequisite)."""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.constants import REWARD_LOOKAHEAD_S
from envs.quidditch.rewards.stack import RewardStack
from envs.quidditch.team_env import QuidditchTeamEnv

_ZERO_ACTION = {"red_0": np.zeros(4, np.float32), "blue_0": np.zeros(4, np.float32)}


def _blue_to_future_red(env: QuidditchTeamEnv) -> float:
    """Recompute the defender(blue)->future-Red distance straight from env state."""
    red_vel = env._world.data.qvel[env._red_dofadr : env._red_dofadr + 3].copy()
    future_red = env._red_pos() + REWARD_LOOKAHEAD_S * red_vel
    return float(np.linalg.norm(env._blue_pos() - future_red))


@pytest.mark.parametrize("learner_id", [None, "red_0", "blue_0"])
def test_intercept_inputs_are_blue_based_regardless_of_learner_id(learner_id):
    env = QuidditchTeamEnv(reward_stack=RewardStack(terms=[]), learner_id=learner_id)
    env.reset(seed=0)

    # Reset seeds both caches to the blue->future-Red distance (a real, nonzero
    # distance because blue and red start apart).
    expected = _blue_to_future_red(env)
    assert expected > 0.0
    assert env._dist_def_to_future_red == pytest.approx(expected)
    assert env._dist_def_to_future_red_prev == pytest.approx(expected)

    prev = env._dist_def_to_future_red
    env.step(_ZERO_ACTION)

    # After a step: prev rolls to the old value, curr is recomputed blue-based.
    assert env._dist_def_to_future_red == pytest.approx(_blue_to_future_red(env))
    assert env._dist_def_to_future_red_prev == pytest.approx(prev)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_team_env_intercept.py -v`
Expected: FAIL. `learner_id=None` → `_dist_def_to_future_red == 0.0` (the else branch) `!= expected`. `learner_id="red_0"` → defender resolves to red, giving ‖red−future_red‖ `!= expected`.

- [ ] **Step 3: Add the `_dist_blue_to_future_red` helper**

In `envs/quidditch/team_env.py`, immediately after the `_blue_pos` method (currently ending ~line 599), add:

```python
    def _dist_blue_to_future_red(self) -> float:
        """Distance from the defender (blue_0) to Red's short-horizon predicted
        position: future_red = red_pos + REWARD_LOOKAHEAD_S * red_vel_world.

        Defender-aware: always blue-based and independent of learner_id, so
        InterceptShaping returns a real signal in two-policy self-play.
        """
        red_vel_world = self._world.data.qvel[
            self._red_dofadr : self._red_dofadr + 3
        ].copy()
        future_red = self._red_pos() + REWARD_LOOKAHEAD_S * red_vel_world
        return float(np.linalg.norm(self._blue_pos() - future_red))
```

- [ ] **Step 4: De-gate the reset cache init**

In `envs/quidditch/team_env.py`, replace the reset block (lines 305-326, from the `# Initialise closing-rate cache (formerly OCE side).` comment through the `else:` branch that zeros the three caches) with:

```python
        # Closing-rate OBS cache stays learner-gated (obs path unchanged).
        if self._learner_id is not None:
            learner_pos = (self._blue_pos() if self._learner_id == self._blue_id
                            else self._red_pos())
            opp_pos = (self._red_pos() if self._learner_id == self._blue_id
                        else self._blue_pos())
            self._prev_dist_to_opp = float(np.linalg.norm(opp_pos - learner_pos))
        else:
            self._prev_dist_to_opp = 0.0

        # Intercept-shaping REWARD cache: defender is always blue_0, populated
        # regardless of learner_id (Step-3 defender-aware refactor).
        self._dist_def_to_future_red_prev = self._dist_blue_to_future_red()
        self._dist_def_to_future_red = self._dist_def_to_future_red_prev
```

- [ ] **Step 5: De-gate the step cache update**

In `envs/quidditch/team_env.py`, replace the step block (lines 442-453, the `# ── InterceptShaping inputs (when a learner is configured) ──` comment and its `if self._learner_id is not None:` body) with:

```python
        # ── InterceptShaping inputs (defender = blue_0, always populated) ────
        self._dist_def_to_future_red_prev = self._dist_def_to_future_red
        self._dist_def_to_future_red = self._dist_blue_to_future_red()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_team_env_intercept.py -v`
Expected: PASS (all three `learner_id` params).

- [ ] **Step 7: Run the existing team-env + reward suites to confirm no regression**

Run: `uv run pytest tests/envs/quidditch/ -v`
Expected: PASS. (The `learner_id="blue_0"` path is byte-identical to before — the helper computes the same quantity the old `defender = blue` branch did. The single-agent canary reward stacks don't include `InterceptShaping`, so the now-populated field is simply unused there.)

- [ ] **Step 8: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_team_env_intercept.py
git commit -S -m "$(cat <<'EOF'
refactor(env): defender-aware intercept-shaping inputs (blue_0, learner_id-independent)

Compute dist_def_to_future_red from blue_0 unconditionally so InterceptShaping
returns a real signal in two-policy self-play (learner_id no longer gates it).
The closing-rate obs cache stays learner-gated; obs path unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task A2: Re-enable InterceptShaping in `team_selfplay_v1`

**Files:**
- Modify: `conf/reward/team_selfplay_v1.yaml`
- Modify: `tests/envs/quidditch/rewards/test_reward_stack.py:543-558` (flip the composition assertion)

- [ ] **Step 1: Update the failing composition test**

In `tests/envs/quidditch/rewards/test_reward_stack.py`, replace `test_team_selfplay_v1_stack_composition` (lines 543-558) with:

```python
def test_team_selfplay_v1_stack_composition():
    """conf/reward/team_selfplay_v1.yaml: Red dense-approach + zero-sum score;
    Blue anchor + INTERCEPT shaping + takedown; both crash-penalised.
    InterceptShaping is now present (Step-3 defender-aware env refactor)."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_selfplay_v1")
    assert [type(t).__name__ for t in stack.terms] == [
        "HoopApproachShaping", "ScoreEvent", "HoopAnchor",
        "InterceptShaping", "TakeDown", "CrashEvent",
    ]
    shaping = stack.terms[0]
    assert shaping.scale == 2.0 and shaping.agent == "red_0"
    score = stack.terms[1]
    assert score.magnitude == 10.0
    assert score.scorer == "red_0"
    assert score.zero_sum_opponent == "blue_0"   # Blue loses 10 when Red scores
    intercept = stack.terms[3]
    assert intercept.defender == "blue_0"
    assert intercept.scale == 0.05
    assert intercept.activation_dist == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/quidditch/rewards/test_reward_stack.py::test_team_selfplay_v1_stack_composition -v`
Expected: FAIL — current term list has no `InterceptShaping`; `stack.terms[3]` is `TakeDown`, so `.defender` raises `AttributeError` / the list-equality assert fails.

- [ ] **Step 3: Add the InterceptShaping term to the YAML**

In `conf/reward/team_selfplay_v1.yaml`, replace the comment block on lines 6-9 (the parenthetical explaining InterceptShaping is omitted) with a one-line note, and insert the term **between** the `HoopAnchor` term (ends line 24) and the `TakeDown` term (begins line 26). The Blue section becomes:

```yaml
# Blue (defender): HoopAnchor keeps it near the hoop; InterceptShaping rewards
# closing on Red's predicted position near the hoop; TakeDown rewards ramming
# Red out of the sky.
# Both: CrashEvent for floor/wall/OOB.
```

and the inserted term (after the `HoopAnchor` block, before `TakeDown`):

```yaml
  - _target_: envs.quidditch.rewards.terms.InterceptShaping
    scale: 0.05
    lookahead_s: 0.5
    activation_dist: 1.5
    defender: blue_0
```

(Values match the prior `conf/reward/team_v3_intercept.yaml` tuning.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/envs/quidditch/rewards/test_reward_stack.py::test_team_selfplay_v1_stack_composition -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/reward/team_selfplay_v1.yaml tests/envs/quidditch/rewards/test_reward_stack.py
git commit -S -m "$(cat <<'EOF'
feat(reward): re-enable InterceptShaping for blue_0 in team_selfplay_v1

The defender-aware env refactor (prior commit) populates the intercept-shaping
StepState inputs in two-policy self-play, so the term the YAML previously
deferred to Step 3 is now active for the Blue defender.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Part B — Step-3 snapshot-population league

### Task B1: League pure helpers (`rllib/league.py` — membership, metric lookup, snapshot predicate)

**Files:**
- Create: `rllib/league.py`
- Test: `tests/rllib/test_league.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/rllib/test_league.py`:

```python
"""Step-3 snapshot-population league: pure helpers + mapping fn + callback."""
from __future__ import annotations

import rllib.league as L


def test_population_members_sorted_by_version():
    ids = {"main_red", "main_blue", "red_pop_v2", "red_pop_v1", "blue_pop_v1"}
    assert L.population_members(ids, L.RED_POP_RE) == ["red_pop_v1", "red_pop_v2"]
    assert L.population_members(ids, L.BLUE_POP_RE) == ["blue_pop_v1"]


def test_next_version():
    assert L.next_version({"main_red", "main_blue"}, L.RED_POP_RE) == 1
    assert L.next_version({"red_pop_v1", "red_pop_v3"}, L.RED_POP_RE) == 4
    assert L.next_version({"red_pop_v1"}, L.BLUE_POP_RE) == 1


def test_read_metric_finds_nested_key():
    result = {"env_runners": {"red_score_rate": 0.8, "other": 1}, "x": 2}
    assert L.read_metric(result, "red_score_rate") == 0.8
    assert L.read_metric(result, "blue_prevention_rate") is None


def test_should_snapshot_threshold_cooldown_and_cap():
    base = dict(threshold=0.7, cooldown=20, pop_size=0, cap=5)
    # at/above threshold + cooldown elapsed + room -> True
    assert L.should_snapshot(metric=0.7, iters_since_last=20, **base)
    assert L.should_snapshot(metric=0.9, iters_since_last=999, **base)
    # below threshold -> False
    assert not L.should_snapshot(metric=0.69, iters_since_last=20, **base)
    # within cooldown -> False
    assert not L.should_snapshot(metric=0.9, iters_since_last=19, **base)
    # population full -> False
    assert not L.should_snapshot(
        metric=0.9, iters_since_last=20, threshold=0.7, cooldown=20, pop_size=5, cap=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rllib.league'`.

- [ ] **Step 3: Create `rllib/league.py` with the pure helpers**

```python
"""Step-3 snapshot-population self-play league.

Membership is derived from module-id naming (red_pop_v{N} / blue_pop_v{N}); the
set of modules in the MultiRLModule IS the population, so RLlib's module
checkpointing makes restore faithful with no separate state file.

This module is split into pure helpers (unit-tested) + a league policy_mapping_fn
factory + the LeagueCallback that drives snapshotting on on_train_result.
"""
from __future__ import annotations

import re
import zlib
from typing import Iterable, Optional

RED_POP_RE = re.compile(r"^red_pop_v(\d+)$")
BLUE_POP_RE = re.compile(r"^blue_pop_v(\d+)$")


def population_members(module_ids: Iterable[str], regex: re.Pattern) -> list[str]:
    """Population module ids matching `regex`, sorted ascending by version."""
    matched = [(int(regex.match(m).group(1)), m) for m in module_ids if regex.match(m)]
    return [m for _, m in sorted(matched)]


def next_version(module_ids: Iterable[str], regex: re.Pattern) -> int:
    """Next snapshot version for a side: max existing version + 1, else 1."""
    versions = [int(regex.match(m).group(1)) for m in module_ids if regex.match(m)]
    return (max(versions) + 1) if versions else 1


def read_metric(result: dict, name: str) -> Optional[float]:
    """Recursively find `name` anywhere in the (nested) train result dict.

    ScoreMetricsCallback logs red_score_rate / blue_prevention_rate via the
    MetricsLogger, which nests them under the env-runner results subtree; a
    recursive search avoids hard-coding a version-specific key path.
    """
    if name in result and isinstance(result[name], (int, float)):
        return float(result[name])
    for v in result.values():
        if isinstance(v, dict):
            found = read_metric(v, name)
            if found is not None:
                return found
    return None


def should_snapshot(
    *, metric: float, threshold: float, iters_since_last: int,
    cooldown: int, pop_size: int, cap: int,
) -> bool:
    """Snapshot iff metric clears threshold, cooldown elapsed, and room remains."""
    return metric >= threshold and iters_since_last >= cooldown and pop_size < cap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -S -m "$(cat <<'EOF'
feat(rllib): league membership/metric/snapshot pure helpers

population_members + next_version (derive membership from module-id naming),
read_metric (recursive lookup in the train result), should_snapshot (threshold
+ cooldown + cap predicate). Pure + unit-tested.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B2: League `policy_mapping_fn` factory

**Files:**
- Modify: `rllib/league.py`
- Modify: `tests/rllib/test_league.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_league.py`:

```python
import types


def _episode(eid):
    return types.SimpleNamespace(id_=eid)


def test_mapping_is_live_when_populations_empty():
    fn = L.make_league_mapping_fn({"main_red", "main_blue"}, {"live_fraction": 0.5})
    for eid in range(50):
        ep = _episode(eid)
        assert fn("red_0", ep) == "main_red"
        assert fn("blue_0", ep) == "main_blue"


def test_mapping_both_agents_coherent_per_episode():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})  # always exploit
    for eid in range(50):
        ep = _episode(eid)
        red, blue = fn("red_0", ep), fn("blue_0", ep)
        # Exactly one side is a frozen pop member; the other is its live main.
        red_exploits = red == "main_red" and blue == "blue_pop_v1"
        blue_exploits = red == "red_pop_v1" and blue == "main_blue"
        assert red_exploits or blue_exploits


def test_mapping_respects_live_fraction():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.5})
    n = 4000
    live = sum(
        1 for eid in range(n)
        if fn("red_0", _episode(eid)) == "main_red"
        and fn("blue_0", _episode(eid)) == "main_blue"
    )
    assert 0.40 < live / n < 0.60   # ~0.5 within tolerance


def test_mapping_empty_pop_falls_back_to_live():
    # Only red_pop exists -> "red exploits" (needs blue_pop) can't happen;
    # blue can still exploit red_pop. Never routes to a nonexistent member.
    ids = {"main_red", "main_blue", "red_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})
    for eid in range(50):
        ep = _episode(eid)
        red, blue = fn("red_0", ep), fn("blue_0", ep)
        assert blue == "main_blue"                 # blue never frozen (no blue_pop)
        assert red in ("main_red", "red_pop_v1")   # live or blue-exploits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rllib/test_league.py -k mapping -v`
Expected: FAIL with `AttributeError: module 'rllib.league' has no attribute 'make_league_mapping_fn'`.

- [ ] **Step 3: Add the mapping fn factory to `rllib/league.py`**

Append to `rllib/league.py`:

```python
def _episode_rng_roll(episode) -> tuple[float, float]:
    """Two deterministic [0,1) draws keyed on the episode id, so both agents in
    one episode see the same matchup (zlib.crc32 is stable across processes,
    unlike hash() under PYTHONHASHSEED)."""
    seed = zlib.crc32(str(getattr(episode, "id_", episode)).encode())
    roll_mode = (seed % 1_000_003) / 1_000_003
    roll_side = ((seed // 1_000_003) % 1_000_003) / 1_000_003
    return roll_mode, roll_side


def make_league_mapping_fn(module_ids: Iterable[str], league_cfg: dict):
    """Build the per-episode matchup fn (uniform sampling, Step 3).

    With prob `live_fraction`: main_red vs main_blue (both learn). Otherwise
    split 50/50: 'red exploits' (main_red vs uniform blue_pop member) or
    'blue exploits' (uniform red_pop member vs main_blue). An empty opposite
    population makes that exploit mode fall back to live. Closes over plain
    lists/floats only, so it pickles across the Ray boundary.
    """
    red_pop = population_members(module_ids, RED_POP_RE)
    blue_pop = population_members(module_ids, BLUE_POP_RE)
    live_fraction = float(league_cfg.get("live_fraction", 0.5))

    def league_mapping_fn(agent_id, episode, **kw):
        roll_mode, roll_side = _episode_rng_roll(episode)
        mode = "live"
        if roll_mode >= live_fraction:
            if roll_side < 0.5:
                mode = "red_exploits" if blue_pop else "live"
            else:
                mode = "blue_exploits" if red_pop else "live"
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        if mode == "red_exploits":
            if agent_id == "red_0":
                return "main_red"
            return blue_pop[int(roll_side * 2 * len(blue_pop)) % len(blue_pop)]
        # blue_exploits
        if agent_id == "red_0":
            return red_pop[int(roll_side * 2 * len(red_pop)) % len(red_pop)]
        return "main_blue"

    return league_mapping_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rllib/test_league.py -k mapping -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -S -m "$(cat <<'EOF'
feat(rllib): league policy_mapping_fn factory (uniform sampling)

Per-episode matchup keyed on episode id (stable crc32): live main-vs-main with
prob live_fraction, else 50/50 red/blue exploits a uniform frozen opposite-pop
member; empty pop falls back to live. Picklable closure for the Ray boundary.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B3: `LeagueCallback` (snapshot driver + restore reconstruction)

**Files:**
- Modify: `rllib/league.py`
- Modify: `tests/rllib/test_league.py`

- [ ] **Step 1: Write the failing test (fake-algorithm unit test of the snapshot decision)**

Append to `tests/rllib/test_league.py`:

```python
class _FakeModule:
    def __init__(self, ids):
        self._ids = set(ids)
    def keys(self):
        return set(self._ids)


class _FakeLearnerGroup:
    def foreach_learner(self, fn):
        return []


class _FakeEnvRunnerGroup:
    def __init__(self):
        self.synced = []
        self.refreshed = 0
    def sync_weights(self, **kw):
        self.synced.append(kw.get("policies"))
    def foreach_env_runner(self, fn, **kw):
        self.refreshed += 1
        return []


class _FakeAlgo:
    def __init__(self, ids, iteration, league_cfg):
        self._module = _FakeModule(ids)
        self.iteration = iteration
        self.config = types.SimpleNamespace(env_config={"league": league_cfg})
        self.learner_group = _FakeLearnerGroup()
        self.env_runner_group = _FakeEnvRunnerGroup()
        self.added = []
    def get_module(self, module_id=None):
        return self._module
    def add_module(self, *, module_id, module_spec, new_should_module_be_updated,
                   new_agent_to_module_mapping_fn, **kw):
        self.added.append(module_id)
        self._module._ids.add(module_id)  # reflect the new member


_LEAGUE_CFG = {"snapshot_threshold": 0.7, "min_iters_between_snapshots": 20,
               "population_cap": 5, "live_fraction": 0.5}


def test_callback_snapshots_red_when_threshold_cleared():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)            # baselines cooldown at iter 25
    algo.iteration = 50                              # 25 iters later (> cooldown)
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.8,
                                               "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added
    assert "blue_pop_v1" not in algo.added           # blue below threshold
    assert algo.env_runner_group.synced == [["red_pop_v1"]]


def test_callback_respects_cooldown():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=10, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 19                               # < cooldown (20)
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.95}})
    assert algo.added == []


def test_callback_reconstructs_membership_and_refreshes_on_init():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "red_pop_v1", "red_pop_v2"},
                     iteration=100, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    assert algo.env_runner_group.refreshed == 1       # mapping fn reinstalled
    # next red snapshot would be v3 (derived from restored module ids)
    assert L.next_version(algo.get_module().keys(), L.RED_POP_RE) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rllib/test_league.py -k callback -v`
Expected: FAIL with `AttributeError: module 'rllib.league' has no attribute 'LeagueCallback'`.

- [ ] **Step 3: Add `LeagueCallback` + the mapping-refresh helper to `rllib/league.py`**

Add the import near the top of `rllib/league.py` (after the stdlib imports):

```python
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
```

Append to `rllib/league.py`:

```python
def _refresh_mapping_fn(algorithm, mapping_fn) -> None:
    """Reinstall the league mapping fn on every env runner. Needed on restore,
    where the modules are present but the config's serialized mapping fn still
    reflects empty populations. During normal snapshotting, add_module's
    new_agent_to_module_mapping_fn handles the refresh instead.
    """
    def _set(runner, fn=mapping_fn):
        # The env runner reads self.config.policy_mapping_fn when creating each
        # episode; overwrite it in place. If the config is frozen in this Ray
        # version, fall back to a thawed copy.
        try:
            runner.config.policy_mapping_fn = fn
        except Exception:
            cfg = runner.config.copy(copy_frozen=False)
            cfg.policy_mapping_fn = fn
            runner.config = cfg

    algorithm.env_runner_group.foreach_env_runner(_set, local_env_runner=True)


_SIDES = (
    ("red", "red_score_rate", "main_red", RED_POP_RE),
    ("blue", "blue_prevention_rate", "main_blue", BLUE_POP_RE),
)


class LeagueCallback(RLlibCallback):
    """Drives snapshot-population growth. Reads the in-training metrics off the
    train result and freezes a snapshot of a main into its population when its
    win-rate clears the threshold (with a cooldown + population cap). Membership
    is derived from module ids, so restore is faithful with no extra state."""

    def __init__(self):
        super().__init__()
        self._cfg: dict = {}
        self._last_snapshot_iter = {"red": 0, "blue": 0}

    def _league_cfg(self, algorithm) -> dict:
        return dict(algorithm.config.env_config.get("league", {}))

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._last_snapshot_iter = {"red": it, "blue": it}
        module_ids = set(algorithm.get_module().keys())
        _refresh_mapping_fn(algorithm, make_league_mapping_fn(module_ids, self._cfg))

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        for side, metric_name, main_id, regex in _SIDES:
            metric = read_metric(result, metric_name)
            if metric is None:
                continue
            module_ids = set(algorithm.get_module().keys())
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg["snapshot_threshold"]),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(population_members(module_ids, regex)),
                cap=int(self._cfg["population_cap"]),
            ):
                self._snapshot(algorithm, main_id, regex, module_ids)
                self._last_snapshot_iter[side] = it

    def _snapshot(self, algorithm, main_id, regex, module_ids) -> None:
        new_id = (
            f"red_pop_v{next_version(module_ids, regex)}" if regex is RED_POP_RE
            else f"blue_pop_v{next_version(module_ids, regex)}"
        )
        new_mapping_fn = make_league_mapping_fn(module_ids | {new_id}, self._cfg)
        # Add a fresh (default-arch) module, kept out of the gradient update, and
        # refresh the mapping fn across the env-runner group in the same call.
        algorithm.add_module(
            module_id=new_id,
            module_spec=RLModuleSpec(),
            new_should_module_be_updated=["main_red", "main_blue"],
            new_agent_to_module_mapping_fn=new_mapping_fn,
        )
        # Copy the live main's weights into the frozen snapshot on the learner(s)...
        algorithm.learner_group.foreach_learner(
            lambda lrnr, m=main_id, n=new_id: lrnr.module[n].set_state(
                lrnr.module[m].get_state()
            )
        )
        # ...then push the snapshot's weights out to all env runners.
        algorithm.env_runner_group.sync_weights(
            policies=[new_id],
            from_worker_or_learner_group=algorithm.learner_group,
            inference_only=True,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rllib/test_league.py -k callback -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the whole league unit suite**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: PASS (all helper + mapping + callback tests).

- [ ] **Step 6: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -S -m "$(cat <<'EOF'
feat(rllib): LeagueCallback — threshold snapshotting + restore reconstruction

on_train_result freezes a main into its population (add_module + copy weights on
the learner + sync to env runners) when its in-training win-rate clears the
threshold (cooldown + cap guarded). on_algorithm_init rebuilds membership from
module ids and reinstalls the league mapping fn, so restore is faithful.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B4: League config files

**Files:**
- Create: `conf/league/default.yaml`
- Create: `conf/league/disabled.yaml`
- Modify: `conf/config.yaml` (register the `league` group in the root defaults)
- Create: `conf/multiagent/red_blue_league.yaml`
- Create: `conf/experiment/rllib_league.yaml`

- [ ] **Step 1: Create `conf/league/default.yaml`**

```yaml
# Step-3 snapshot-population league knobs.
#   snapshot_threshold:          win-rate (red_score_rate / blue_prevention_rate)
#                                a main must clear to freeze a snapshot.
#   min_iters_between_snapshots: cooldown so a main doesn't snapshot every iter.
#   population_cap:              max frozen members per side (stop when full;
#                                pruning is Step 4).
#   live_fraction:               P(main-vs-main episode); rest splits 50/50 into
#                                red/blue exploits a uniform frozen opponent.
enabled: true
snapshot_threshold: 0.7
min_iters_between_snapshots: 20
population_cap: 5
live_fraction: 0.5
```

- [ ] **Step 2: Create the disabled baseline and register the group in the root defaults**

Create `conf/league/disabled.yaml` (the root default — league off unless an experiment opts in; `config_builder` reads no further keys when `enabled` is false):

```yaml
# League off (root default). Step-2 / SB3 runs never grow populations.
enabled: false
```

In `conf/config.yaml`, add the `league` group to the defaults list, immediately after the `tune: default` line (keeping it with the other RLlib-path groups, before `_self_`):

```yaml
  - tune: default                   # RLlib path; ignored by the SB3 trainer
  - league: disabled                # RLlib league (Step 3); off unless an experiment opts in
```

(Registering `league` as a root group is what lets the experiment use `override /league: default`, matching how `multiagent`/`tune`/`algo` are declared. The disabled baseline keeps `cfg.league.enabled == false` for every run that doesn't opt in.)

- [ ] **Step 3: Create `conf/multiagent/red_blue_league.yaml`**

```yaml
# Step-3 league topology: same two trainable mains as red_blue_selfplay, but
# config_builder installs the league policy_mapping_fn + LeagueCallback (which
# grow frozen red_pop_v*/blue_pop_v* members) because cfg.league.enabled is set.
# Populations start empty -> identical to naive self-play until the first
# snapshot fires.
learner_id: red_0
policies_to_train: [main_red, main_blue]
mapping:
  red_0: main_red
  blue_0: main_blue
modules:
  main_red:
    kind: learned
  main_blue:
    kind: learned
```

- [ ] **Step 4: Create `conf/experiment/rllib_league.yaml`**

```yaml
# @package _global_
# RLlib migration Step 3: snapshot-population self-play league. Two trainable
# mains bootstrap main-vs-main (as Step 2); frozen snapshots accumulate as each
# main clears the win-rate threshold, then get mixed in as uniform opponents.
# Uses the defender-aware team_selfplay_v1 reward (InterceptShaping active).
run_name: rllib_league

defaults:
  - override /obs: duel_v1_body_n1
  - override /reward: team_selfplay_v1
  - override /curriculum: fixed_red_start
  - override /multiagent: red_blue_league
  - override /league: default

algo:
  total_timesteps: 5_000_000
seed: 42
```

- [ ] **Step 5: Verify the experiment composes**

Run:
```bash
uv run python -c "
import hydra
from hydra import compose, initialize
with initialize(version_base=None, config_path='conf'):
    cfg = compose(config_name='config', overrides=['+experiment=rllib_league'])
    print('league.enabled =', cfg.league.enabled)
    print('multiagent =', cfg.multiagent.mapping)
    print('reward terms =', [t['_target_'].split('.')[-1] for t in cfg.reward.terms])
"
```
Expected: prints `league.enabled = True`, the red/blue mapping, and a reward-term list that **includes** `InterceptShaping`. (`config` is the root config name — `conf/config.yaml`, the Hydra entrypoint referenced by `scripts/train_rllib.py`.)

- [ ] **Step 6: Commit**

```bash
git add conf/league/default.yaml conf/league/disabled.yaml conf/config.yaml conf/multiagent/red_blue_league.yaml conf/experiment/rllib_league.yaml
git commit -S -m "$(cat <<'EOF'
feat(conf): league config group + red_blue_league + rllib_league experiment

conf/league/default.yaml (threshold/cooldown/cap/live_fraction), the league
multiagent topology, and the Step-3 experiment composing the defender-aware
team_selfplay_v1 reward.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B5: Wire `LeagueCallback` + league mapping fn into `config_builder`

**Files:**
- Modify: `rllib/config_builder.py` (imports; `build_ppo_config` lines 61-132)
- Test: `tests/rllib/test_config_builder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/rllib/test_config_builder.py`:

```python
def _league_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
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
        "league": {"enabled": True, "snapshot_threshold": 0.7,
                   "min_iters_between_snapshots": 20, "population_cap": 5,
                   "live_fraction": 0.5},
    })


def test_league_callback_and_mapping_wired_when_enabled():
    from rllib.config_builder import build_ppo_config
    from rllib.league import LeagueCallback
    cfg = _league_cfg()
    config = build_ppo_config(cfg)
    # LeagueCallback is among the registered callbacks.
    cbs = config.callbacks_class
    classes = cbs if isinstance(cbs, (list, tuple)) else [cbs]
    assert LeagueCallback in classes
    # The league knobs are stashed in env_config for the callback to read.
    assert config.env_config["league"]["snapshot_threshold"] == 0.7
    # The mapping fn routes the two mains live when populations are empty.
    import types
    fn = config.policy_mapping_fn
    ep = types.SimpleNamespace(id_=7)
    assert fn("red_0", ep) == "main_red"
    assert fn("blue_0", ep) == "main_blue"


def test_no_league_callback_when_league_absent():
    """Step-2 configs (no `league` group) keep the static mapping + score-only
    callback — league wiring is opt-in."""
    from rllib.config_builder import build_ppo_config
    from rllib.league import LeagueCallback
    cfg = _league_cfg()
    del cfg["league"]
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    classes = cbs if isinstance(cbs, (list, tuple)) else [cbs]
    assert LeagueCallback not in classes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rllib/test_config_builder.py -k league -v`
Expected: FAIL — `build_ppo_config` always registers only `ScoreMetricsCallback` and the static dict mapping; `LeagueCallback not in classes`.

- [ ] **Step 3: Wire the league path in `config_builder.py`**

Add to the imports block (after line 19, `from rllib.metrics import ScoreMetricsCallback`):

```python
from rllib.league import LeagueCallback, make_league_mapping_fn
```

In `build_ppo_config`, replace the static mapping fn definition (lines 92-93):

```python
    def policy_mapping_fn(agent_id, episode, **kw):
        return mapping[agent_id]
```

with league-aware selection:

```python
    league_cfg = cfg.get("league")
    league_on = bool(league_cfg is not None and league_cfg.get("enabled"))
    if league_on:
        league_dict = OmegaConf.to_container(league_cfg, resolve=True)
        # Populations start empty: only the two mains exist at build time.
        policy_mapping_fn = make_league_mapping_fn(set(ma.modules.keys()), league_dict)
        callbacks = [ScoreMetricsCallback, LeagueCallback]
    else:
        league_dict = None

        def policy_mapping_fn(agent_id, episode, **kw):
            return mapping[agent_id]

        callbacks = ScoreMetricsCallback
```

Replace the `.callbacks(ScoreMetricsCallback)` call (line 111) with:

```python
        .callbacks(callbacks)
```

Add a `"league"` entry into the `env_config` dict (inside the `.environment(...)` call, lines 103-108) so `LeagueCallback` can read its knobs off `algorithm.config.env_config`:

```python
            env_config={
                "learner_id": ma.learner_id,
                "obs_blocks": obs_blocks,
                "team_cfg": _team_cfg_from(cfg),
                "reward_stack": reward_stack,
                "league": league_dict,
            },
```

(`make_team_env` reads only its specific keys, so the extra `"league"` entry — `None` when the league is off — is inert for the env.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rllib/test_config_builder.py -k league -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full config-builder + self-play unit suites for no regression**

Run: `uv run pytest tests/rllib/test_config_builder.py tests/rllib/test_self_play.py -v`
Expected: PASS (the Step-2 paths still build; `policies_to_train` and static-mapping behavior unchanged when `league` is absent).

- [ ] **Step 6: Commit**

```bash
git add rllib/config_builder.py tests/rllib/test_config_builder.py
git commit -S -m "$(cat <<'EOF'
feat(rllib): wire LeagueCallback + league mapping fn into config_builder

When cfg.league.enabled, build_ppo_config installs the league policy_mapping_fn
(empty pops at build), registers [ScoreMetricsCallback, LeagueCallback], and
stashes the league knobs in env_config for the callback. Step-2 configs without
a league group keep the static mapping + score-only callback.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B6: Integration smoke — snapshot fires, freezes, checkpoints, restores

**Files:**
- Modify: `tests/rllib/test_smoke_train.py` (append)

- [ ] **Step 1: Write the failing slow test**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_league_snapshots_freezes_and_restores(tmp_path):
    """A league run with the threshold forced low takes a snapshot: the frozen
    module's weights match the main at snapshot time, do NOT change across the
    next iteration, and survive a checkpoint round-trip with membership intact."""
    import numpy as np
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.league import RED_POP_RE, BLUE_POP_RE, population_members
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
        # threshold 0 + cooldown 0 -> a snapshot fires on the first iteration
        # for whichever side records a metric.
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()  # iteration 1: metrics logged -> at least one snapshot added
        ids = set(algo.get_module().keys())
        pops = population_members(ids, RED_POP_RE) + population_members(ids, BLUE_POP_RE)
        assert pops, "expected at least one frozen snapshot after iter 1"
        snap_id = pops[0]
        main_id = "main_red" if snap_id.startswith("red") else "main_blue"

        # Snapshot weights equal the main's weights now (copied at snapshot time).
        def _flat(state):
            return np.concatenate([np.ravel(v) for v in state.values()
                                   if hasattr(v, "shape")])
        snap0 = _flat(algo.get_module(snap_id).get_state())

        algo.train()  # iteration 2: main updates; frozen snapshot must NOT.
        snap1 = _flat(algo.get_module(snap_id).get_state())
        assert np.allclose(snap0, snap1), "frozen snapshot weights changed"

        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    # Restore into a fresh algo: membership is reconstructed from module ids.
    algo2 = build_ppo_config(cfg).build_algo()
    try:
        algo2.restore_from_path(ckpt)
        ids2 = set(algo2.get_module().keys())
        assert snap_id in ids2, "snapshot module missing after restore"
    finally:
        algo2.stop()
```

- [ ] **Step 2: Run test to verify it fails (or errors) before the league path is exercised end-to-end**

Run: `uv run pytest tests/rllib/test_smoke_train.py::test_league_snapshots_freezes_and_restores -v -m slow`
Expected: This is the integration gate for Tasks B3+B5. If a real-RLlib API detail differs from what was coded (the `set_state`/`sync_weights`/`add_module` calls in `_snapshot`, or the `_refresh_mapping_fn` env-runner config write), this is where it surfaces. Run it and read the failure.

- [ ] **Step 3: Resolve any API mismatches surfaced by the run**

The following were confirmed present on Ray 2.55.1 and should work as written: `Algorithm.add_module(...)`, `Algorithm.get_module()` / `.get_module(id)`, `MultiRLModule.keys()`, `LearnerGroup.foreach_learner`, `learner.module[id].get_state()/set_state(...)`, `EnvRunnerGroup.sync_weights(policies=..., from_worker_or_learner_group=..., inference_only=True)`, `EnvRunnerGroup.foreach_env_runner(fn, local_env_runner=True)`. If the run reveals a deviation:
- **Weight copy:** if `learner.module[id]` is not subscriptable, use `learner.module._rl_modules[id]` or `learner.module.get_module(id)`.
- **Env-runner config write:** if `runner.config.policy_mapping_fn = fn` raises even with the thawed-copy fallback, set it via `runner.config.multi_agent(policy_mapping_fn=fn)` on a `copy(copy_frozen=False)`.
Apply the minimal fix in `rllib/league.py`, re-run, and keep iterating until the test passes. (Do not weaken the assertions — they encode the design contract.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rllib/test_smoke_train.py::test_league_snapshots_freezes_and_restores -v -m slow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/rllib/test_smoke_train.py rllib/league.py
git commit -S -m "$(cat <<'EOF'
test(rllib): integration smoke — league snapshot, freeze, checkpoint, restore

A few-iteration league run with the threshold forced low: a frozen snapshot is
added, its weights equal the main at snapshot time and stay fixed across the
next iteration, and a checkpoint round-trip reconstructs membership from module
ids. Gates the add_module/weight-copy/sync mechanism end-to-end.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full fast suite** (excludes slow/render):

Run: `uv run pytest -m "not slow" -q`
Expected: PASS (no regressions in env, rewards, or rllib unit tests).

- [ ] **Run the RLlib slow suite** (Ray plumbing + league integration):

Run: `uv run pytest tests/rllib/ -v -m slow`
Expected: PASS (`test_cartpole_one_iteration`, `test_team_skeleton_trains_and_restores`, `test_league_snapshots_freezes_and_restores`).

- [ ] **Behavioral validation (user-driven, before merge):** a short `rllib_league` run on real hardware to confirm (a) `InterceptShaping` produces a non-trivial Blue learning signal (Blue defends without the Step-2 reward-magnitude asymmetry collapsing it), and (b) at least one snapshot is added once a main clears 0.7, and training stays stable with frozen opponents mixed in. Per project convention, do NOT claim the league "works" or merge to `develop` until this real-run is confirmed by the user. Capture the run id + observations in `brain/changelog.md`.

---

## Notes for the executor

- **Scope of the `learner_id` change (Part A):** only the *intercept-shaping reward* inputs are de-gated. The `closing_rate` *obs* feature (`team_env.py:659-665`) and the `_spec_for_agent` obs-spec selection stay learner-gated — obs layout and checkpoint compatibility are deliberately untouched (brainstorming decision Q2).
- **Why module-id naming is the membership store (Part B):** RLlib checkpoints every module in the `MultiRLModule`, so frozen `*_pop_v*` modules persist and restore automatically. Deriving membership from their ids in `on_algorithm_init` means no separate league-state file and no drift between "what modules exist" and "what the mapping fn thinks exists."
- **Cooldown after restore:** `on_algorithm_init` rebaselines `_last_snapshot_iter` to the current iteration, so the next snapshot is delayed by at most `min_iters_between_snapshots` after a resume. Accepted (bounded, harmless).
- **Deferred to Step 4:** PFSP prioritized sampling, population pruning/eviction, and a dedicated head-to-head eval battery for the snapshot trigger. Step 3 uses uniform sampling, a stop-when-full cap, and the in-training metrics.
```
