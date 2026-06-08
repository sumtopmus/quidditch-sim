# RLlib Two-Policy Self-Play Implementation Plan (Migration Step 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `main_blue` a second *trainable* PPO policy and train it simultaneously against `main_red` on `QuidditchTeamEnv` — naive two-policy self-play — proving gradients flow to both policies, the `policy_mapping_fn` routes each agent to its own policy, and the two co-adapt under a competitive (zero-sum-on-score) reward.

**Architecture:** Step 1 left a fully general config bridge: `rllib/config_builder.py` already builds per-policy `RLModuleSpec`s from `cfg.multiagent.modules` and sets `policies_to_train` from config, so flipping `main_blue` from a frozen `ScriptedRLModule` to a learned default `RLModule` is a *config* change, not a code change. The new work is: a competitive two-policy reward stack, a generalized metrics callback that reports both Red's offense and Blue's defense, and a gradient-flow proof test. No env, simulator, or obs-spec changes — both policies use the existing 22-d `DUEL_V1_BODY` obs (the env already computes each agent's obs relative to its own goal target, so Red and Blue see role-appropriate observations from the same block list).

**Tech Stack:** Ray RLlib (new API stack: RLModule + Learner + ConnectorV2), Ray Tune, PyTorch, Hydra, Weights & Biases, MuJoCo, Python 3.13 (uv venv).

**Scope note:** This is **Step 2 of 6** from the spec (`docs/superpowers/specs/2026-06-07-rllib-league-migration-design.md`, build sequence #2). It deliberately stops before any *population* machinery: there is exactly one trainable policy per side, training live against the other (`main_red` vs `main_blue`, both updating every iteration). Snapshotting, frozen populations, PFSP sampling, and lagged-opponent variants are **Steps 3–4**, each its own plan. SB3 is **not** removed (Step 6). The skeleton's stabilizer (`entropy_coeff_schedule`, `grad_clip`, `lr 3e-5` in `conf/algo/ppo_rllib.yaml`) and the potential-based `HoopApproachShaping` reward carry over unchanged.

**Deferred to Step 3+ (explicitly out of scope, with reasons):**
- **`InterceptShaping` for Blue.** Its StepState inputs (`dist_def_to_future_red[_prev]`) are populated in `team_env.step` only for the single configured `learner_id`, computing the *learner's* distance to future-Red — correct only when the learner IS the defender. Two-policy self-play has no single learner, so wiring Blue's intercept reward correctly needs a defender-aware env refactor. Step 2 uses Blue reward terms that need no such plumbing (`HoopAnchor`, `TakeDown`, zero-sum `ScoreEvent`, `CrashEvent`). The refactor + `InterceptShaping` re-introduction is noted as a Step 3 prerequisite.
- **Lagged / frozen-copy opponents and populations** — Step 3.
- **Per-policy obs/net asymmetry** — both policies use `DUEL_V1_BODY` + the default PPO MLP for now.

---

## File Structure

**New files:**
- `conf/reward/team_selfplay_v1.yaml` — competitive two-policy reward stack. Red: `HoopApproachShaping` + zero-sum `ScoreEvent`. Blue: `HoopAnchor` + `TakeDown`. Both: `CrashEvent`. One responsibility: the Step-2 reward magnitudes + composition.
- `conf/multiagent/red_blue_selfplay.yaml` — both `main_red` and `main_blue` `kind: learned`; `policies_to_train: [main_red, main_blue]`. One responsibility: the two-policy topology.
- `conf/experiment/rllib_selfplay.yaml` — composes `red_blue_selfplay` + `team_selfplay_v1` + the fixed Red start, overrides timesteps/seed. One responsibility: the runnable Step-2 experiment.
- `tests/rllib/test_self_play.py` — config-builder two-trainable-policy contract + the gradient-flow proof (both modules' weights change after one train iteration; frozen-none invariant). One responsibility: the Step-2 behavioral guarantees.

**Modified files:**
- `rllib/metrics.py` — generalize the per-episode aggregation to also track Blue defense (`blue_prevention_rate`, `blue_min_dist_to_red`); `ScoreMetricsCallback` logs both sides. (Red metrics unchanged in name/meaning.)
- `envs/quidditch/team_env.py` — add `dist_b2r` to `infos["blue_0"]` (one line; mirrors the `dist_red_to_hoop` already in `infos["red_0"]`) so the callback can track Blue's closest approach to Red without re-deriving it from obs.
- `tests/rllib/test_metrics.py` — extend for the Blue aggregation helpers.
- `tests/envs/quidditch/test_hoop_progress_reward.py` — add a one-test assertion that `infos["blue_0"]["dist_b2r"]` is exposed (mirrors the existing `dist_red_to_hoop` info test).

---

## Task 1: Competitive two-policy reward stack

A reward where Red and Blue have opposed objectives: Red is rewarded for approaching + scoring (and scoring *penalizes* Blue, zero-sum); Blue is rewarded for guarding the hoop + taking Red down. Both are penalized for crashing/OOB. Uses only existing reward terms — no new term code — so the env's single-`learner_id` `InterceptShaping` plumbing is sidestepped (see Scope note).

**Files:**
- Create: `conf/reward/team_selfplay_v1.yaml`
- Test: `tests/envs/quidditch/rewards/test_reward_stack.py` (append one composition test)

- [ ] **Step 1: Write the failing composition test**

Append to `tests/envs/quidditch/rewards/test_reward_stack.py`:

```python
def test_team_selfplay_v1_stack_composition():
    """conf/reward/team_selfplay_v1.yaml: Red dense-approach + zero-sum score;
    Blue anchor + takedown; both crash-penalised. No InterceptShaping (needs
    defender-aware env plumbing not present until Step 3)."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_selfplay_v1")
    assert [type(t).__name__ for t in stack.terms] == [
        "HoopApproachShaping", "ScoreEvent", "HoopAnchor", "TakeDown", "CrashEvent",
    ]
    shaping = stack.terms[0]
    assert shaping.scale == 2.0 and shaping.agent == "red_0"
    score = stack.terms[1]
    assert score.magnitude == 10.0
    assert score.scorer == "red_0"
    assert score.zero_sum_opponent == "blue_0"   # Blue loses 10 when Red scores
    assert "InterceptShaping" not in [type(t).__name__ for t in stack.terms]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/envs/quidditch/rewards/test_reward_stack.py::test_team_selfplay_v1_stack_composition -q`
Expected: FAIL with `FileNotFoundError: .../conf/reward/team_selfplay_v1.yaml`.

- [ ] **Step 3: Create the reward YAML**

Create `conf/reward/team_selfplay_v1.yaml`:

```yaml
# Competitive two-policy self-play reward (RLlib migration Step 2).
#
# Red (attacker): the dense potential-based hoop pull from the skeleton
# (HoopApproachShaping) + the +10 score, made ZERO-SUM so Blue is penalised -10
# whenever Red scores -> Blue has a direct incentive to prevent.
# Blue (defender): HoopAnchor keeps it near the hoop; TakeDown rewards ramming
# Red out of the sky.  (InterceptShaping is intentionally omitted: its StepState
# inputs are only populated for a single env learner_id, which two-policy
# self-play does not have -- deferred to Step 3 with a defender-aware refactor.)
# Both: CrashEvent for floor/wall/OOB.
_target_: envs.quidditch.rewards.stack.RewardStack
terms:
  - _target_: envs.quidditch.rewards.terms.HoopApproachShaping
    scale: 2.0
    agent: red_0

  - _target_: envs.quidditch.rewards.terms.ScoreEvent
    magnitude: 10.0
    scorer: red_0
    zero_sum_opponent: blue_0

  - _target_: envs.quidditch.rewards.terms.HoopAnchor
    scale: 0.005
    agents: [blue_0]

  - _target_: envs.quidditch.rewards.terms.TakeDown
    aggressor_reward: 20.0
    victim_penalty: -20.0
    aggressor: blue_0
    victim: red_0

  - _target_: envs.quidditch.rewards.terms.CrashEvent
    magnitude: -10.0
    agent_to_crash_flags:
      red_0: [red_floor, red_wall_crash, red_oob]
      blue_0: [blue_floor, blue_wall_crash, blue_oob]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run python -m pytest tests/envs/quidditch/rewards/test_reward_stack.py::test_team_selfplay_v1_stack_composition -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/reward/team_selfplay_v1.yaml tests/envs/quidditch/rewards/test_reward_stack.py
git commit -S -m "feat(reward): competitive two-policy self-play reward stack (team_selfplay_v1)"
```

---

## Task 2: Two-policy multiagent topology + builder contract

Flip `main_blue` from a frozen `ScriptedRLModule` to a learned policy and add it to `policies_to_train`. The builder already loops over `cfg.multiagent.modules`, so this is a config change plus a contract test proving both policies are trainable and routed correctly.

**Files:**
- Create: `conf/multiagent/red_blue_selfplay.yaml`
- Create: `tests/rllib/test_self_play.py`

- [ ] **Step 1: Write the failing builder-contract test**

Create `tests/rllib/test_self_play.py`:

```python
"""Step 2 two-policy self-play: both mains trainable + gradient flow."""
from __future__ import annotations

from omegaconf import OmegaConf

from rllib.config_builder import build_ppo_config


def _selfplay_cfg():
    """A bare cfg shaped like the composed experiment, both mains learned."""
    return OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY_MIXED",
        ]},
        "algo": {
            "lr": 3e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
            "entropy_coeff": 0.0, "num_epochs": 1, "minibatch_size": 64,
            "train_batch_size_per_learner": 256, "num_env_runners": 0,
            "total_timesteps": 256, "grad_clip": 1.0,
        },
        "multiagent": {
            "learner_id": "red_0",
            "policies_to_train": ["main_red", "main_blue"],
            "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
            "modules": {
                "main_red": {"kind": "learned"},
                "main_blue": {"kind": "learned"},
            },
        },
        "reward": None,
    })


def test_both_mains_are_trainable_and_routed():
    config = build_ppo_config(_selfplay_cfg())
    assert set(config.policies) == {"main_red", "main_blue"}
    assert sorted(config.policies_to_train) == ["main_blue", "main_red"]
    assert config.policy_mapping_fn("red_0", None) == "main_red"
    assert config.policy_mapping_fn("blue_0", None) == "main_blue"
```

- [ ] **Step 2: Run it to verify it fails for the right reason**

Run: `uv run python -m pytest tests/rllib/test_self_play.py::test_both_mains_are_trainable_and_routed -q`
Expected: PASS already if the builder is general — but first confirm the config group exists for the *experiment* path. The unit test above constructs its cfg inline, so it should PASS immediately (proving the builder needs no change). If it FAILS, the builder is not general over two learned modules and Task 2 must fix `build_ppo_config`; investigate before proceeding.

> Note: this is the rare case where the contract is already satisfied by Step 1's general builder. The test exists to *lock that in* and to fail loudly if a later refactor breaks two-policy support. Proceed to create the config group (Step 3) regardless — the experiment composition needs it.

- [ ] **Step 3: Create the multiagent config group**

Create `conf/multiagent/red_blue_selfplay.yaml`:

```yaml
# Two-policy self-play topology (RLlib migration Step 2): red_0 and blue_0 are
# BOTH learned, trainable policies, training live against each other every
# iteration.  No frozen populations yet (Step 3).  learner_id stays red_0 only
# to select which agent the env feeds the (here identical) learner obs spec to;
# both agents use DUEL_V1_BODY so it is immaterial.
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

- [ ] **Step 4: Run the contract test to confirm it passes**

Run: `uv run python -m pytest tests/rllib/test_self_play.py::test_both_mains_are_trainable_and_routed -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/multiagent/red_blue_selfplay.yaml tests/rllib/test_self_play.py
git commit -S -m "feat(rllib): two-policy self-play multiagent topology (red_blue_selfplay)"
```

---

## Task 3: Expose Blue's distance-to-Red + generalize the metrics callback

The skeleton callback logs only Red's offense (`red_score_rate`, `red_min_dist_to_hoop`). For self-play we also want Blue's defense legible: how often Blue prevents a score, and how close Blue gets to Red (its takedown opportunity). Add `dist_b2r` to Blue's info dict, and generalize the pure aggregation + callback to log both sides.

**Files:**
- Modify: `envs/quidditch/team_env.py` (Blue info dict)
- Modify: `rllib/metrics.py`
- Modify: `tests/rllib/test_metrics.py`
- Modify: `tests/envs/quidditch/test_hoop_progress_reward.py`

- [ ] **Step 1: Write the failing env-info test**

Append to `tests/envs/quidditch/test_hoop_progress_reward.py`:

```python
def test_step_info_exposes_blue_distance_to_red():
    """infos['blue_0']['dist_b2r'] lets the metrics callback track Blue's
    closest approach to Red (its takedown opportunity)."""
    import numpy as np
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, red_start_pos=(0.5, 0.0, 2.0)))
    try:
        env.reset(seed=0)
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        expected = float(np.linalg.norm(env._red_pos() - env._blue_pos()))
        assert infos["blue_0"]["dist_b2r"] == expected
    finally:
        env.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/envs/quidditch/test_hoop_progress_reward.py::test_step_info_exposes_blue_distance_to_red -q`
Expected: FAIL with `KeyError: 'dist_b2r'`.

- [ ] **Step 3: Add `dist_b2r` to Blue's info dict**

In `envs/quidditch/team_env.py`, find the `infos[self._blue_id].update({...})` block in `step()` (it currently carries `scored`, `drone_drone_crash`, `blue_floor`, `blue_wall_crash`, `blue_oob`, `step`). Add the distance (the local `dist_b2r` is already computed earlier in `step()`):

```python
        infos[self._blue_id].update({
            "scored": scored, "drone_drone_crash": drone_drone_crash,
            "blue_floor": blue_floor, "blue_wall_crash": blue_wall_crash,
            "blue_oob": blue_oob, "step": self._step_count,
            "dist_b2r": dist_b2r,
        })
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run python -m pytest tests/envs/quidditch/test_hoop_progress_reward.py::test_step_info_exposes_blue_distance_to_red -q`
Expected: PASS.

- [ ] **Step 5: Write the failing Blue-aggregation tests**

Append to `tests/rllib/test_metrics.py`:

```python
from rllib.metrics import (
    blue_episode_metrics, init_blue_acc, update_blue_acc,
)


def test_blue_acc_tracks_min_dist_to_red_and_prevention():
    acc = init_blue_acc()
    update_blue_acc(acc, {"dist_b2r": 1.2, "scored": False})
    update_blue_acc(acc, {"dist_b2r": 0.3, "scored": False})
    m = blue_episode_metrics(acc)
    assert m["blue_min_dist_to_red"] == 0.3
    assert m["blue_prevention_rate"] == 1.0   # Red did not score this episode


def test_blue_prevention_rate_zero_when_red_scores():
    acc = init_blue_acc()
    update_blue_acc(acc, {"dist_b2r": 0.8, "scored": True})
    m = blue_episode_metrics(acc)
    assert m["blue_prevention_rate"] == 0.0
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run python -m pytest tests/rllib/test_metrics.py -k blue -q`
Expected: FAIL with `ImportError: cannot import name 'blue_episode_metrics' ...`.

- [ ] **Step 7: Add the Blue aggregation helpers + log them in the callback**

In `rllib/metrics.py`, add the Blue pure helpers next to the Red ones:

```python
def init_blue_acc() -> dict:
    return {"scored": False, "min_dist_to_red": math.inf}


def update_blue_acc(acc: dict, blue_info: dict) -> None:
    dist = blue_info.get("dist_b2r")
    if dist is not None:
        acc["min_dist_to_red"] = min(acc["min_dist_to_red"], float(dist))
    if blue_info.get("scored"):
        acc["scored"] = True


def blue_episode_metrics(acc: dict) -> dict:
    out: dict[str, float] = {"blue_prevention_rate": 0.0 if acc["scored"] else 1.0}
    if math.isfinite(acc["min_dist_to_red"]):
        out["blue_min_dist_to_red"] = acc["min_dist_to_red"]
    return out
```

Add a `_blue_info` extractor mirroring `_red_info` (replace `_LEARNER` with `"blue_0"`):

```python
_BLUE = "blue_0"


def _blue_info(episode) -> dict | None:
    info = episode.get_infos(-1, _BLUE)
    if isinstance(info, dict) and _BLUE in info and isinstance(info[_BLUE], dict):
        info = info[_BLUE]
    return info if isinstance(info, dict) else None
```

Extend `ScoreMetricsCallback` to drive both accumulators:

```python
    def on_episode_start(self, *, episode, **kwargs) -> None:
        episode.custom_data["score_acc"] = init_episode_acc()
        episode.custom_data["blue_acc"] = init_blue_acc()

    def on_episode_step(self, *, episode, **kwargs) -> None:
        red_acc = episode.custom_data.setdefault("score_acc", init_episode_acc())
        blue_acc = episode.custom_data.setdefault("blue_acc", init_blue_acc())
        red = _red_info(episode)
        if red is not None:
            update_episode_acc(red_acc, red)
        blue = _blue_info(episode)
        if blue is not None:
            update_blue_acc(blue_acc, blue)

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        if metrics_logger is None:
            return
        red_acc = episode.custom_data.get("score_acc") or init_episode_acc()
        blue_acc = episode.custom_data.get("blue_acc") or init_blue_acc()
        metrics = {**episode_metrics(red_acc), **blue_episode_metrics(blue_acc)}
        for key, value in metrics.items():
            metrics_logger.log_value(key, value, reduce="mean")
```

- [ ] **Step 8: Run the metrics tests to verify they pass**

Run: `uv run python -m pytest tests/rllib/test_metrics.py -q`
Expected: PASS (Red tests from Step 1 + the 2 new Blue tests).

- [ ] **Step 9: Commit**

```bash
git add envs/quidditch/team_env.py rllib/metrics.py tests/rllib/test_metrics.py tests/envs/quidditch/test_hoop_progress_reward.py
git commit -S -m "feat(rllib): log Blue defense metrics (prevention rate + min dist to Red)"
```

---

## Task 4: Self-play experiment config + Hydra composition

Wire a runnable experiment composing the two-policy topology, the competitive reward, and the fixed Red start, and prove the full Hydra graph resolves with both policies trainable.

**Files:**
- Create: `conf/experiment/rllib_selfplay.yaml`
- Test: `tests/rllib/test_self_play.py` (append a composition test)

- [ ] **Step 1: Write the failing composition test**

Append to `tests/rllib/test_self_play.py`:

```python
def test_selfplay_experiment_composes_two_trainable_policies():
    from hydra import initialize, compose
    from config_schema import register_configs

    register_configs()
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config",
                      overrides=["+experiment=rllib_selfplay"])
    assert sorted(cfg.multiagent.policies_to_train) == ["main_blue", "main_red"]
    assert cfg.multiagent.modules.main_blue.kind == "learned"
    assert cfg.reward._target_.endswith("RewardStack")
    assert cfg.curriculum.randomise_start is False
    assert list(cfg.curriculum.red_start_pos) == [0.5, 0.0, 2.0]
```

> Path note: `config_path` is relative to the test file. `tests/rllib/` → `../../conf`. Adjust only if the repo's test layout differs.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_self_play.py::test_selfplay_experiment_composes_two_trainable_policies -q`
Expected: FAIL — Hydra cannot find `experiment=rllib_selfplay`.

- [ ] **Step 3: Create the experiment YAML**

Create `conf/experiment/rllib_selfplay.yaml`:

```yaml
# @package _global_
# RLlib migration Step 2: two-policy naive self-play. main_red vs main_blue,
# both trainable, on the fixed-start scoring scenario from the skeleton.
run_name: rllib_selfplay

defaults:
  - override /obs: duel_v1_body_n1
  - override /reward: team_selfplay_v1
  - override /curriculum: fixed_red_start
  - override /multiagent: red_blue_selfplay

algo:
  total_timesteps: 2_000_000
seed: 42
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_self_play.py::test_selfplay_experiment_composes_two_trainable_policies -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/experiment/rllib_selfplay.yaml tests/rllib/test_self_play.py
git commit -S -m "feat(rllib): rllib_selfplay experiment (two-policy, competitive reward, fixed start)"
```

---

## Task 5: Gradient-flow proof — both policies learn, neither is frozen

The defining Step-2 guarantee: a real train iteration updates **both** `main_red` and `main_blue` weights (and the checkpoint round-trips). This is an integration test (`@pytest.mark.slow`) building a real algo and training one iteration. It is the load-bearing proof that two-policy gradient flow works on this stack.

**Files:**
- Test: `tests/rllib/test_self_play.py` (append the gradient-flow test)

- [ ] **Step 1: Write the failing gradient-flow test**

Append to `tests/rllib/test_self_play.py`:

```python
import pytest


def _flat_params(algo, module_id):
    import torch
    sd = algo.get_module(module_id).state_dict()
    return torch.cat([t.flatten() for t in sd.values() if t.numel() > 0]).clone()


@pytest.mark.slow
def test_both_policies_receive_gradients(tmp_path):
    """One train iteration changes BOTH main_red and main_blue weights, and the
    checkpoint round-trips. This is the Step-2 dual-gradient-flow guarantee."""
    import torch
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    algo = build_ppo_config(_selfplay_cfg()).build_algo()
    try:
        before_red = _flat_params(algo, "main_red")
        before_blue = _flat_params(algo, "main_blue")
        algo.train()
        after_red = _flat_params(algo, "main_red")
        after_blue = _flat_params(algo, "main_blue")
        assert not torch.allclose(before_red, after_red), "main_red did not update"
        assert not torch.allclose(before_blue, after_blue), "main_blue did not update"
        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
        algo.restore_from_path(ckpt)
    finally:
        algo.stop()
```

- [ ] **Step 2: Run it to verify it fails or errors first**

Run: `uv run python -m pytest tests/rllib/test_self_play.py::test_both_policies_receive_gradients -q`
Expected: With Tasks 2–4 done the builder already produces two learned policies, so this should PASS. Run it explicitly to *confirm dual gradient flow on the real stack* — if either `assert ... did not update` fires, a policy is silently frozen (check `policies_to_train` actually contains both, and that `_selfplay_cfg()` sets both modules `kind: learned`). Run from a regular Terminal window (MuJoCo render init under the env-runner).

- [ ] **Step 3: If it fails, fix the builder; if it passes, proceed**

No code change is expected (Step 1's builder is general). If `main_blue` did not update, the cause is almost certainly `policies_to_train` not including it — verify `build_ppo_config` does `policies_to_train=list(ma.policies_to_train)` and that the cfg lists both. Do not weaken the test; fix the routing.

- [ ] **Step 4: Run the full RLlib + reward suites green**

Run: `uv run python -m pytest tests/rllib/ tests/envs/quidditch/rewards/ -q`
Expected: PASS (all Step-1 tests + the new Step-2 tests; the slow gradient-flow + smoke-train tests included).

- [ ] **Step 5: Commit**

```bash
git add tests/rllib/test_self_play.py
git commit -S -m "test(rllib): prove two-policy self-play gradient flow + checkpoint round-trip"
```

---

## Task 6: "Both co-adapt" validation milestone

A **behavioral verification gate**, not auto-committed code — same pattern as the skeleton's Task 6. The longer run and its judgement happen with the user in the loop (verify behavior before committing; GUI/W&B watching needs the user).

**Files:** none created. Produces a run under `runs/rllib_selfplay/...` and a W&B run.

- [ ] **Step 1: Launch a real (short) self-play run**

Run: `make train-rllib EXP=rllib_selfplay OVERRIDES="algo.total_timesteps=2_000_000"`
Expected: training proceeds; W&B logs `env_runners/red_score_rate`, `env_runners/blue_prevention_rate`, `env_runners/red_min_dist_to_hoop`, `env_runners/blue_min_dist_to_red`, and per-policy learner curves (`learners/main_red/...`, `learners/main_blue/...`); a checkpoint dir appears under `runs/rllib_selfplay/`.

- [ ] **Step 2: Confirm co-adaptation (the signal to look for)**

Watch on W&B (query `wandb.Api()` for exact values — the stdout summary omits most keys, a Step-1 lesson):
- **Both** policies' weights/loss curves are active (not flat) — both are learning.
- `red_score_rate` and `blue_prevention_rate` are **complementary** (`≈ 1 − each other`) and move *against* each other over training — the hallmark of co-adaptation (as Blue improves, Red's score rate dips, then Red adapts, etc.), rather than one side trivially dominating to 0/1 and pinning there.
- Entropy on **both** `main_red` and `main_blue` stays bounded (the `entropy_coeff_schedule` from the skeleton applies to both learners) — no repeat of the skeleton's entropy-explosion collapse.

- [ ] **Step 3: User verification checkpoint**

STOP and report the W&B curves (both score/prevention rates + both entropy curves + the run id) to the user. Do not declare two-policy self-play "working" until the user confirms both policies are visibly co-adapting (curves, or a rendered head-to-head rollout). This proves dual-policy gradient flow + `policy_mapping_fn` end-to-end under RLlib — the Step-2 milestone.

- [ ] **Step 4: (After user confirmation) Record the result**

Once confirmed, note the run id + the co-adaptation trend in `brain/changelog.md` and update `brain/index.md` Current State to "Step 2 (two-policy self-play) green". This unblocks the Step 3 plan (snapshot populations) — and flags its prerequisite: the defender-aware env refactor so `InterceptShaping` can return for Blue.

---

## Self-Review

**Spec coverage (build sequence #2 — "Two-policy simultaneous: add `main_blue`; naive self-play; proves dual-policy gradient flow + `policy_mapping_fn`"):** `main_blue` made trainable ✓ (Task 2). Competitive reward so both have objectives ✓ (Task 1). `policy_mapping_fn` routing both agents ✓ (Task 2 contract test). Dual gradient flow proven ✓ (Task 5). Both-sides metrics ✓ (Task 3). Runnable experiment + validation gate ✓ (Tasks 4, 6). Spec §B1 `q`-fraction main-vs-lagged pairing: **intentionally deferred** — build sequence #2 is "single live opponent"; lagged/population pairing is Step 3 (noted in Scope). Spec §B2 reward anneal, §B3 PFSP, §B4 promotion, §B5 curriculum levers: **Steps 3–5**, out of scope. The `InterceptShaping`/defender-aware-env gap is explicitly flagged for Step 3 (Scope note + Task 6 Step 4).

**Placeholder scan:** No TBD/TODO/"add error handling" — every code step shows complete content. The one judgement gate (Task 6) is a human-verification milestone, not a placeholder. The single "no change expected" step (Task 2 Step 2 / Task 5 Step 2–3) is deliberate and explained: Step 1's builder is already general over the modules dict; the tests lock that in and localize a regression if a future refactor breaks it.

**Type consistency:** `_selfplay_cfg()` (Task 2) is reused by Task 5; both reference `multiagent.modules.{main_red,main_blue}.kind`, `policies_to_train`, `mapping` — identical names to `conf/multiagent/red_blue_selfplay.yaml` (Task 2 Step 3) and the builder's `cfg.multiagent` reads. Reward term names (`HoopApproachShaping`, `ScoreEvent`, `HoopAnchor`, `TakeDown`, `CrashEvent`) and their fields (`scale`/`agent`, `magnitude`/`scorer`/`zero_sum_opponent`, `agents`, `aggressor`/`victim`/`aggressor_reward`/`victim_penalty`, `agent_to_crash_flags`) match `envs/quidditch/rewards/terms.py` as merged in Step 1. Metric keys (`red_score_rate`, `red_min_dist_to_hoop`, `blue_prevention_rate`, `blue_min_dist_to_red`) are consistent across `rllib/metrics.py`, its tests, and Task 6's watch list. `infos["blue_0"]["dist_b2r"]` (Task 3) matches the `dist_b2r` local already computed in `team_env.step`.

**Known risk:** if RLlib's default multi-agent setup does not auto-infer a separate `observation_space`/`action_space` for `main_blue` from the env's `blue_0` spaces, Task 5's `build_algo` will raise at construction. Mitigation: the env adapter already exposes per-agent `observation_spaces`/`action_spaces` (Step 1 `rllib_env.py`); if inference still fails, pass explicit spaces into each `RLModuleSpec` in `config_builder` — a localized builder change the failing Task 5 test will pinpoint.
