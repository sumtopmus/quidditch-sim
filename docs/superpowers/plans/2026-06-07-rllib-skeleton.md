# RLlib Walking Skeleton Implementation Plan (Migration Step 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up RLlib's new API stack + Ray Tune training a single PPO policy (`main_red`) to score through the hoop on `QuidditchTeamEnv`, with a frozen hovering `main_blue`, proving every piece of env/obs/reward/PID/Ray/macOS plumbing before any league machinery exists.

**Architecture:** A thin `MultiAgentEnv` adapter wraps the existing PettingZoo `QuidditchTeamEnv` (adding the RLlib-required `"__all__"` done keys) and is registered with Tune. A custom `ScriptedRLModule` wraps the existing `Opponent` protocol so scripted/frozen agents plug into RLlib as non-learning policies. A Hydra→`PPOConfig` builder maps the existing `conf/` tree onto the new-stack `AlgorithmConfig`. Ray Tune owns the loop with the native W&B logger. Everything trains on CPU learners + CPU env-runners.

**Tech Stack:** Ray RLlib (new API stack: RLModule + Learner + ConnectorV2), Ray Tune, PyTorch, Hydra, Weights & Biases, MuJoCo, Python 3.13 (uv venv).

**Scope note:** This is **Step 1 of 6** from the spec (`docs/superpowers/specs/2026-06-07-rllib-league-migration-design.md`). It deliberately stops before multi-policy co-adaptation. Steps 2–6 (two-policy self-play → snapshot population → PFSP → curriculum/promotion → SB3 deletion) each get their own plan, written **after** this one lands and resolves its open questions (frame-stack location OQ-2, canary determinism OQ-3, Ray-on-macOS reality). SB3 is **not** removed in this plan — the old training path keeps working in parallel until Step 6.

---

## File Structure

**New files:**
- `envs/quidditch/rllib_env.py` — `make_team_env(env_config)` + `QuidditchMultiAgentEnv` (RLlib `MultiAgentEnv` adapter). One responsibility: present `QuidditchTeamEnv` to RLlib.
- `envs/quidditch/rllib_modules.py` — `ScriptedRLModule` (non-learning `TorchRLModule` wrapping an `Opponent`). One responsibility: scripted/frozen agents as RLlib modules.
- `rllib/__init__.py`, `rllib/config_builder.py` — `build_ppo_config(cfg) -> PPOConfig`. One responsibility: Hydra `DictConfig` → new-stack `AlgorithmConfig`.
- `rllib/runtime.py` — `ray_init_for_project()` (the macOS env-var/runtime_env plumbing in one place).
- `scripts/train_rllib.py` — Hydra entrypoint that builds the config and runs `tune.Tuner`.
- `conf/algo/ppo_rllib.yaml`, `conf/multiagent/red_solo.yaml`, `conf/tune/default.yaml`, `conf/obs/duel_v1_body_n1.yaml`, `conf/experiment/rllib_red_skeleton.yaml` — new Hydra config groups for the RLlib path.
- `tests/rllib/test_rllib_env_adapter.py`, `tests/rllib/test_scripted_module.py`, `tests/rllib/test_config_builder.py`, `tests/rllib/test_smoke_train.py` — tests mirroring the source layout.

**Modified files:**
- `pyproject.toml` — add `ray[rllib]`, `torch`.
- `Makefile` — add a `train-rllib` target.

---

## Task 1: Dependency + Ray plumbing gate (the blocker check)

This task answers one question before any project code is touched: **does Ray RLlib's new API stack install and run a PPO iteration on this exact macOS / Python 3.13 machine?** If it cannot, stop and escalate (fallback: a dedicated Python 3.12 venv for the RLlib path) — do not proceed.

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Create: `rllib/__init__.py` (empty)
- Create: `rllib/runtime.py`
- Create: `tests/rllib/__init__.py` (empty)
- Test: `tests/rllib/test_smoke_train.py` (CartPole portion only in this task)

- [ ] **Step 1: Add Ray + torch to dependencies**

Edit `pyproject.toml`, add to the `[project] dependencies` list:

```toml
    "ray[rllib]>=2.49",
    "torch>=2.2",
```

- [ ] **Step 2: Sync and verify wheels resolve on this machine**

Run: `uv sync`
Expected: completes without a build-from-source for Ray (a prebuilt `cp313` macOS wheel is pulled). If it fails to find a `cp313` wheel, STOP — record the error and escalate the Python-version fallback before continuing.

- [ ] **Step 3: Verify the import + a trivial build**

Run:
```bash
uv run python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; from ray.rllib.core.rl_module.rl_module import RLModuleSpec; from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec; from ray.rllib.env.multi_agent_env import MultiAgentEnv; print(ray.__version__)"
```
Expected: prints a version `>= 2.49` with no ImportError.

- [ ] **Step 4: Write the project Ray-init helper**

Create `rllib/runtime.py`:

```python
"""Centralized Ray initialization for the project.

The macOS libomp double-init guard (KMP_DUPLICATE_LIB_OK) must reach every
Ray worker process — env-runners and learners run in separate processes that
do NOT inherit a module-level os.environ set in the driver. We propagate it
explicitly via runtime_env.
"""
from __future__ import annotations

import os

import ray

# Driver-side guard (mirrors tests/conftest.py and scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WORKER_ENV_VARS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    # MuJoCo must render off-screen inside headless workers; keep GL out of
    # env-runners entirely (rendering happens in a dedicated eval worker).
    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
}


def ray_init_for_project(**kwargs) -> None:
    """Idempotent ray.init with the project's worker env-var propagation."""
    if ray.is_initialized():
        return
    runtime_env = {"env_vars": dict(_WORKER_ENV_VARS)}
    ray.init(runtime_env=runtime_env, ignore_reinit_error=True, **kwargs)
```

- [ ] **Step 5: Write the CartPole smoke test (Ray plumbing only, no MuJoCo)**

Create `tests/rllib/test_smoke_train.py`:

```python
"""End-to-end smoke tests for the RLlib path. Marked slow."""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_cartpole_one_iteration():
    """Ray + RLlib new stack runs one PPO iteration on this machine.

    This is the install/plumbing gate — it touches no project code. If this
    fails, the Ray/Python-3.13/macOS stack is the problem, not our adapters.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(num_env_runners=1)
        .training(train_batch_size_per_learner=256, minibatch_size=64, num_epochs=1)
    )
    algo = config.build_algo()
    try:
        result = algo.train()
        assert "env_runners" in result
    finally:
        algo.stop()
```

- [ ] **Step 6: Run the smoke test**

Run: `uv run python -m pytest tests/rllib/test_smoke_train.py::test_cartpole_one_iteration -v`
Expected: PASS. (First run downloads nothing else; Ray starts a local cluster.) If it fails on a Ray/macOS issue, STOP and escalate.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock rllib/ tests/rllib/__init__.py tests/rllib/test_smoke_train.py
git commit -S -m "feat(rllib): add ray[rllib] dep + macOS-safe ray.init + CartPole plumbing gate"
```

---

## Task 2: MultiAgentEnv adapter

Wrap `QuidditchTeamEnv` so RLlib can consume it. The only behavioral change is adding the RLlib-required `"__all__"` keys to the done dicts; obs/action/reward dicts pass through unchanged.

**Files:**
- Create: `envs/quidditch/rllib_env.py`
- Test: `tests/rllib/test_rllib_env_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rllib/test_rllib_env_adapter.py`:

```python
"""Contract tests for the RLlib MultiAgentEnv adapter."""
from __future__ import annotations

import numpy as np

from envs.quidditch.rllib_env import make_team_env


def _env():
    return make_team_env({"learner_id": "red_0", "obs_blocks": [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY_MIXED",
    ]})


def test_spaces_are_per_agent_and_keyed():
    env = make_env = _env()
    assert set(env.possible_agents) == {"red_0", "blue_0"}
    assert set(env.observation_spaces.keys()) == {"red_0", "blue_0"}
    assert set(env.action_spaces.keys()) == {"red_0", "blue_0"}
    assert env.action_spaces["red_0"].shape == (4,)


def test_reset_returns_obs_and_info_dicts():
    env = _env()
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"red_0", "blue_0"}
    assert obs["red_0"].dtype == np.float32
    assert obs["red_0"].shape == env.observation_spaces["red_0"].shape


def test_step_adds_all_done_keys():
    env = _env()
    env.reset(seed=0)
    actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    assert "__all__" in term
    assert "__all__" in trunc
    assert set(rew.keys()) == {"red_0", "blue_0"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_rllib_env_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: envs.quidditch.rllib_env`.

- [ ] **Step 3: Write the adapter**

Create `envs/quidditch/rllib_env.py`:

```python
"""RLlib MultiAgentEnv adapter for QuidditchTeamEnv.

QuidditchTeamEnv is already a PettingZoo ParallelEnv with per-agent obs/action
spaces and dict-keyed reset/step. RLlib's MultiAgentEnv contract is nearly
identical, with one addition: the terminateds/truncateds dicts must carry an
"__all__" key signalling whole-episode termination. This adapter adds it and
otherwise passes everything through.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_spec_from_block_names


class QuidditchMultiAgentEnv(MultiAgentEnv):
    """Thin RLlib wrapper around QuidditchTeamEnv."""

    def __init__(self, inner: QuidditchTeamEnv) -> None:
        super().__init__()
        self._inner = inner
        self.possible_agents = list(inner.possible_agents)
        self.agents = list(inner.agents)
        self.observation_spaces: dict[str, spaces.Box] = dict(inner.observation_spaces)
        self.action_spaces: dict[str, spaces.Box] = dict(inner.action_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self._inner.reset(seed=seed, options=options)
        self.agents = list(self._inner.agents)
        return obs, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        obs, rew, term, trunc, infos = self._inner.step(action_dict)
        term = dict(term)
        trunc = dict(trunc)
        term["__all__"] = all(term.get(a, False) for a in self.possible_agents)
        trunc["__all__"] = all(trunc.get(a, False) for a in self.possible_agents)
        self.agents = [a for a in self._inner.agents]
        return obs, rew, term, trunc, infos


def make_team_env(env_config: dict[str, Any]) -> QuidditchMultiAgentEnv:
    """Env creator for tune.register_env.

    Reads EVERYTHING from env_config — no closure capture survives Ray's
    serialization boundary, so the creator must be self-contained.

    Expected keys:
      learner_id:  str           (default "red_0")
      obs_blocks:  list[str]     (the learner's ObsSpec block names)
      team_cfg:    dict | None   (overrides for TeamConfig fields)
      reward_stack: RewardStack | None
    """
    learner_id = env_config.get("learner_id", "red_0")
    obs_blocks = list(env_config["obs_blocks"])
    team_cfg_overrides = dict(env_config.get("team_cfg", {}) or {})
    reward_stack = env_config.get("reward_stack")

    learner_spec = build_spec_from_block_names(obs_blocks)
    cfg = TeamConfig(**team_cfg_overrides)
    inner = QuidditchTeamEnv(
        cfg=cfg,
        reward_stack=reward_stack,
        learner_id=learner_id,
        learner_spec=learner_spec,
    )
    return QuidditchMultiAgentEnv(inner)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_rllib_env_adapter.py -v`
Expected: PASS (3 tests). Run from a regular Terminal window if MuJoCo render init complains in a headless shell.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/rllib_env.py tests/rllib/test_rllib_env_adapter.py
git commit -S -m "feat(rllib): MultiAgentEnv adapter for QuidditchTeamEnv with __all__ done keys"
```

---

## Task 3: ScriptedRLModule (non-learning policy)

Wrap the existing `Opponent` protocol as a new-stack `TorchRLModule` that returns actions directly (bypassing sampling). The skeleton uses `ZeroOpponent` (hover) for `main_blue`; the same class wraps beeline/frozen opponents in later plans.

**Files:**
- Create: `envs/quidditch/rllib_modules.py`
- Test: `tests/rllib/test_scripted_module.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rllib/test_scripted_module.py`:

```python
"""Tests for the non-learning ScriptedRLModule."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.quidditch.rllib_modules import ScriptedRLModule


def _build(opponent_spec="zero", obs_dim=22):
    spec = RLModuleSpec(
        module_class=ScriptedRLModule,
        observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=spaces.Box(-1.0, 1.0, (4,), np.float32),
        model_config={"opponent_spec": opponent_spec},
    )
    return spec.build()


def test_zero_opponent_returns_zero_actions_for_batch():
    module = _build("zero")
    batch = {Columns.OBS: torch.zeros((5, 22), dtype=torch.float32)}
    out = module._forward_inference(batch)
    actions = out[Columns.ACTIONS].numpy()
    assert actions.shape == (5, 4)
    assert np.allclose(actions, 0.0)


def test_exploration_matches_inference():
    module = _build("zero")
    batch = {Columns.OBS: torch.zeros((3, 22), dtype=torch.float32)}
    a_inf = module._forward_inference(batch)[Columns.ACTIONS]
    a_exp = module._forward_exploration(batch)[Columns.ACTIONS]
    assert torch.allclose(a_inf, a_exp)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_scripted_module.py -v`
Expected: FAIL with `ModuleNotFoundError: envs.quidditch.rllib_modules`.

- [ ] **Step 3: Write the module**

Create `envs/quidditch/rllib_modules.py`:

```python
"""Non-learning RLModule that defers to a scripted Opponent.

Wraps the existing Opponent protocol (envs.quidditch.opponents) so scripted
and frozen policies plug into RLlib's multi-agent stack as modules that are
NEVER in policies_to_train. Returns actions directly via Columns.ACTIONS,
bypassing the action-distribution sampling step.
"""
from __future__ import annotations

import numpy as np
import torch

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

from envs.quidditch.opponents import from_spec


class ScriptedRLModule(TorchRLModule):
    """A frozen module driven by a scripted/frozen Opponent.

    model_config keys:
        opponent_spec: str  — passed to envs.quidditch.opponents.from_spec
                              (e.g. "zero", "beeline_blue", "frozen:path").
    """

    def setup(self) -> None:
        spec = self.model_config["opponent_spec"]
        self._opponent = from_spec(spec)
        self._opponent.reset()

    def _act(self, batch: dict) -> dict:
        obs = batch[Columns.OBS].detach().cpu().numpy()
        actions = np.stack([self._opponent.act(row) for row in obs]).astype(np.float32)
        return {Columns.ACTIONS: torch.from_numpy(actions)}

    def _forward_inference(self, batch, **kw):
        return self._act(batch)

    def _forward_exploration(self, batch, **kw):
        return self._act(batch)

    def _forward_train(self, batch, **kw):
        raise RuntimeError("ScriptedRLModule is non-learning; do not add it to policies_to_train.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_scripted_module.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/rllib_modules.py tests/rllib/test_scripted_module.py
git commit -S -m "feat(rllib): ScriptedRLModule wrapping Opponent protocol as a frozen module"
```

---

## Task 4: Hydra → PPOConfig builder + config groups

Map the existing `conf/` tree onto a new-stack `PPOConfig` with `main_red` trainable and `main_blue` a frozen `ScriptedRLModule`. Add the RLlib-specific config groups.

**Files:**
- Create: `rllib/config_builder.py`
- Create: `conf/algo/ppo_rllib.yaml`, `conf/multiagent/red_solo.yaml`, `conf/obs/duel_v1_body_n1.yaml`, `conf/tune/default.yaml`, `conf/experiment/rllib_red_skeleton.yaml`
- Test: `tests/rllib/test_config_builder.py`

- [ ] **Step 1: Create the new config-group YAMLs**

Create `conf/algo/ppo_rllib.yaml`:

```yaml
# New-stack PPO hyperparameters. Names map to RLlib's AlgorithmConfig.training().
lr: 5e-5
gamma: 0.99
lambda_: 0.95           # RLlib name for gae_lambda
clip_param: 0.2         # RLlib name for clip_range
entropy_coeff: 0.01     # RLlib name for ent_coef
num_epochs: 6           # SB3 n_epochs
minibatch_size: 512     # SB3 batch_size
train_batch_size_per_learner: 8192   # ~ SB3 n_steps(1024) * n_envs(8)
num_env_runners: 4
total_timesteps: 20_000_000
```

Create `conf/obs/duel_v1_body_n1.yaml` (single-stack skeleton obs — sidesteps OQ-2 frame-stacking):

```yaml
name: DUEL_V1_BODY
n_stack: 1
blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - UNIT_TO_GOAL
  - SIGNED_DIST_NORM
  - OPP_POS_REL
  - OPP_VEL_REL_BODY_MIXED
```

Create `conf/multiagent/red_solo.yaml`:

```yaml
# Skeleton topology: main_red learns; blue_0 is a frozen hovering script.
learner_id: red_0
policies_to_train: [main_red]
mapping:
  red_0: main_red
  blue_0: main_blue
modules:
  main_red:
    kind: learned
  main_blue:
    kind: scripted
    opponent_spec: zero
```

Create `conf/tune/default.yaml`:

```yaml
checkpoint_frequency: 10     # iterations
checkpoint_at_end: true
wandb:
  project: drone-quidditch
  enabled: true
```

Create `conf/experiment/rllib_red_skeleton.yaml`:

```yaml
# @package _global_
run_name: rllib_red_skeleton

defaults:
  - override /obs: duel_v1_body_n1
  - override /reward: team_v4_cone

algo:
  total_timesteps: 2_000_000
seed: 42
```

- [ ] **Step 2: Write the failing test**

Create `tests/rllib/test_config_builder.py`:

```python
"""Tests that the Hydra→PPOConfig builder produces a valid new-stack config."""
from __future__ import annotations

from omegaconf import OmegaConf

from rllib.config_builder import build_ppo_config


def _cfg():
    return OmegaConf.create({
        "seed": 7,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY_MIXED",
        ]},
        "algo": {
            "lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
            "entropy_coeff": 0.01, "num_epochs": 6, "minibatch_size": 512,
            "train_batch_size_per_learner": 8192, "num_env_runners": 0,
            "total_timesteps": 1000,
        },
        "multiagent": {
            "learner_id": "red_0",
            "policies_to_train": ["main_red"],
            "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
            "modules": {
                "main_red": {"kind": "learned"},
                "main_blue": {"kind": "scripted", "opponent_spec": "zero"},
            },
        },
        "reward": None,
    })


def test_builder_returns_buildable_config():
    config = build_ppo_config(_cfg())
    assert set(config.policies) == {"main_red", "main_blue"}
    assert config.policies_to_train == ["main_red"]
    # mapping routes agents to the right module
    assert config.policy_mapping_fn("red_0", None) == "main_red"
    assert config.policy_mapping_fn("blue_0", None) == "main_blue"


def test_built_algo_constructs():
    config = build_ppo_config(_cfg())
    algo = config.build_algo()
    try:
        mod = algo.get_module("main_blue")
        assert mod is not None
    finally:
        algo.stop()
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: rllib.config_builder`.

- [ ] **Step 4: Write the builder**

Create `rllib/config_builder.py`:

```python
"""Hydra DictConfig -> new-stack PPOConfig.

Maps the conf/ tree (algo, obs, multiagent, reward) onto RLlib's
AlgorithmConfig builder. Registers the env under a stable name. main_red is a
learned PPO RLModule; main_blue (and future frozen members) are
ScriptedRLModules excluded from policies_to_train.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env

from envs.quidditch.rllib_env import make_team_env
from envs.quidditch.rllib_modules import ScriptedRLModule

_ENV_NAME = "quidditch_team"


def _ensure_env_registered() -> None:
    register_env(_ENV_NAME, lambda env_config: make_team_env(env_config))


def build_ppo_config(cfg: DictConfig) -> PPOConfig:
    _ensure_env_registered()
    ma = cfg.multiagent
    obs_blocks = list(cfg.obs.blocks)
    reward_stack = cfg.get("reward_stack")  # injected by the entrypoint (built dataclass)

    # Per-policy module specs.
    module_specs: dict[str, RLModuleSpec] = {}
    for name, spec in ma.modules.items():
        if spec.kind == "learned":
            module_specs[name] = RLModuleSpec()  # default PPO torch module
        elif spec.kind == "scripted":
            module_specs[name] = RLModuleSpec(
                module_class=ScriptedRLModule,
                model_config={"opponent_spec": spec.opponent_spec},
            )
        else:
            raise ValueError(f"unknown module kind: {spec.kind!r}")

    mapping = OmegaConf.to_container(ma.mapping, resolve=True)

    def policy_mapping_fn(agent_id, episode, **kw):
        return mapping[agent_id]

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            _ENV_NAME,
            env_config={
                "learner_id": ma.learner_id,
                "obs_blocks": obs_blocks,
                "team_cfg": {},
                "reward_stack": reward_stack,
            },
        )
        .framework("torch")
        .env_runners(num_env_runners=int(cfg.algo.num_env_runners))
        .multi_agent(
            policies=set(ma.modules.keys()),
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(ma.policies_to_train),
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .training(
            lr=float(cfg.algo.lr),
            gamma=float(cfg.algo.gamma),
            lambda_=float(cfg.algo.lambda_),
            clip_param=float(cfg.algo.clip_param),
            entropy_coeff=float(cfg.algo.entropy_coeff),
            num_epochs=int(cfg.algo.num_epochs),
            minibatch_size=int(cfg.algo.minibatch_size),
            train_batch_size_per_learner=int(cfg.algo.train_batch_size_per_learner),
        )
        .debugging(seed=int(cfg.seed))
    )
    return config
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run python -m pytest tests/rllib/test_config_builder.py -v`
Expected: PASS (2 tests). `test_built_algo_constructs` builds a real algo with `num_env_runners=0` (local sampling) — slowish but no subprocess.

- [ ] **Step 6: Commit**

```bash
git add rllib/config_builder.py conf/algo/ppo_rllib.yaml conf/multiagent/red_solo.yaml conf/obs/duel_v1_body_n1.yaml conf/tune/default.yaml conf/experiment/rllib_red_skeleton.yaml tests/rllib/test_config_builder.py
git commit -S -m "feat(rllib): Hydra->PPOConfig builder + RLlib config groups (algo/multiagent/tune)"
```

---

## Task 5: Tune entrypoint + W&B + smoke run

Wire the Hydra entrypoint that builds the config, injects the reward stack, and runs `tune.Tuner` with the native W&B logger. Prove a short run completes, checkpoints, and restores.

**Files:**
- Create: `scripts/train_rllib.py`
- Modify: `Makefile` (add `train-rllib`)
- Modify: `conf/config.yaml` (add `algo`, `multiagent`, `tune` to defaults — see Step 1)
- Test: `tests/rllib/test_smoke_train.py` (add the team-env portion)

- [ ] **Step 1: Add RLlib groups to the top-level defaults**

Edit `conf/config.yaml` `defaults:` list, append (after `wandb: default`):

```yaml
  - algo: ppo_rllib
  - multiagent: red_solo
  - tune: default
```

These are additive — the SB3 path ignores them, so existing experiments keep composing.

- [ ] **Step 2: Write the entrypoint**

Create `scripts/train_rllib.py`:

```python
"""RLlib + Tune training entrypoint (new API stack).

Parallel to scripts/train.py (SB3) during the migration. Builds a PPOConfig
from Hydra, injects the instantiated reward stack into the env_config, and
hands the loop to Tune with the native W&B logger.
"""
from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import RunConfig, CheckpointConfig

from config_schema import register_configs
from rllib.config_builder import build_ppo_config
from rllib.runtime import ray_init_for_project

register_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ray_init_for_project()

    reward_stack = instantiate(cfg.reward, _convert_="all") if cfg.get("reward") else None
    # Stash the built stack so the builder can put it into env_config.
    cfg_with_reward = cfg.copy()
    cfg_with_reward.reward_stack = reward_stack  # type: ignore[attr-defined]

    ppo_config = build_ppo_config(cfg_with_reward)

    callbacks = []
    if cfg.tune.wandb.enabled:
        callbacks.append(WandbLoggerCallback(project=cfg.tune.wandb.project))

    tuner = tune.Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name=cfg.run_name,
            stop={"num_env_steps_sampled_lifetime": int(cfg.algo.total_timesteps)},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=int(cfg.tune.checkpoint_frequency),
                checkpoint_at_end=bool(cfg.tune.checkpoint_at_end),
            ),
            callbacks=callbacks,
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add the Makefile target**

Edit `Makefile`, after the `train` target:

```makefile
train-rllib: ## 🚀 Launch an RLlib+Tune run  EXP=<name> [OVERRIDES="key=val"]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required"; exit 1; }
	@$(PYTHON) -m scripts.train_rllib +experiment=$(EXP) $(OVERRIDES)
```

- [ ] **Step 4: Add the team-env smoke test (checkpoint + restore)**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_team_skeleton_trains_and_restores(tmp_path):
    """main_red trains on the team env vs frozen hover blue; checkpoint round-trips."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY_MIXED",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0", "policies_to_train": ["main_red"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "scripted", "opponent_spec": "zero"}}},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()
        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
        algo.restore_from_path(ckpt)
    finally:
        algo.stop()
```

- [ ] **Step 5: Run the team smoke test**

Run: `uv run python -m pytest tests/rllib/test_smoke_train.py::test_team_skeleton_trains_and_restores -v`
Expected: PASS. Run from a regular Terminal window (MuJoCo render init). This is the first time MuJoCo runs under RLlib's sampling loop — if it fails on a libomp/`KMP` or `MUJOCO_GL` error, the `runtime_env` propagation in Task 1 Step 4 is the place to fix.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_rllib.py Makefile conf/config.yaml tests/rllib/test_smoke_train.py
git commit -S -m "feat(rllib): Tune entrypoint + W&B logger + team-env smoke/restore test"
```

---

## Task 6: "Red learns to score" validation milestone

This is a **behavioral verification gate**, not an auto-committed code change. Per project convention (verify behavior before committing; when verification needs the user/GUI, wait for their confirmation), the longer run and its judgement happen with the user in the loop.

**Files:** none created. Produces a run under `runs/rllib_red_skeleton/...` and a W&B run.

- [ ] **Step 1: Launch a real (but short) skeleton run**

Run: `make train-rllib EXP=rllib_red_skeleton OVERRIDES="algo.total_timesteps=2_000_000"`
Expected: training proceeds; W&B logs `env_runners/...` return and episode-length metrics; a checkpoint dir appears under `runs/rllib_red_skeleton/`.

- [ ] **Step 2: Confirm the learning signal**

Watch the W&B run (or stdout): the episode return for `main_red` should trend up and the scoring rate (hoop crossings) should rise above the random-init baseline within ~1–2M steps — Red is learning to fly through the hoop vs a hovering Blue, the known-good R4t condition.

- [ ] **Step 3: User verification checkpoint**

STOP and report the W&B curves + scoring trend to the user. Do not declare the skeleton "working" until the user confirms Red is visibly learning to score (curves, or a rendered rollout). This is the milestone that proves the env/obs/reward/PID/Ray plumbing is correct end-to-end under RLlib.

- [ ] **Step 4: (After user confirmation) Record the result**

Once confirmed, note the run id + scoring trend in `brain/changelog.md` (and update `brain/index.md` Current State to reflect that the RLlib skeleton is green). This unblocks writing the Step 2 plan (two-policy self-play).

---

## Self-Review

**Spec coverage (Part A new components):** Env adapter ✓ (Task 2). Config bridge ✓ (Task 4). RLModule spec ✓ (Tasks 3–4). Tune harness ✓ (Task 5). W&B ✓ (Task 5). macOS/Ray gotchas: `KMP`/`MUJOCO_GL` propagation ✓ (Task 1 `runtime.py`), env-creator-reads-everything ✓ (Task 2 `make_team_env` docstring + impl), `==`-not-`is` — N/A in skeleton (no singleton identity checks introduced; flagged for the reward-anneal plan). League callback, checkpoint/lineage adapter, eval port, PFSP, populations, reward anneal, SB3 deletion: **intentionally out of scope** — they are Steps 2–6, each its own plan.

**Placeholder scan:** No TBD/TODO/"add error handling" — every code step shows complete code. The one judgement gate (Task 6) is explicitly a human-verification milestone, not a placeholder.

**Type consistency:** `make_team_env(env_config: dict)` keys (`learner_id`, `obs_blocks`, `team_cfg`, `reward_stack`) are identical in `rllib_env.py`, `config_builder.py`'s `.environment(env_config=...)`, and the tests. `ScriptedRLModule` reads `model_config["opponent_spec"]` consistently in the module, its test, and the builder. `policies_to_train`, `mapping`, `modules` names match across `conf/multiagent/red_solo.yaml`, the builder, and `test_config_builder.py`. RLlib method names (`build_algo`, `restore_from_path`, `train_batch_size_per_learner`, `lambda_`, `clip_param`, `entropy_coeff`, `num_epochs`, `minibatch_size`) are the new-stack spellings confirmed against current Ray docs.

**Known version-sensitivity:** RLlib's API stack moves fast. If a method name drifts on the installed Ray version, the failing test in Tasks 4–5 localizes it immediately. Pin the resolved Ray version in `uv.lock` (Task 1) so the plan is reproducible.
