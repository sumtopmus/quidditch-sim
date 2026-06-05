# Oracle-Privileged Critic + World-Frame Observables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in CTDE (centralized-training / decentralized-execution) training path where the SB3 learner emits a `Dict({"actor","critic"})` observation — a world-frame, normalized, pruned actor view and a privileged value-network view with oracle/future + ground-truth signals — driven by a custom `AsymmetricActorCriticPolicy`.

**Architecture:** A new policy class gives the value network a wider observation than the actor (gradient-isolated). The team env gains a `ctde_mode` that emits the dual obs; only the learner's space becomes a `Dict` (the scripted/frozen opponent stays flat). Everything is gated so the existing flat path, both canaries, and all promoted-model loading stay byte-identical. CTDE runs are `init=scratch` only.

**Tech Stack:** Python 3.13 (uv), Stable-Baselines3 2.8.0 (gymnasium), PettingZoo, MuJoCo, Hydra, pytest.

---

## Design Reference (read before starting)

Spec: `docs/superpowers/specs/2026-06-04-oracle-critic-obs-design.md`.

Codebase facts this plan relies on:

- **`Quadrotor.state()`** (`core/quadrotor.py:181`) returns a 4-row array: `[0]=ang_vel (body)`, `[1]=ang_pos (euler, ground)`, `[2]=lin_vel (body)`, `[3]=lin_pos (world)`. `Quadrotor.step_period` is `dt`.
- **World-frame linear velocity** is read as `world.data.qvel[dofadr:dofadr+3]`; `team_env` already caches `self._red_dofadr` / `self._blue_dofadr` in `_build_world` and uses them in `step()`.
- **Hoop geometry** (`envs/quidditch/constants.py`): `HOOP_CENTER=[2,0,2]`, `HOOP_OUTWARD_NORMAL=[1,0,0]`, `ARENA_RADIUS=3.0`, `ARENA_WALL_HEIGHT=4.5`, `CRASH_VEL_THR=1.0`, `TAG_RADIUS=0.3`. Red scores crossing the plane from `-x` to `+x`, so `signed_dist = (pos-HOOP_CENTER)·normal` is negative while Red approaches.
- **`team_env._build_agent_features(agent_id)`** (`team_env.py:608`) returns a `{block_name: float32-array}` dict; `obs_spec.pack(spec, values)` selects the subset a spec asks for. Adding new keys to this dict is harmless to existing flat specs (they don't request them).
- **`obs_spec.ObsBlock(name, dim, frame, notes)`** + **`ObsSpec(blocks)`**; `BLOCK_BY_NAME` auto-registers every module-level `ObsBlock`. `build_spec_from_block_names([...])` resolves Python-identifier names to a spec.
- **`OpponentControlledEnv`** (`opponents.py:247`) forwards `obs[learner_id]` to SB3 and drives the opponent from `obs[opponent_id]`. It computes `opp_action = self.opponent.act(self._last_opp_obs)` inside `step()`.
- **PPO is constructed at one site**, `scripts/train.py:_build_or_load_model` (line 152); `main()` has an **inline `eval_env_fn`** (line 336) that mirrors the factory and must be touched too (memory `feedback_thread_kwarg_through_all_sites`).
- **Frame stacking**: `TeamEnvFactory.build_train_env` wraps in `VecFrameStack`; `_train_common.build_callbacks` wraps the eval env in `VecFrameStack`; `opponents.FrameStackWrapper` stacks the single video env.

Conventions: `uv run pytest …` (or `make test-fast`). Commit types: `feat/test/refactor/fix/docs/chore`. Tests mirror source: `tests/<package>/<module>/test_<name>.py`. GPG signing is on by default.

**Stability anchors — DO NOT modify:** `conf/reward/{single_agent,team_v2}.yaml`, `conf/experiment/{canary_single,canary_team}.yaml`, `tests/integration/test_scoring_canary.py`, `tests/unit/test_team_env_canary.py` (and `tests/integration/test_team_env_canary.py` if present), `conf/trainer/ppo.yaml`.

---

## File Structure

**Create:**
- `core/policies/asymmetric.py` — `AsymmetricActorCriticPolicy` + keyed extractor + asymmetric MLP extractor.
- `envs/quidditch/dict_frame_stack.py` — `SelectiveDictFrameStack` vec-env wrapper.
- `conf/policy/mlp.yaml`, `conf/policy/asymmetric.yaml` — policy group.
- `conf/obs/ctde_v1.yaml` — dual block lists.
- `conf/experiment/blue_oracle_v1.yaml` — first CTDE experiment.
- Tests under `tests/core/policies/`, `tests/envs/quidditch/`, `tests/scripts/`.

**Modify:**
- `envs/quidditch/constants.py` — `ORACLE_HORIZON_S`, `ORACLE_TIME_CAP_S`, `TAKEDOWN_CONTACT_DIST`.
- `envs/quidditch/obs_spec.py` — 11 new blocks; `NORM_BY_BLOCK`; `pack_normalized`; `build_ctde_specs_from_yaml`.
- `envs/quidditch/team_env.py` — `ctde_mode`/`critic_spec`/`actor_spec` kwargs; actor feature additions; `_build_critic_features`; Dict obs; `inject_opp_next_action`; `_pending_opp_action`; `_opponent_act` seam; `_last_tag_during`.
- `envs/quidditch/opponents.py` — OCE CTDE path; `FrameStackWrapper` Dict support.
- `envs/quidditch/env_factories.py` — `ctde_mode`; dict-aware frame stacking.
- `scripts/_train_common.py` — `build_callbacks` dict-aware eval stacking.
- `scripts/train.py` — policy dispatch + CTDE obs path in `_build_or_load_model` AND inline `eval_env_fn`/video.
- `config_schema.py` — `PolicyConfig`; `ObsConfig` extension; register.
- `conf/config.yaml` — add `- policy: mlp` default.
- `core/obs_compat.py` — flat↔dict incompatibility.
- `core/policies/__init__.py` — export.

---

## PHASE 0 — Asymmetric policy (de-risk first)

> The custom SB3 policy (custom `_build_mlp_extractor` + checkpoint serialization) is the single highest-risk piece. Build and fully validate it against a **mock Dict env** before any quidditch code depends on it. If serialization proves intractable in SB3 2.8.0, the fallback is a single `CombinedExtractor` over the full Dict with the actor MLP masking the critic slice — surface this immediately.

### Task 0.1: Keyed extractor + asymmetric MLP extractor

**Files:**
- Create: `core/policies/asymmetric.py`
- Test: `tests/core/policies/test_asymmetric_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/policies/test_asymmetric_policy.py
import numpy as np
import torch
from gymnasium import spaces

from core.policies.asymmetric import _KeyedExtractor, _AsymmetricMlpExtractor


def _dict_space(actor_dim=6, critic_dim=4):
    return spaces.Dict({
        "actor":  spaces.Box(-np.inf, np.inf, (actor_dim,), np.float32),
        "critic": spaces.Box(-np.inf, np.inf, (critic_dim,), np.float32),
    })


def test_keyed_extractor_selects_and_concats_keys():
    sp = _dict_space(6, 4)
    pi_ext = _KeyedExtractor(sp, ("actor",))
    vf_ext = _KeyedExtractor(sp, ("actor", "critic"))
    assert pi_ext.features_dim == 6
    assert vf_ext.features_dim == 10
    obs = {
        "actor":  torch.zeros(2, 6),
        "critic": torch.ones(2, 4),
    }
    assert pi_ext(obs).shape == (2, 6)
    out = vf_ext(obs)
    assert out.shape == (2, 10)
    # actor slice zeros, critic slice ones (concat order = keys order).
    assert torch.allclose(out[:, :6], torch.zeros(2, 6))
    assert torch.allclose(out[:, 6:], torch.ones(2, 4))


def test_asymmetric_mlp_extractor_has_separate_input_dims():
    ext = _AsymmetricMlpExtractor(pi_dim=6, vf_dim=10, net_arch=[8, 8],
                                  activation_fn=torch.nn.Tanh, device="cpu")
    assert ext.latent_dim_pi == 8
    assert ext.latent_dim_vf == 8
    lat_pi = ext.forward_actor(torch.zeros(3, 6))
    lat_vf = ext.forward_critic(torch.zeros(3, 10))
    assert lat_pi.shape == (3, 8)
    assert lat_vf.shape == (3, 8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py -x -q`
Expected: FAIL — `ModuleNotFoundError: core.policies.asymmetric`.

- [ ] **Step 3: Write the extractors**

```python
# core/policies/asymmetric.py
"""AsymmetricActorCriticPolicy — privileged-critic (CTDE) PPO policy for SB3.

The actor (policy) network reads only obs["actor"]; the value network reads
the concatenation of the chosen critic keys (obs["actor"] + obs["critic"]).
Because the critic key never enters the policy branch, the action distribution
has zero gradient w.r.t. obs["critic"] — the privileged info lowers value
variance without biasing the policy gradient (Pinto et al. 2017).

See docs/superpowers/specs/2026-06-04-oracle-critic-obs-design.md §6.2.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _KeyedExtractor(BaseFeaturesExtractor):
    """Flatten + concat a fixed subset of a Dict obs's keys, in key order."""

    def __init__(self, observation_space: spaces.Dict, keys: tuple[str, ...]) -> None:
        dim = sum(int(np.prod(observation_space[k].shape)) for k in keys)
        super().__init__(observation_space, features_dim=dim)
        self._keys = tuple(keys)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([torch.flatten(obs[k], start_dim=1) for k in self._keys], dim=1)


def _make_trunk(in_dim: int, layers: list[int], activation_fn) -> tuple[nn.Sequential, int]:
    mods: list[nn.Module] = []
    last = in_dim
    for h in layers:
        mods.append(nn.Linear(last, h))
        mods.append(activation_fn())
        last = h
    return nn.Sequential(*mods), last


class _AsymmetricMlpExtractor(nn.Module):
    """Like SB3's MlpExtractor but with independent input dims for pi and vf."""

    def __init__(self, *, pi_dim: int, vf_dim: int, net_arch, activation_fn, device) -> None:
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]
        if isinstance(net_arch, dict):
            pi_layers = list(net_arch.get("pi", []))
            vf_layers = list(net_arch.get("vf", []))
        else:
            pi_layers = vf_layers = list(net_arch)
        self.policy_net, self.latent_dim_pi = _make_trunk(pi_dim, pi_layers, activation_fn)
        self.value_net,  self.latent_dim_vf = _make_trunk(vf_dim, vf_layers, activation_fn)
        self.policy_net.to(device)
        self.value_net.to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py -x -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add core/policies/asymmetric.py tests/core/policies/test_asymmetric_policy.py
git commit -m "feat(policy): keyed + asymmetric-dim MLP extractors for CTDE"
```

### Task 0.2: AsymmetricActorCriticPolicy + gradient isolation

**Files:**
- Modify: `core/policies/asymmetric.py`
- Modify: `core/policies/__init__.py`
- Test: `tests/core/policies/test_asymmetric_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/core/policies/test_asymmetric_policy.py
from core.policies.asymmetric import AsymmetricActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class _MockDictEnv(gym.Env):
    """Tiny Dict-obs env: actor=6-d, critic=4-d, action=2-d Box."""
    def __init__(self):
        super().__init__()
        self.observation_space = _dict_space(6, 4)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0

    def _obs(self):
        return {
            "actor":  self.observation_space["actor"].sample() * 0.0 + 0.1,
            "critic": self.observation_space["critic"].sample() * 0.0 + 0.2,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, self._t >= 16, False, {}


def _make_policy():
    env = DummyVecEnv([_MockDictEnv])
    lr = lambda _: 3e-4
    return AsymmetricActorCriticPolicy(
        env.observation_space, env.action_space, lr,
        net_arch=[8, 8],
    )


def test_policy_constructs_with_split_dims():
    pol = _make_policy()
    assert pol.pi_features_extractor.features_dim == 6
    assert pol.vf_features_extractor.features_dim == 10


def test_critic_obs_has_no_gradient_path_to_action():
    pol = _make_policy()
    obs = {
        "actor":  torch.full((1, 6), 0.1, requires_grad=False),
        "critic": torch.full((1, 4), 0.2, requires_grad=True),
    }
    dist = pol.get_distribution(obs)
    logp = dist.log_prob(torch.zeros(1, 2))
    logp.sum().backward()
    # The action distribution must not depend on the critic key.
    assert obs["critic"].grad is None or torch.allclose(
        obs["critic"].grad, torch.zeros_like(obs["critic"].grad))


def test_critic_obs_does_affect_value():
    pol = _make_policy()
    base = {"actor": torch.full((1, 6), 0.1), "critic": torch.full((1, 4), 0.2)}
    other = {"actor": torch.full((1, 6), 0.1), "critic": torch.full((1, 4), 5.0)}
    v0 = pol.predict_values(base)
    v1 = pol.predict_values(other)
    assert not torch.allclose(v0, v1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py -x -q -k policy or critic`
Expected: FAIL — `AsymmetricActorCriticPolicy` not defined.

- [ ] **Step 3: Implement the policy**

Append to `core/policies/asymmetric.py`:

```python
class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """PPO policy whose value net sees more of the Dict obs than the actor.

    actor_key:   the single Dict key the policy network reads.
    critic_keys: the Dict keys the value network reads (concatenated, in order).
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args,
                 actor_key: str = "actor",
                 critic_keys: tuple[str, ...] = ("actor", "critic"),
                 **kwargs):
        self._actor_key = actor_key
        self._critic_keys = tuple(critic_keys)
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Replace the base class's two identical CombinedExtractors with keyed
        # ones, then build an MLP extractor with per-head input dims.
        self.pi_features_extractor = _KeyedExtractor(
            self.observation_space, (self._actor_key,)).to(self.device)
        self.vf_features_extractor = _KeyedExtractor(
            self.observation_space, self._critic_keys).to(self.device)
        self.mlp_extractor = _AsymmetricMlpExtractor(
            pi_dim=self.pi_features_extractor.features_dim,
            vf_dim=self.vf_features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(actor_key=self._actor_key, critic_keys=self._critic_keys)
        return data
```

Update `core/policies/__init__.py`:

```python
"""Custom policy utilities (warm-start, asymmetric CTDE policy)."""
from core.policies.asymmetric import AsymmetricActorCriticPolicy

__all__ = ["AsymmetricActorCriticPolicy"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py -x -q`
Expected: PASS. *If construction fails inside SB3's `_build` because `self.features_dim` is consulted before `_build_mlp_extractor`, set `self.features_dim = self.observation_space["actor"].shape[0]` at the top of `_build_mlp_extractor` and re-run; if `get_distribution`/`predict_values` signatures differ in 2.8.0, inspect `stable_baselines3/common/policies.py` and adapt — this is the documented gating risk.*

- [ ] **Step 5: Commit**

```bash
git add core/policies/asymmetric.py core/policies/__init__.py tests/core/policies/test_asymmetric_policy.py
git commit -m "feat(policy): AsymmetricActorCriticPolicy with gradient-isolated critic"
```

### Task 0.3: PPO learn + checkpoint round-trip smoke

**Files:**
- Test: `tests/core/policies/test_asymmetric_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/core/policies/test_asymmetric_policy.py
from stable_baselines3 import PPO


def test_ppo_learns_and_round_trips(tmp_path):
    env = DummyVecEnv([_MockDictEnv])
    model = PPO(AsymmetricActorCriticPolicy, env,
                policy_kwargs=dict(net_arch=[8, 8]),
                n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=64)
    p = tmp_path / "m.zip"
    model.save(str(p))
    loaded = PPO.load(str(p), env=env)            # reconstructs the custom policy
    obs = env.reset()
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert a1.shape == a2.shape
    np.testing.assert_allclose(a1, a2, atol=1e-5)  # constructor params survived save/load
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py::test_ppo_learns_and_round_trips -x -q`
Expected: FAIL initially only if a serialization gap exists; otherwise it should pass once 0.2 lands. Run it to confirm the round-trip.

- [ ] **Step 3: (No new code if green.)** If `PPO.load` raises about missing `actor_key`/`critic_keys`, confirm `_get_constructor_parameters` is returning them (Task 0.2 Step 3). If load rebuilds with default keys, ensure the kwargs are not being dropped by SB3's `policy_kwargs` filtering — pass them via `policy_kwargs` in the experiment instead and document.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/policies/test_asymmetric_policy.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tests/core/policies/test_asymmetric_policy.py
git commit -m "test(policy): PPO learn + checkpoint round-trip on mock Dict env"
```

---

## PHASE 1 — Obs-spec dual plumbing (pure, no env behavior change)

### Task 1.1: New ObsBlock constants

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Test: `tests/envs/quidditch/test_ctde_obs_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/quidditch/test_ctde_obs_spec.py
from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import BLOCK_BY_NAME, build_spec_from_block_names

CTDE_ACTOR = ["ANG_VEL", "ANG_POS", "LIN_VEL_WORLD", "LIN_POS",
              "VEC_TO_HOOP", "OPP_POS_REL", "OPP_VEL_REL_WORLD",
              "CLOSING_RATE", "TIME_REMAINING"]
CTDE_CRITIC = ["OPP_NEXT_ACTION", "SELF_FUTURE_DISP", "OPP_FUTURE_REL",
               "SCORE_PRED", "TAKEDOWN_PRED", "RED_POS_ABS", "BLUE_POS_ABS",
               "TAG_STATE_ONEHOT", "TERMINAL_MARGINS"]


def test_new_blocks_registered():
    for name in CTDE_ACTOR + CTDE_CRITIC:
        assert name in BLOCK_BY_NAME, name


def test_actor_and_critic_dims():
    actor = build_spec_from_block_names(CTDE_ACTOR)
    critic = build_spec_from_block_names(CTDE_CRITIC)
    assert actor.dim == 23
    assert critic.dim == 28
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py::test_new_blocks_registered -x -q`
Expected: FAIL — `LIN_VEL_WORLD` not in `BLOCK_BY_NAME`.

- [ ] **Step 3: Add the blocks**

In `envs/quidditch/obs_spec.py`, **above** the `BLOCK_BY_NAME = {…}` line (so the auto-registry picks them up), add:

```python
LIN_VEL_WORLD = ObsBlock("lin_vel_world", dim=3, frame="world",
                         notes="world-frame linear velocity (qvel[dofadr:+3])")
TIME_REMAINING = ObsBlock("time_remaining", dim=1,
                          notes="(max_steps - step) / max_steps, in [0,1]")

# ── Critic-only (privileged) blocks — CTDE value head, world frame ───────────
OPP_NEXT_ACTION = ObsBlock("opp_next_action", dim=4,
                           notes="opponent's applied action this step (injected)")
SELF_FUTURE_DISP = ObsBlock("self_future_disp", dim=3, frame="world",
                            notes="k·v_self_world / arena")
OPP_FUTURE_REL = ObsBlock("opp_future_rel", dim=3, frame="world",
                          notes="((opp_pos + k·v_opp) - self_pos)/arena")
SCORE_PRED = ObsBlock("score_pred", dim=3,
                      notes="[time_to_plane/Tcap, lateral_miss/arena, approach_align]")
TAKEDOWN_PRED = ObsBlock("takedown_pred", dim=3,
                         notes="[time_to_cpa/Tcap, min_sep/arena, imminent_flag]")
RED_POS_ABS = ObsBlock("red_pos_abs", dim=3, frame="world", notes="red_pos/arena")
BLUE_POS_ABS = ObsBlock("blue_pos_abs", dim=3, frame="world", notes="blue_pos/arena")
TAG_STATE_ONEHOT = ObsBlock("tag_state_onehot", dim=2,
                            notes="[tag_during, tag_cooldown_active]")
TERMINAL_MARGINS = ObsBlock("terminal_margins", dim=4,
                            notes="[red_wall, blue_wall, red_floor, blue_floor] margins")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_ctde_obs_spec.py
git commit -m "feat(obs): add CTDE actor + critic ObsBlock constants"
```

### Task 1.2: NORM_BY_BLOCK + pack_normalized

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Test: `tests/envs/quidditch/test_ctde_obs_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/envs/quidditch/test_ctde_obs_spec.py
import numpy as np
from envs.quidditch.obs_spec import pack_normalized, NORM_BY_BLOCK, ARENA_NORM


def test_pack_normalized_scales_known_blocks_and_passes_through_unknown():
    spec = build_spec_from_block_names(["LIN_POS", "TAG_STATE_ONEHOT"])
    values = {"lin_pos": np.array([3.0, 0.0, 0.0], np.float32),
              "tag_state_onehot": np.array([1.0, 0.0], np.float32)}
    out = pack_normalized(spec, values, NORM_BY_BLOCK)
    # lin_pos divided by ARENA_NORM (=3) -> 1.0; tag_state passes through (no scale).
    np.testing.assert_allclose(out[:3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[3:], [1.0, 0.0], atol=1e-6)


def test_plain_pack_unchanged_for_flat_specs():
    # The flat path must stay byte-identical: pack() ignores NORM_BY_BLOCK.
    spec = build_spec_from_block_names(["LIN_POS"])
    out = obs_spec.pack(spec, {"lin_pos": np.array([3.0, 0.0, 0.0], np.float32)})
    np.testing.assert_allclose(out, [3.0, 0.0, 0.0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py -x -q -k normalized`
Expected: FAIL — `pack_normalized` not defined.

- [ ] **Step 3: Implement**

In `envs/quidditch/obs_spec.py`, after `pack(...)`:

```python
ARENA_NORM: float = 3.0       # ARENA_RADIUS; kept local to avoid a constants import cycle
_VEL_NORM: float = 3.0
_ANGVEL_NORM: float = 10.0

# Per-block divisors for the CTDE actor view.  Blocks absent from this map are
# passed through unchanged (scale 1.0) — including all critic blocks, which are
# normalized at source in team_env._build_critic_features.
NORM_BY_BLOCK: dict[str, float] = {
    "ang_vel":            _ANGVEL_NORM,
    "ang_pos":            float(np.pi),
    "lin_vel_world":      _VEL_NORM,
    "lin_pos":            ARENA_NORM,
    "vec_to_hoop_world":  ARENA_NORM,
    "opp_pos_rel_world":  ARENA_NORM,
    "opp_vel_rel_world":  _VEL_NORM,
    "closing_rate":       _VEL_NORM,
    "time_remaining":     1.0,
}


def pack_normalized(spec: ObsSpec, values: dict[str, ArrayLike],
                    norm_by_block: dict[str, float]) -> np.ndarray:
    """Like pack(), but divide each block by norm_by_block.get(name, 1.0).

    Used only on the CTDE Dict views; the flat path keeps calling pack() so it
    stays byte-identical."""
    arrays: list[np.ndarray] = []
    for block in spec.blocks:
        v = np.asarray(values[block.name], dtype=np.float32)
        if v.shape != (block.dim,):
            raise ValueError(
                f"pack_normalized: block {block.name!r} expects ({block.dim},), got {v.shape}")
        arrays.append(v / np.float32(norm_by_block.get(block.name, 1.0)))
    return np.concatenate(arrays, dtype=np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_ctde_obs_spec.py
git commit -m "feat(obs): NORM_BY_BLOCK + pack_normalized for CTDE actor view"
```

### Task 1.3: build_ctde_specs_from_yaml + conf/obs/ctde_v1.yaml

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Create: `conf/obs/ctde_v1.yaml`
- Test: `tests/envs/quidditch/test_ctde_obs_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/envs/quidditch/test_ctde_obs_spec.py
from envs.quidditch.obs_spec import build_ctde_specs_from_yaml


def test_build_ctde_specs_from_yaml():
    actor, critic = build_ctde_specs_from_yaml("ctde_v1")
    assert actor.dim == 23
    assert critic.dim == 28
    assert [b.name for b in actor.blocks][:2] == ["ang_vel", "ang_pos"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py -x -q -k from_yaml`
Expected: FAIL — `build_ctde_specs_from_yaml` not defined.

- [ ] **Step 3: Create the YAML + the loader**

`conf/obs/ctde_v1.yaml`:

```yaml
# CTDE dual-view obs.  Learner emits Dict({actor, critic}); opponent stays flat.
# Actor: world-frame, normalized, pruned (23-d). Critic: oracle/future +
# ground-truth privileged extras (28-d).  See the 2026-06-04 design spec.
name: CTDE_V1
obs_mode: dict
n_stack: 3
actor_blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_WORLD
  - LIN_POS
  - VEC_TO_HOOP
  - OPP_POS_REL
  - OPP_VEL_REL_WORLD
  - CLOSING_RATE
  - TIME_REMAINING
critic_blocks:
  - OPP_NEXT_ACTION
  - SELF_FUTURE_DISP
  - OPP_FUTURE_REL
  - SCORE_PRED
  - TAKEDOWN_PRED
  - RED_POS_ABS
  - BLUE_POS_ABS
  - TAG_STATE_ONEHOT
  - TERMINAL_MARGINS
```

In `obs_spec.py`, after `load_obs_yaml`:

```python
def build_ctde_specs_from_yaml(stem: str) -> tuple[ObsSpec, ObsSpec]:
    """Load conf/obs/<stem>.yaml and return (actor_spec, critic_spec).

    Requires `actor_blocks:` + `critic_blocks:` (the dual-view schema).
    Raises KeyError if either is missing."""
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo_root / "conf" / "obs" / f"{stem}.yaml").read_text())
    for key in ("actor_blocks", "critic_blocks"):
        if key not in cfg:
            raise KeyError(f"conf/obs/{stem}.yaml missing `{key}:` (CTDE dual-view schema)")
    return (build_spec_from_block_names(cfg["actor_blocks"]),
            build_spec_from_block_names(cfg["critic_blocks"]))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs_spec.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/obs_spec.py conf/obs/ctde_v1.yaml tests/envs/quidditch/test_ctde_obs_spec.py
git commit -m "feat(obs): build_ctde_specs_from_yaml + conf/obs/ctde_v1.yaml"
```

### Task 1.4: ObsConfig schema extension

**Files:**
- Modify: `config_schema.py`
- Test: `tests/test_config_schema.py` (or a new `tests/test_ctde_config.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ctde_config.py
from config_schema import ObsConfig


def test_obs_config_has_ctde_fields_with_back_compat_defaults():
    c = ObsConfig()
    assert c.obs_mode == "flat"        # default keeps the existing path
    assert c.actor_blocks == []
    assert c.critic_blocks == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ctde_config.py -x -q`
Expected: FAIL — `ObsConfig` has no `obs_mode`.

- [ ] **Step 3: Extend the dataclass**

In `config_schema.py`, replace the `ObsConfig` body with:

```python
@dataclass
class ObsConfig:
    """Names a canonical ObsSpec.

    flat mode: `blocks` carries the ordered ObsBlock identifiers (existing path).
    dict mode (CTDE): `actor_blocks` + `critic_blocks` carry the dual views.
    """
    name: str = "DUEL_V2_WORLD"
    n_stack: int = 3
    blocks: list[str] = field(default_factory=list)
    obs_mode: str = "flat"                       # "flat" | "dict"
    actor_blocks: list[str] = field(default_factory=list)
    critic_blocks: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ctde_config.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config_schema.py tests/test_ctde_config.py
git commit -m "feat(config): ObsConfig gains obs_mode + actor/critic block lists"
```

---

## PHASE 2 — Env feature computation (ctde-gated; flat path byte-identical)

### Task 2.1: Actor additions (lin_vel_world, time_remaining) + constants

**Files:**
- Modify: `envs/quidditch/constants.py`, `envs/quidditch/team_env.py`
- Test: `tests/envs/quidditch/test_ctde_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/quidditch/test_ctde_features.py
import numpy as np
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_spec_from_block_names


def _env():
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False),
                           learner_id="blue_0",
                           learner_spec=build_spec_from_block_names(["LIN_POS"]))
    env.reset(seed=0)
    return env


def test_feature_dict_has_world_vel_and_time_remaining():
    env = _env()
    feats = env._build_agent_features("blue_0")
    assert feats["lin_vel_world"].shape == (3,)
    assert feats["time_remaining"].shape == (1,)
    assert 0.0 <= float(feats["time_remaining"][0]) <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py::test_feature_dict_has_world_vel_and_time_remaining -x -q`
Expected: FAIL — `KeyError: 'lin_vel_world'`.

- [ ] **Step 3: Implement**

In `envs/quidditch/constants.py`, after `REWARD_LOOKAHEAD_S`:

```python
# ── Oracle critic (CTDE) ─────────────────────────────────────────────────────
ORACLE_HORIZON_S: float = 0.5      # kinematic lookahead for future-position blocks
ORACLE_TIME_CAP_S: float = 3.0     # cap + normalizer for predicted time-to-event
TAKEDOWN_CONTACT_DIST: float = 0.3 # m; predicted min-sep below this ⇒ imminent-crash
```

In `team_env.py` `_build_agent_features`, the `self_vel_world` is already computed (`data.qvel[self_dofadr:self_dofadr+3]`). Add two keys to the returned dict (insert before the closing `}` of the return):

```python
            "lin_vel_world":          self_vel_world.astype(np.float32),
            "time_remaining":         np.array(
                [(self._max_steps - self._step_count) / max(1, self._max_steps)],
                dtype=np.float32),
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py -x -q && uv run pytest tests/integration/test_team_env_canary.py -q`
Expected: PASS — and the team canary stays green (the flat spec doesn't request the new keys).

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/constants.py envs/quidditch/team_env.py tests/envs/quidditch/test_ctde_features.py
git commit -m "feat(env): lin_vel_world + time_remaining actor features + oracle constants"
```

### Task 2.2: Ground-truth critic features

**Files:**
- Modify: `envs/quidditch/team_env.py`
- Test: `tests/envs/quidditch/test_ctde_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/envs/quidditch/test_ctde_features.py
def test_ground_truth_critic_features():
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    assert crit["red_pos_abs"].shape == (3,)
    assert crit["blue_pos_abs"].shape == (3,)
    assert crit["tag_state_onehot"].shape == (2,)
    assert crit["terminal_margins"].shape == (4,)
    # margins are normalized to [0,1].
    assert np.all(crit["terminal_margins"] >= 0.0)
    assert np.all(crit["terminal_margins"] <= 1.0)
    # opp_next_action passes straight through.
    a = np.array([0.1, -0.2, 0.3, 0.4], np.float32)
    assert np.allclose(env._build_critic_features("blue_0", a)["opp_next_action"], a)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py::test_ground_truth_critic_features -x -q`
Expected: FAIL — `_build_critic_features` not defined.

- [ ] **Step 3: Implement (stub the oracle keys to zeros for now; Task 2.3 fills them)**

Add a `self._last_tag_during: bool = False` init in `__init__` (next to `self._prev_dist_to_opp`), set `self._last_tag_during = False` in `reset()` (next to the other resets), and `self._last_tag_during = tag_during` in `step()` right after `tag_during` is finalized (after the tag state machine block, ~line 383).

Add the import of the new constants to the existing `from envs.quidditch.constants import (…)` block:
`ARENA_WALL_HEIGHT` (already imported), and add `ORACLE_HORIZON_S, ORACLE_TIME_CAP_S, TAKEDOWN_CONTACT_DIST`.

Add the method (next to `_build_agent_features`):

```python
    def _world_vel(self, dofadr: int) -> np.ndarray:
        return self._world.data.qvel[dofadr:dofadr + 3].copy().astype(np.float32)

    def _build_critic_features(
        self, agent_id: str, opp_next_action: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Privileged (critic-only) features for the learner, normalized at source.

        opp_next_action is injected by the caller (OCE) — the action the
        opponent applies this step.  All other features are pure functions of
        physics state.  See the 2026-06-04 design spec §5.2."""
        red_pos, blue_pos = self._red_pos(), self._blue_pos()
        red_vel = self._world_vel(self._red_dofadr)
        blue_vel = self._world_vel(self._blue_dofadr)
        cool = (self._tag_blue_on_red.state == _TagState.COOLDOWN)

        # terminal margins (1 = safe, 0 = at boundary), clipped.
        def _wall_margin(p):
            return float(np.clip((ARENA_RADIUS - np.linalg.norm(p[:2])) / ARENA_RADIUS, 0.0, 1.0))
        def _floor_margin(p):
            return float(np.clip(p[2] / ARENA_WALL_HEIGHT, 0.0, 1.0))

        return {
            "opp_next_action": np.asarray(opp_next_action, np.float32).reshape(4),
            "red_pos_abs":     (red_pos / ARENA_RADIUS).astype(np.float32),
            "blue_pos_abs":    (blue_pos / ARENA_RADIUS).astype(np.float32),
            "tag_state_onehot": np.array(
                [float(self._last_tag_during), float(cool)], dtype=np.float32),
            "terminal_margins": np.array(
                [_wall_margin(red_pos), _wall_margin(blue_pos),
                 _floor_margin(red_pos), _floor_margin(blue_pos)], dtype=np.float32),
            # Oracle keys filled in Task 2.3:
            "self_future_disp": np.zeros(3, np.float32),
            "opp_future_rel":   np.zeros(3, np.float32),
            "score_pred":       np.zeros(3, np.float32),
            "takedown_pred":    np.zeros(3, np.float32),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_ctde_features.py
git commit -m "feat(env): ground-truth critic features (abs pos, tag state, margins)"
```

### Task 2.3: Oracle/future critic features (kinematic + sentinels)

**Files:**
- Modify: `envs/quidditch/team_env.py`
- Test: `tests/envs/quidditch/test_ctde_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/envs/quidditch/test_ctde_features.py
def test_oracle_features_finite_and_shaped():
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    for k in ("self_future_disp", "opp_future_rel", "score_pred", "takedown_pred"):
        assert crit[k].shape == (3,)
        assert np.all(np.isfinite(crit[k])), k
    # score_pred[2] is a cosine alignment in [-1, 1].
    assert -1.0001 <= float(crit["score_pred"][2]) <= 1.0001
    # takedown_pred[2] is a 0/1 flag.
    assert float(crit["takedown_pred"][2]) in (0.0, 1.0)


def test_score_pred_degenerate_velocity_is_sentinel():
    # At reset Red hovers (near-zero world velocity) -> time-to-plane sentinel = 1.0
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    assert float(crit["score_pred"][0]) == 1.0       # capped time / Tcap
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py -x -q -k oracle or degenerate`
Expected: FAIL — oracle keys are zeros (so `score_pred[0]` is 0.0, not 1.0).

- [ ] **Step 3: Replace the four oracle-key zeros with real computations**

In `_build_critic_features`, before the `return`, compute (insert after `blue_vel = …`):

```python
        learner_pos = blue_pos if agent_id == self._blue_id else red_pos
        learner_vel = blue_vel if agent_id == self._blue_id else red_vel
        opp_pos = red_pos if agent_id == self._blue_id else blue_pos
        opp_vel = red_vel if agent_id == self._blue_id else blue_vel

        k = ORACLE_HORIZON_S
        self_future_disp = (k * learner_vel) / ARENA_RADIUS
        opp_future_rel = ((opp_pos + k * opp_vel) - learner_pos) / ARENA_RADIUS

        # score_pred (attacker = red_0): will Red's current trajectory score?
        n = HOOP_OUTWARD_NORMAL.astype(np.float32)
        signed = float(np.dot(red_pos - HOOP_CENTER, n))
        v_n = float(np.dot(red_vel, n))
        Tcap = ORACLE_TIME_CAP_S
        if v_n > 1e-4 and signed < 0.0:
            t_plane = min(-signed / v_n, Tcap)
        else:
            t_plane = Tcap
        crossing = red_pos + t_plane * red_vel
        lateral = (crossing - HOOP_CENTER) - np.dot(crossing - HOOP_CENTER, n) * n
        lateral_miss = float(np.clip(np.linalg.norm(lateral) / ARENA_RADIUS, 0.0, 1.0))
        speed = float(np.linalg.norm(red_vel))
        approach_align = float(v_n / speed) if speed > 1e-6 else 0.0
        score_pred = np.array(
            [t_plane / Tcap, lateral_miss, approach_align], dtype=np.float32)

        # takedown_pred: closest point of approach between the two drones.
        r = blue_pos - red_pos
        v = blue_vel - red_vel
        vv = float(np.dot(v, v))
        t_cpa = float(np.clip(-np.dot(r, v) / vv, 0.0, Tcap)) if vv > 1e-8 else 0.0
        min_sep = float(np.linalg.norm(r + t_cpa * v))
        imminent = 1.0 if (min_sep < TAKEDOWN_CONTACT_DIST
                           and np.linalg.norm(v) > self.cfg.crash_vel_thr) else 0.0
        takedown_pred = np.array(
            [t_cpa / Tcap, float(np.clip(min_sep / ARENA_RADIUS, 0.0, 1.0)), imminent],
            dtype=np.float32)
```

Then replace the four zero placeholders in the returned dict with `self_future_disp`, `opp_future_rel`, `score_pred`, `takedown_pred`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_features.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_ctde_features.py
git commit -m "feat(env): oracle critic features (future, score-pred, takedown-pred)"
```

---

## PHASE 3 — CTDE Dict obs + opponent-action injection

### Task 3.1: ctde_mode + Dict obs for the learner

**Files:**
- Modify: `envs/quidditch/team_env.py`
- Test: `tests/envs/quidditch/test_ctde_obs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/quidditch/test_ctde_obs.py
import numpy as np
from gymnasium import spaces
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_ctde_specs_from_yaml, build_spec_from_block_names


def _ctde_env():
    actor, critic = build_ctde_specs_from_yaml("ctde_v1")
    return QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=actor,
        ctde_mode=True, critic_spec=critic,
    )


def test_learner_obs_is_dict_opponent_flat():
    env = _ctde_env()
    assert isinstance(env.observation_space("blue_0"), spaces.Dict)
    assert isinstance(env.observation_space("red_0"), spaces.Box)
    obs, _ = env.reset(seed=0)
    assert set(obs["blue_0"].keys()) == {"actor", "critic"}
    assert obs["blue_0"]["actor"].shape == (23,)
    assert obs["blue_0"]["critic"].shape == (28,)
    assert obs["red_0"].shape == (22,)        # DUEL_V1_BODY flat


def test_flat_mode_unchanged():
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False),
                           learner_id="blue_0",
                           learner_spec=build_spec_from_block_names(
                               ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS"]))
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["blue_0"], np.ndarray)   # still flat
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs.py -x -q`
Expected: FAIL — `QuidditchTeamEnv.__init__` has no `ctde_mode`.

- [ ] **Step 3: Implement**

In `team_env.py __init__` signature add kwargs `ctde_mode: bool = False` and `critic_spec: ObsSpec | None = None`. Store:

```python
        self._ctde_mode = bool(ctde_mode)
        self._critic_spec = critic_spec
        if self._ctde_mode and critic_spec is None:
            raise ValueError("ctde_mode=True requires a critic_spec")
        self._pending_opp_action = np.zeros(4, dtype=np.float32)
        self._opponent_act = None     # set by OCE; (flat_opp_obs) -> action[4]
```

Replace the `observation_spaces` dict-comprehension with a helper call:

```python
        self.observation_spaces: dict[str, spaces.Space] = {
            agent: self._obs_space_for_agent(agent) for agent in self.possible_agents
        }
```

Add the helper + Dict obs builders (near `_spec_for_agent`):

```python
    def _obs_space_for_agent(self, agent_id: str) -> spaces.Space:
        if self._ctde_mode and agent_id == self._learner_id:
            return spaces.Dict({
                "actor":  spaces.Box(-np.inf, np.inf,
                                     (self._learner_spec.dim,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf,
                                     (self._critic_spec.dim,), np.float32),
            })
        return spaces.Box(-np.inf, np.inf,
                          (self._spec_for_agent(agent_id).dim,), np.float32)
```

Change `_build_agent_obs` to branch for the CTDE learner:

```python
    def _build_agent_obs(self, agent_id: str):
        if self._ctde_mode and agent_id == self._learner_id:
            feats = self._build_agent_features(agent_id)
            actor = obs_spec.pack_normalized(self._learner_spec, feats,
                                             obs_spec.NORM_BY_BLOCK)
            opp_a = self._opp_next_action()
            crit_feats = self._build_critic_features(agent_id, opp_a)
            critic = obs_spec.pack_normalized(self._critic_spec, crit_feats,
                                              obs_spec.NORM_BY_BLOCK)
            return {"actor": actor, "critic": critic}
        return self._pack_agent_obs(agent_id, self._spec_for_agent(agent_id))

    def _opp_next_action(self) -> np.ndarray:
        """Deterministic opponent action for the current state (or zeros).

        Computed here so it lands in obs_t's critic view; cached so OCE applies
        the exact same action this step (embedded == applied)."""
        if self._opponent_act is None:
            return np.zeros(4, dtype=np.float32)
        opp_flat = self._pack_agent_obs(self._opponent_id(),
                                        self._spec_for_agent(self._opponent_id()))
        a = np.asarray(self._opponent_act(opp_flat), dtype=np.float32).reshape(4)
        self._pending_opp_action = a
        return a

    def _opponent_id(self) -> str:
        return self._blue_id if self._learner_id == self._red_id else self._red_id
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_ctde_obs.py
git commit -m "feat(env): CTDE Dict obs for the learner (opponent stays flat)"
```

### Task 3.2: OCE drives the opponent from the embedded action

**Files:**
- Modify: `envs/quidditch/opponents.py`
- Test: `tests/envs/quidditch/test_ctde_obs.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/envs/quidditch/test_ctde_obs.py
from envs.quidditch.opponents import OpponentControlledEnv, BeelineRed


def test_oce_ctde_embedded_action_equals_applied():
    env = _ctde_env()
    oce = OpponentControlledEnv(env, learner_id="blue_0", opponent=BeelineRed())
    obs, _ = oce.reset(seed=0)
    assert set(obs.keys()) == {"actor", "critic"}
    # opp_next_action is the first 4 dims of the (unnormalized passthrough) critic.
    embedded = obs["critic"][:4].copy()
    # Stepping applies env._pending_opp_action; it must equal what was embedded.
    np.testing.assert_allclose(env._pending_opp_action, embedded, atol=1e-6)
    oce.step(np.zeros(4, np.float32))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs.py::test_oce_ctde_embedded_action_equals_applied -x -q`
Expected: FAIL — OCE doesn't set `_opponent_act` / doesn't apply `_pending_opp_action`.

- [ ] **Step 3: Implement the CTDE branch in OCE**

In `OpponentControlledEnv.__init__`, after `self.opponent = opponent`:

```python
        # CTDE: let the team env compute opp_next_action at obs-build time so it
        # lands in the learner's critic view; we then apply the exact same
        # cached action (embedded == applied).
        self._ctde = getattr(team_env, "_ctde_mode", False)
        if self._ctde:
            team_env._opponent_act = self.opponent.act
```

In `reset()` (after `self.opponent.reset()`), nothing extra is needed — the env's `_build_agent_obs` already called `_opp_next_action()` during `team_env.reset()`. In `step()`, branch the opponent action source:

```python
    def step(self, action):
        if self._ctde:
            opp_action = self.team_env._pending_opp_action
        else:
            opp_action = self.opponent.act(self._last_opp_obs)
        actions = {self.learner_id: action, self.opponent_id: opp_action}
        obs, rew, term, trunc, infos = self.team_env.step(actions)
        self._last_opp_obs = obs[self.opponent_id]
        self.last_team_infos = infos
        return (obs[self.learner_id], float(rew[self.learner_id]),
                bool(term[self.learner_id]), bool(trunc[self.learner_id]),
                infos[self.learner_id])
```

(`reset()` already returns `obs[self.learner_id]`, which is the Dict for a CTDE learner — no change needed there.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_obs.py -q`
Expected: PASS (all). Also run `uv run pytest tests/integration/test_team_env_canary.py -q` — flat OCE path unchanged (`_ctde=False`).

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/opponents.py tests/envs/quidditch/test_ctde_obs.py
git commit -m "feat(env): OCE applies the env-embedded opp action in CTDE mode"
```

---

## PHASE 4 — Frame stacking + factory wiring

### Task 4.1: SelectiveDictFrameStack

**Files:**
- Create: `envs/quidditch/dict_frame_stack.py`
- Test: `tests/envs/quidditch/test_dict_frame_stack.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/quidditch/test_dict_frame_stack.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.quidditch.dict_frame_stack import SelectiveDictFrameStack


class _DictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "actor":  spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "critic": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return {"actor": np.full(2, self._i, np.float32),
                "critic": np.full(3, self._i, np.float32)}, {}

    def step(self, a):
        self._i += 1
        return ({"actor": np.full(2, self._i, np.float32),
                 "critic": np.full(3, self._i, np.float32)}, 0.0,
                self._i >= 5, False, {})


def test_stacks_actor_only():
    vec = SelectiveDictFrameStack(DummyVecEnv([_DictEnv]), n_stack=3, keys=("actor",))
    assert vec.observation_space["actor"].shape == (6,)    # 2 * 3
    assert vec.observation_space["critic"].shape == (3,)   # unstacked
    obs = vec.reset()
    assert obs["actor"].shape == (1, 6)
    assert obs["critic"].shape == (1, 3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_dict_frame_stack.py -x -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement (reuse SB3's per-key StackedObservations)**

```python
# envs/quidditch/dict_frame_stack.py
"""VecFrameStack variant that stacks only selected keys of a Dict obs.

SB3's VecFrameStack stacks every key of a Dict space with the same depth; the
CTDE critic view already encodes temporal/future content, so we stack only the
actor key and pass the critic key through.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


class SelectiveDictFrameStack(VecEnvWrapper):
    def __init__(self, venv, n_stack: int, keys: tuple[str, ...] = ("actor",)) -> None:
        assert isinstance(venv.observation_space, spaces.Dict)
        self._keys = tuple(keys)
        new_spaces: dict = {}
        self._stackers: dict[str, StackedObservations] = {}
        for key, sub in venv.observation_space.spaces.items():
            if key in self._keys:
                st = StackedObservations(venv.num_envs, n_stack, sub)
                self._stackers[key] = st
                new_spaces[key] = st.stacked_observation_space
            else:
                new_spaces[key] = sub
        super().__init__(venv, observation_space=spaces.Dict(new_spaces))

    def reset(self):
        obs = self.venv.reset()
        out = dict(obs)
        for key, st in self._stackers.items():
            out[key] = st.reset(np.asarray(obs[key]))
        return out

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        out = dict(obs)
        for key, st in self._stackers.items():
            stacked, infos = st.update(np.asarray(obs[key]), dones, infos)
            out[key] = stacked
        return out, rewards, dones, infos
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_dict_frame_stack.py -x -q`
Expected: PASS. *If `StackedObservations.update` returns a different tuple arity in 2.8.0, adapt to its signature (`update(obs, dones, infos)` → `(stacked_obs, infos)`).*

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/dict_frame_stack.py tests/envs/quidditch/test_dict_frame_stack.py
git commit -m "feat(env): SelectiveDictFrameStack (stack actor key only)"
```

### Task 4.2: Factory CTDE wiring

**Files:**
- Modify: `envs/quidditch/env_factories.py`, `envs/quidditch/opponents.py`
- Test: `tests/envs/quidditch/test_ctde_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/quidditch/test_ctde_factory.py
from gymnasium import spaces
from envs.quidditch.env_factories import TeamEnvFactory
from envs.quidditch.team_env import TeamConfig


def _factory(**kw):
    return TeamEnvFactory(
        n_envs=1, team_cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", opponent_spec="beeline_red",
        obs_blocks=[], obs_name="CTDE_V1", frame_stack=3,
        ctde_mode=True, obs_stem="ctde_v1", **kw)


def test_factory_builds_dict_train_env():
    vec = _factory().build_train_env()
    assert isinstance(vec.observation_space, spaces.Dict)
    assert vec.observation_space["actor"].shape == (69,)   # 23 * 3
    assert vec.observation_space["critic"].shape == (28,)  # unstacked
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/quidditch/test_ctde_factory.py -x -q`
Expected: FAIL — `TeamEnvFactory` has no `ctde_mode`/`obs_stem`.

- [ ] **Step 3: Implement**

In `env_factories.py`, add fields to `TeamEnvFactory`:

```python
    ctde_mode: bool = False
    obs_stem: str = ""        # conf/obs/<stem>.yaml for build_ctde_specs_from_yaml
```

Replace `_make_thunk` to build CTDE envs when `ctde_mode`:

```python
    def _make_thunk(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        from envs.quidditch.obs_spec import (
            build_spec_from_block_names, build_ctde_specs_from_yaml,
        )
        cfg, learner, opp_spec = self.team_cfg, self.learner_id, self.opponent_spec
        reward_stack = self.reward_stack
        ctde, stem = self.ctde_mode, self.obs_stem
        if ctde:
            actor_spec, critic_spec = build_ctde_specs_from_yaml(stem)
        else:
            actor_spec, critic_spec = build_spec_from_block_names(self.obs_blocks), None

        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, reward_stack=reward_stack,
                learner_id=learner, learner_spec=actor_spec,
                ctde_mode=ctde, critic_spec=critic_spec,
            )
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
        return _thunk
```

Add a stack helper and use it in `build_train_env`/`build_eval_env`:

```python
    def _wrap_stack(self, vec):
        if self.frame_stack <= 1:
            return vec
        if self.ctde_mode:
            from envs.quidditch.dict_frame_stack import SelectiveDictFrameStack
            return SelectiveDictFrameStack(vec, n_stack=self.frame_stack, keys=("actor",))
        return VecFrameStack(vec, n_stack=self.frame_stack)

    def build_train_env(self) -> VecEnv:
        thunk = self._make_thunk()
        env_fns = [thunk for _ in range(self.n_envs)]
        vec = SubprocVecEnv(env_fns) if self.n_envs > 1 else DummyVecEnv(env_fns)
        return self._wrap_stack(vec)

    def build_eval_env(self) -> VecEnv:
        return self._wrap_stack(DummyVecEnv([self._make_thunk()]))
```

In `build_video_env_fn`, thread CTDE into the per-env build and use a Dict-aware single-env stack (Task 4.2b below extends `FrameStackWrapper`):

```python
        ctde, stem = self.ctde_mode, self.obs_stem
        if ctde:
            from envs.quidditch.obs_spec import build_ctde_specs_from_yaml
            actor_spec, critic_spec = build_ctde_specs_from_yaml(stem)
        else:
            actor_spec, critic_spec = build_spec_from_block_names(self.obs_blocks), None
        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, render_mode="rgb_array", reward_stack=reward_stack,
                learner_id=learner, learner_spec=actor_spec,
                ctde_mode=ctde, critic_spec=critic_spec,
            )
            opp = from_spec(opp_spec, deterministic=True)
            env = OpponentControlledEnv(team, learner_id=learner, opponent=opp)
            if frame_stack > 1:
                return FrameStackWrapper(env, n_stack=frame_stack,
                                         stack_keys=("actor",) if ctde else None)
            return env
        return _thunk
```

**Task 4.2b — `FrameStackWrapper` Dict support.** In `opponents.py`, extend `FrameStackWrapper.__init__` to accept `stack_keys: tuple[str, ...] | None = None`. When the wrapped obs space is a `gym.spaces.Dict` and `stack_keys` is set, stack only those keys (per-key ring buffers); otherwise keep the existing 1-D path. Add a focused unit test `tests/envs/quidditch/test_frame_stack_wrapper_dict.py` asserting the actor key grows and the critic key does not. *(Mirror the buffer logic already in the class; one ring buffer per stacked key.)*

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/envs/quidditch/test_ctde_factory.py tests/envs/quidditch/test_frame_stack_wrapper_dict.py -q`
Expected: PASS. Run `uv run pytest tests/unit/test_env_factories.py -q` — flat factory defaults unchanged.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/env_factories.py envs/quidditch/opponents.py tests/envs/quidditch/test_ctde_factory.py tests/envs/quidditch/test_frame_stack_wrapper_dict.py
git commit -m "feat(env): TeamEnvFactory + FrameStackWrapper CTDE/Dict support"
```

### Task 4.3: build_callbacks dict-aware eval stacking

**Files:**
- Modify: `scripts/_train_common.py`
- Test: `tests/scripts/test_build_callbacks_ctde.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_callbacks_ctde.py
from gymnasium import spaces
from envs.quidditch.env_factories import TeamEnvFactory
from envs.quidditch.team_env import TeamConfig
from scripts._train_common import build_callbacks


def _legacy_cfg():
    return {"training": {"eval": {"eval_freq_steps": 1000, "n_eval_episodes": 1},
                         "callbacks": {"checkpoint_freq_steps": 1000,
                                       "video_every_n_evals": 2, "video_fps": 20}}}


def test_eval_env_is_dict_stacked_for_ctde(tmp_path):
    f = TeamEnvFactory(n_envs=1, team_cfg=TeamConfig(randomise_red_start=False),
                       learner_id="blue_0", opponent_spec="beeline_red",
                       obs_blocks=[], obs_name="CTDE_V1", frame_stack=3,
                       ctde_mode=True, obs_stem="ctde_v1")
    cbs = build_callbacks(run_dir=tmp_path, eval_env_fn=f._make_thunk(),
                          config=_legacy_cfg(), n_envs=1, frame_stack=3,
                          ctde_mode=True)
    eval_cb = next(c for c in cbs if hasattr(c, "eval_env"))
    assert isinstance(eval_cb.eval_env.observation_space, spaces.Dict)
    assert eval_cb.eval_env.observation_space["actor"].shape == (69,)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/scripts/test_build_callbacks_ctde.py -x -q`
Expected: FAIL — `build_callbacks` has no `ctde_mode` kwarg.

- [ ] **Step 3: Implement**

In `_train_common.build_callbacks`, add `ctde_mode: bool = False` to the signature and replace the eval-env stacking block:

```python
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_fn())])
    if frame_stack > 1:
        if ctde_mode:
            from envs.quidditch.dict_frame_stack import SelectiveDictFrameStack
            eval_env = SelectiveDictFrameStack(eval_env, n_stack=frame_stack, keys=("actor",))
        else:
            eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/scripts/test_build_callbacks_ctde.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_train_common.py tests/scripts/test_build_callbacks_ctde.py
git commit -m "feat(train): build_callbacks dict-aware eval frame stacking"
```

---

## PHASE 5 — Train wiring, obs-compat, experiment, smoke

### Task 5.1: PolicyConfig + conf/policy + default

**Files:**
- Modify: `config_schema.py`, `conf/config.yaml`
- Create: `conf/policy/mlp.yaml`, `conf/policy/asymmetric.yaml`
- Test: `tests/test_ctde_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ctde_config.py
from config_schema import PolicyConfig


def test_policy_config_defaults():
    p = PolicyConfig()
    assert p.policy_class == "MlpPolicy"
    assert p.share_features_extractor is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ctde_config.py::test_policy_config_defaults -x -q`
Expected: FAIL — `PolicyConfig` not importable.

- [ ] **Step 3: Implement**

In `config_schema.py` add the dataclass, register it, and add it to `Config`:

```python
@dataclass
class PolicyConfig:
    """Selects the SB3 policy + arch.  MlpPolicy = flat path; AsymmetricActorCriticPolicy = CTDE."""
    policy_class: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    share_features_extractor: bool = True
```

In `Config`, add: `policy: PolicyConfig = field(default_factory=PolicyConfig)`. In `register_configs`, add: `cs.store(group="policy", name="schema", node=PolicyConfig)`.

`conf/policy/mlp.yaml`:

```yaml
policy_class: MlpPolicy
net_arch: [64, 64]
share_features_extractor: true
```

`conf/policy/asymmetric.yaml`:

```yaml
# CTDE privileged-critic policy (core.policies.asymmetric.AsymmetricActorCriticPolicy).
policy_class: AsymmetricActorCriticPolicy
net_arch: [64, 64]
share_features_extractor: false
```

In `conf/config.yaml`, add `- policy: mlp` to the `defaults:` list (e.g. after `- trainer: ppo`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ctde_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config_schema.py conf/config.yaml conf/policy/mlp.yaml conf/policy/asymmetric.yaml tests/test_ctde_config.py
git commit -m "feat(config): policy group (mlp | asymmetric)"
```

### Task 5.2: _build_or_load_model — policy dispatch + CTDE scratch path

**Files:**
- Modify: `scripts/train.py`
- Test: covered by the Task 5.5 smoke (unit-testing `_build_or_load_model` needs a vec env; the smoke exercises it end-to-end).

- [ ] **Step 1: Implement the dispatch**

At the top of `_build_or_load_model`, after building `ppo_kwargs`, resolve the policy + obs mode:

```python
    obs_mode = cfg.obs.get("obs_mode", "flat")
    policy_class_name = cfg.get("policy", {}).get("policy_class", "MlpPolicy")
    net_arch = list(cfg.get("policy", {}).get("net_arch", [64, 64]))

    if obs_mode == "dict":
        from core.policies.asymmetric import AsymmetricActorCriticPolicy
        from envs.quidditch.obs_spec import build_ctde_specs_from_yaml
        # Resolve the obs stem from cfg.obs.name (CTDE_V1 -> ctde_v1) unless an
        # explicit obs_stem is provided.
        stem = cfg.obs.get("obs_stem") or str(cfg.obs.name).lower()
        _actor_spec, _critic_spec = build_ctde_specs_from_yaml(stem)  # validates the yaml
        if cfg.init.mode != "scratch":
            raise SystemExit(
                "CTDE (obs_mode=dict) supports init=scratch only — flat↔dict "
                "warm-start is not defined.  Set init=scratch.")
        return PPO(
            AsymmetricActorCriticPolicy, vec_env,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=None, seed=seed, verbose=0, **ppo_kwargs,
        ), 0
```

Place this block **before** the existing `if cfg.init.mode == "scratch":` flat branch so dict mode is handled first. The flat branches are unchanged.

- [ ] **Step 2: Quick compile check**

Run: `uv run python -c "import scripts.train"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): CTDE policy/obs dispatch in _build_or_load_model"
```

### Task 5.3: Inline eval_env_fn + callbacks CTDE threading

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Implement**

In `main()`, the team-branch `eval_env_fn` (line ~336) must mirror the factory's CTDE wiring. Replace the `learner_spec = …` line and the `eval_env_fn` body with:

```python
        learner = env_factory.learner_id
        ctde = (cfg.obs.get("obs_mode", "flat") == "dict")
        if ctde:
            from envs.quidditch.obs_spec import build_ctde_specs_from_yaml
            stem = cfg.obs.get("obs_stem") or str(cfg.obs.name).lower()
            actor_spec, critic_spec = build_ctde_specs_from_yaml(stem)
        else:
            actor_spec = build_spec_from_block_names(cfg.obs.blocks)
            critic_spec = None

        def eval_env_fn():
            team = QuidditchTeamEnv(
                cfg=team_cfg, reward_stack=env_factory.reward_stack,
                learner_id=learner, learner_spec=actor_spec,
                ctde_mode=ctde, critic_spec=critic_spec,
            )
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
```

Pass `ctde_mode` into `build_callbacks`:

```python
    callbacks = build_callbacks(
        run_dir=run_dir, eval_env_fn=eval_env_fn, config=legacy_cfg,
        n_envs=cfg.env.n_envs, video_env_fn=video_env_fn, verbose=0,
        frame_stack=frame_stack,
        ctde_mode=(cfg.obs.get("obs_mode", "flat") == "dict"),
    )
```

Thread `ctde_mode` + `obs_stem` into the factory instantiation. The factory is built by `_build_env_factory` from `cfg.env`; add to `conf/env/team.yaml` (Task 5.5) the fields `ctde_mode: ${obs.obs_mode == dict ...}`. Simpler and explicit: in `_build_env_factory`, after the `TeamEnvFactory` branch, set the CTDE fields from cfg:

```python
    if cfg.env._target_.endswith("TeamEnvFactory"):
        env_cfg.pop("team_env_params", None)
        extra["team_cfg"] = _build_team_cfg(cfg)
        extra["opponent_spec"] = _opponent_spec_from_cfg(cfg)
        if cfg.obs.get("obs_mode", "flat") == "dict":
            extra["ctde_mode"] = True
            extra["obs_stem"] = cfg.obs.get("obs_stem") or str(cfg.obs.name).lower()
```

- [ ] **Step 2: Compile check**

Run: `uv run python -c "import scripts.train"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): thread CTDE through eval_env_fn, factory, callbacks"
```

### Task 5.4: obs_compat flat↔dict incompatibility

**Files:**
- Modify: `core/obs_compat.py`
- Test: `tests/test_obs_compat_ctde.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obs_compat_ctde.py
import pytest
from core.obs_compat import obs_modes_compatible


def test_flat_to_dict_is_incompatible():
    assert obs_modes_compatible("flat", "flat") is True
    assert obs_modes_compatible("dict", "dict") is True
    assert obs_modes_compatible("flat", "dict") is False
    assert obs_modes_compatible("dict", "flat") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_obs_compat_ctde.py -x -q`
Expected: FAIL — `obs_modes_compatible` not defined.

- [ ] **Step 3: Implement**

Append to `core/obs_compat.py`:

```python
def obs_modes_compatible(parent_mode: str, child_mode: str) -> bool:
    """flat↔dict is never compatible: a CTDE (dict) run can only init=scratch.

    Same-mode is compatible at this coarse level; block-level compat is still
    judged by preflight() for flat↔flat, and (future) actor-spec-only for
    dict↔dict."""
    return parent_mode == child_mode
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_obs_compat_ctde.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/obs_compat.py tests/test_obs_compat_ctde.py
git commit -m "feat(obs-compat): flat<->dict incompatibility helper (CTDE scratch-only)"
```

### Task 5.5: Experiment YAML + end-to-end smoke

**Files:**
- Create: `conf/experiment/blue_oracle_v1.yaml`, `tests/scripts/test_ctde_train_smoke.py`

- [ ] **Step 1: Write the experiment + the failing smoke test**

`conf/experiment/blue_oracle_v1.yaml`:

```yaml
# @package _global_
# First CTDE run: privileged oracle critic + world-frame actor.  Blue defender
# vs beeline_red (scripted, always on disk), scratch.  The frozen:red_v1 rung
# follows once models/ is restored.  See the 2026-06-04 design spec.
run_name: ppo_hoop_blue_oracle_1

defaults:
  - override /env: team
  - override /obs: ctde_v1
  - override /reward: team_v3_intercept
  - override /opponent: beeline_red
  - override /policy: asymmetric
  - override /init: scratch
  - override /curriculum: random_start

env:
  learner_id: blue_0

trainer:
  lr: 3e-4
  gamma: 0.9995
  gae_lambda: 0.98
  total_timesteps: 10_000_000
```

`tests/scripts/test_ctde_train_smoke.py`:

```python
# tests/scripts/test_ctde_train_smoke.py
import os
import pytest
from hydra import initialize, compose
from config_schema import register_configs


@pytest.mark.slow
def test_ctde_experiment_trains_a_few_steps(tmp_path, monkeypatch):
    os.environ["WANDB_MODE"] = "disabled"
    register_configs()
    monkeypatch.chdir(tmp_path)
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=[
            "+experiment=blue_oracle_v1",
            "env.n_envs=1",
            "trainer.total_timesteps=512",
            "trainer.n_steps=128",
            "trainer.batch_size=64",
            "eval.eval_freq_steps=128",          # force at least one eval boundary
            "eval.n_eval_episodes=1",
            "eval.video.enabled=false",
        ])
        from scripts.train import main
        main(cfg)   # must not raise: Dict obs rollout + an eval both exercised
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/scripts/test_ctde_train_smoke.py -x -q`
Expected: FAIL first because `conf/obs/ctde_v1.yaml` lacks the flat `blocks:` key that some code paths read. Resolve any `cfg.obs.blocks` access on the dict path (the dispatch added in 5.2/5.3 should avoid `cfg.obs.blocks` when `obs_mode==dict`; `frame_stack = int(cfg.obs.n_stack)` is fine). Iterate until green.

- [ ] **Step 3: Fix fall-through reads**

Audit `scripts/train.py` for unconditional `cfg.obs.blocks` reads on the dict path:
- `_build_or_load_model`: the flat branch computes `current_spec = build_spec_from_block_names(cfg.obs.blocks)` at function top — move it **inside** the non-dict branches (after the dict early-return) so dict mode never touches `cfg.obs.blocks`.
- `main()`: `learner_spec = build_spec_from_block_names(cfg.obs.blocks)` is replaced in Task 5.3.
Add `obs_stem: ""` to `ObsConfig` if `cfg.obs.get("obs_stem")` is consulted (optional; `.get` already tolerates absence).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/scripts/test_ctde_train_smoke.py -q`
Expected: PASS (one `learn(512)` with a Dict-obs rollout and one eval boundary, WANDB disabled).

- [ ] **Step 5: Commit**

```bash
git add conf/experiment/blue_oracle_v1.yaml tests/scripts/test_ctde_train_smoke.py scripts/train.py
git commit -m "feat(train): blue_oracle_v1 CTDE experiment + end-to-end smoke"
```

---

## Final verification

- [ ] **Full suite + canaries byte-identical**

Run: `uv run pytest -q` (or `make test`)
Expected: all green; `tests/integration/test_scoring_canary.py` still asserts `SCORED at step 434 / total reward 7.3837`; the team-env canary unchanged (CTDE is opt-in; defaults are `obs_mode=flat`, `policy=mlp`).

- [ ] **Cross-process pickling smoke** (catches `SubprocVecEnv` spec / Dict-space issues)

Run: `uv run pytest tests/scripts/test_ctde_train_smoke.py -q` with `env.n_envs=2` (edit the override locally, or add a parametrized variant). Expected: PASS — the frozen specs pickle and Dict obs survive subprocess round-trips.

- [ ] **Manual real run (user-driven, NOT part of CI):**

`make train EXP=blue_oracle_v1` — Blue vs `beeline_red`, scratch. Watch W&B for higher critic explained-variance / lower value-loss than a flat-critic baseline, rising score-prevention/takedown rate, and decisive multi-second behavior under gamma=0.9995. Per memory `feedback_verify_before_commit`, qualitative viewer confirmation (`make eval-team … GUI=1`) is the user's call before any promotion.

---

## Self-Review Checklist (completed at authoring)

- **Spec coverage:** §4 architecture → Phase 0 + Phase 3/5; §5.1 actor blocks → Task 1.1/1.3/2.1; §5.2 critic blocks → 1.1/2.2/2.3; §5.3 normalization → 1.2; §5.4 frame-stack → 4.1/4.2/4.3; §6.1 opp-action timing → 3.1/3.2; §6.2 policy/gradient isolation → 0.2; §6.3 eval → automatic via pi extractor (3.1 returns Dict, actor ignores critic); §7 obs-spec/persistence/compat → 1.x + 5.4 (persistence rides on existing `.hydra/config.yaml` dump of `cfg.obs`); §8 gamma/λ → 5.5 experiment YAML; §9 reward unchanged → reused `team_v3_intercept`; §10 edge cases → sentinels (2.3), cache (3.2), threading (5.3); §11 tests → every task; §12 build order → phases match; §13/14 scope/files → File Structure.
- **Placeholder scan:** Task 4.2b ("mirror the buffer logic") and the optional `StackedObservations` arity note are the only adaptive steps; both name the exact method to mirror and the fallback. No TBD/TODO.
- **Type consistency:** `ctde_mode`/`critic_spec`/`obs_stem` names are identical across `team_env`, `env_factories`, `build_callbacks`, `train.py`; feature-dict keys match `ObsBlock.name` strings (`lin_vel_world`, `time_remaining`, `opp_next_action`, `self_future_disp`, `opp_future_rel`, `score_pred`, `takedown_pred`, `red_pos_abs`, `blue_pos_abs`, `tag_state_onehot`, `terminal_margins`); `pack_normalized(spec, values, norm_by_block)` signature consistent between 1.2 and its callers in 3.1.
