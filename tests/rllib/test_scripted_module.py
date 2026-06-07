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
