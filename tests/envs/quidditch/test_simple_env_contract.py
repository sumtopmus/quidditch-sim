"""SB3 env-checker contract + zero-action smoke episode for QuidditchSimpleEnv.

Ported from parts 1 & 2 of scripts/check_env.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.quidditch.simple_env import QuidditchSimpleEnv

pytestmark = pytest.mark.slow


def test_sb3_check_env_passes() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        check_env(env, warn=True)
    finally:
        env.close()


def test_zero_action_episode_runs_10_steps() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        obs, _ = env.reset()
        assert obs.shape == (16,), f"unexpected obs shape {obs.shape}"
        assert obs.dtype == np.float32, f"unexpected obs dtype {obs.dtype}"

        for _ in range(10):
            action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, _info = env.step(action)
            assert np.isfinite(reward), f"non-finite reward: {reward}"
            if terminated or truncated:
                break
    finally:
        env.close()
