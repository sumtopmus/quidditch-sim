"""Warm-start preserves single-agent behavior on obs[:16] when obs[16:] is
zeroed out.

Ported from scripts/check_team_warm.py.  Requires a trained single-agent
checkpoint identified by MODEL=<run-name>; the test resolves it as
models/<run-name>/best_model.zip.  Skipped if MODEL is unset.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.policies.warm_start import warm_start_ppo
from envs.quidditch.opponents import BeelineBlue, OpponentControlledEnv
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        "MODEL" not in os.environ,
        reason="set MODEL=<run-name> to run warm-start regression",
    ),
]


def test_warm_started_policy_matches_old_on_obs_prefix() -> None:
    old_path = f"models/{os.environ['MODEL']}/best_model.zip"
    n_samples = 64
    tol = 0.1

    old = PPO.load(old_path)

    def _env_fn():
        return OpponentControlledEnv(
            QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False)),
            learner_id="red_0",
            opponent=BeelineBlue(),
        )

    new_env = DummyVecEnv([_env_fn])
    try:
        new = warm_start_ppo(
            old_checkpoint=old_path,
            new_env=new_env,
            new_input_dim=22,
            old_input_dim=16,
            new_dim_init_scale=0.01,
        )

        rng = np.random.default_rng(42)
        obs_seed = new_env.reset()
        diffs: list[float] = []
        for _ in range(n_samples):
            obs_zeroed = obs_seed.copy()
            obs_zeroed[..., 16:] = 0.0
            action,     _ = new.predict(obs_zeroed,         deterministic=True)
            old_action, _ = old.predict(obs_seed[..., :16], deterministic=True)
            diffs.append(float(np.linalg.norm(action - old_action)))
            random_a = rng.uniform(-1, 1, size=action.shape).astype(np.float32)
            obs_seed, _, _, _ = new_env.step(random_a)

        mean_diff = float(np.mean(diffs))
        assert mean_diff < tol, (
            f"warm-started policy drifts {mean_diff:.4f} > tol {tol}"
        )
    finally:
        new_env.close()
