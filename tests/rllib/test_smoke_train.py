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
