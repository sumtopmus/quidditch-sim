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
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
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
