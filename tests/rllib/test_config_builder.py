"""Tests that the Hydra→PPOConfig builder produces a valid new-stack config."""
from __future__ import annotations

from omegaconf import OmegaConf

from rllib.config_builder import build_ppo_config


def _cfg():
    return OmegaConf.create({
        "seed": 7,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
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
