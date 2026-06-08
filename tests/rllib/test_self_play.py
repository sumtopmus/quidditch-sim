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
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
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
