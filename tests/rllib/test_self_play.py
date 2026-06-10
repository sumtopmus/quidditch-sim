"""Step 2 two-policy self-play: both mains trainable + gradient flow."""
from __future__ import annotations

import pytest
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
