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


def test_team_cfg_threaded_from_curriculum_and_env():
    """cfg.curriculum + cfg.env.team_env_params reach env_config['team_cfg']."""
    cfg = _cfg()
    cfg.curriculum = {
        "randomise_start": False,
        "episode_seconds": 30.0,
        "red_start_pos": [-1.0, 0.0, 0.5],
        "red_start_yaw": 0.0,
    }
    cfg.env = {"team_env_params": {
        "red_prefix": "red_0", "blue_prefix": "blue_0", "hoop_prefix": "hoop_0",
        "midpoint_alpha": 0.3, "tag_radius": 0.3, "tag_cooldown_s": 1.0,
        "crash_vel_thr": 1.0, "walls_collide": True,
    }}
    config = build_ppo_config(cfg)
    tc = config.env_config["team_cfg"]
    assert tc["randomise_red_start"] is False
    assert list(tc["red_start_pos"]) == [-1.0, 0.0, 0.5]
    assert tc["tag_radius"] == 0.3


def test_team_cfg_empty_when_no_env_or_curriculum():
    """Bare cfg (the unit-test shape) yields no overrides → TeamConfig defaults."""
    config = build_ppo_config(_cfg())
    assert config.env_config["team_cfg"] == {}


def test_score_metrics_callback_registered():
    from rllib.metrics import ScoreMetricsCallback
    config = build_ppo_config(_cfg())
    assert config.callbacks_class is ScoreMetricsCallback
