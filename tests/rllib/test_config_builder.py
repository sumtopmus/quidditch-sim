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


def test_entropy_coeff_scalar_when_no_schedule():
    config = build_ppo_config(_cfg())
    assert config.entropy_coeff == 0.01


def test_entropy_coeff_schedule_passed_through():
    """A schedule under algo.entropy_coeff_schedule overrides the scalar and
    reaches the config as a plain list (anneal-to-zero stabilizer)."""
    cfg = _cfg()
    cfg.algo.entropy_coeff_schedule = [[0, 0.01], [1000, 0.0]]
    config = build_ppo_config(cfg)
    assert config.entropy_coeff == [[0, 0.01], [1000, 0.0]]


def test_grad_clip_passed_when_present():
    cfg = _cfg()
    cfg.algo.grad_clip = 1.0
    config = build_ppo_config(cfg)
    assert config.grad_clip == 1.0


def _league_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        "league": {"enabled": True, "snapshot_threshold": 0.7,
                   "min_iters_between_snapshots": 20, "population_cap": 5,
                   "live_fraction": 0.5},
    })


def test_league_callback_and_mapping_wired_when_enabled():
    from rllib.config_builder import build_ppo_config
    from rllib.league import LeagueCallback
    cfg = _league_cfg()
    config = build_ppo_config(cfg)
    # LeagueCallback is among the registered callbacks.
    cbs = config.callbacks_class
    classes = cbs if isinstance(cbs, (list, tuple)) else [cbs]
    assert LeagueCallback in classes
    # The league knobs are stashed in env_config for the callback to read.
    assert config.env_config["league"]["snapshot_threshold"] == 0.7
    # The mapping fn routes the two mains live when populations are empty.
    import types
    fn = config.policy_mapping_fn
    ep = types.SimpleNamespace(id_=7)
    assert fn("red_0", ep) == "main_red"
    assert fn("blue_0", ep) == "main_blue"


def test_no_league_callback_when_league_absent():
    """Step-2 configs (no `league` group) keep the static mapping + score-only
    callback — league wiring is opt-in."""
    from rllib.config_builder import build_ppo_config
    from rllib.league import LeagueCallback
    cfg = _league_cfg()
    del cfg["league"]
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    classes = cbs if isinstance(cbs, (list, tuple)) else [cbs]
    assert LeagueCallback not in classes


def test_eval_battery_callback_registered_before_league_when_enabled():
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.metrics import ScoreMetricsCallback
    from rllib.eval_battery import EvalBatteryCallback
    from rllib.league import LeagueCallback

    cfg = _league_cfg()  # the file's existing league-enabled inline cfg factory
    cfg = OmegaConf.merge(cfg, OmegaConf.create(
        {"league": {"eval_enabled": True, "eval_interval_iters": 10,
                    "eval_episodes": 4, "eval_seed": 0}}))
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    assert isinstance(cbs, (list, tuple))
    names = [c.__name__ for c in cbs]
    # Order matters: EvalBatteryCallback writes result["eval"] before the league
    # gate reads it in the same on_train_result sweep.
    assert names == ["ScoreMetricsCallback", "EvalBatteryCallback", "LeagueCallback"]
    assert EvalBatteryCallback in cbs and LeagueCallback in cbs
    assert ScoreMetricsCallback in cbs


def test_eval_battery_callback_absent_when_disabled():
    from rllib.config_builder import build_ppo_config
    from rllib.eval_battery import EvalBatteryCallback

    cfg = _league_cfg()
    cfg.league.eval_enabled = False
    config = build_ppo_config(cfg)
    cbs = config.callbacks_class
    cbs = list(cbs) if isinstance(cbs, (list, tuple)) else [cbs]
    assert EvalBatteryCallback not in cbs


def test_curriculum_levers_and_schedules_thread_into_env_config():
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config

    cfg = _league_cfg()  # existing league-enabled inline cfg factory in this file
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"curriculum": {
        "randomise_start": False, "episode_seconds": 30.0,
        "red_start_pos": [0.5, 0.0, 2.0], "red_start_yaw": 0.0,
        "red_action_scale": 0.6, "red_start_r_max": None,
        "dense_scale_schedule": [[0, 1.0], [1000, 0.0]],
        "red_action_scale_schedule": [[0, 0.6], [1000, 1.0]],
        "red_start_r_max_schedule": None,
    }}))
    config = build_ppo_config(cfg)
    ec = config.env_config
    # Static lever lands in team_cfg (env construction reads it).
    assert ec["team_cfg"]["red_action_scale"] == 0.6
    # Schedules land in a dedicated curriculum block (the callback reads it).
    assert ec["curriculum"]["dense_scale_schedule"] == [[0, 1.0], [1000, 0.0]]
    assert ec["curriculum"]["red_action_scale_schedule"] == [[0, 0.6], [1000, 1.0]]
