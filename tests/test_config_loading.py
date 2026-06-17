"""Hydra composition smoke tests."""
from __future__ import annotations

import pytest

from tests.conftest import hydra_compose


def test_default_compose_succeeds():
    with hydra_compose() as cfg:
        assert cfg.run_name == "_adhoc"
        assert cfg.trainer.lr > 0
        assert cfg.env.team_env_params.tag_radius == 0.3


@pytest.mark.parametrize("name", [
    "rllib_red_skeleton", "rllib_selfplay",
    "rllib_league", "rllib_league_step5",
])
def test_experiment_composes(name: str):
    with hydra_compose(experiment=name) as cfg:
        assert cfg.run_name
        assert cfg.algo.total_timesteps > 0


def test_init_groups_compose():
    # scratch is the only init mode after the SB3 retirement (Step 6).
    with hydra_compose(overrides=["init=scratch"]) as cfg:
        assert cfg.init.mode == "scratch"


def test_rllib_league_step5_experiment_composes():
    from tests.conftest import hydra_compose
    with hydra_compose(overrides=["+experiment=rllib_league_step5"]) as cfg:
        assert cfg.curriculum.dense_scale_schedule == [[0, 1.0], [3_000_000, 0.0]]
        assert cfg.curriculum.red_action_scale == 0.6
        assert cfg.league.snapshot_threshold_red == 0.5
        assert cfg.league.eval_enabled is True
        assert cfg.reward._target_.endswith("RewardStack")
