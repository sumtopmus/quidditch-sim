"""Hydra composition smoke tests."""
from __future__ import annotations

import pytest

from tests.conftest import hydra_compose


def test_default_compose_succeeds():
    with hydra_compose() as cfg:
        assert cfg.run_name == "_adhoc"
        assert cfg.trainer.lr > 0
        assert cfg.env._target_.endswith("Factory")


@pytest.mark.parametrize("name", [
    "canary_single", "canary_team",
    "red_v1", "blue_v4", "blue_v5",
])
def test_experiment_composes(name: str):
    with hydra_compose(experiment=name) as cfg:
        assert cfg.run_name
        assert cfg.trainer.total_timesteps > 0


@pytest.mark.parametrize("init_choice", ["scratch", "pretrain", "resume", "warm_start"])
def test_init_groups_compose(init_choice: str):
    overrides: list[str] = []
    if init_choice in ("pretrain", "warm_start"):
        overrides = ["+init.parent=x"]
    elif init_choice == "resume":
        overrides = ["+init.parent_run=y"]
    with hydra_compose(overrides=[f"init={init_choice}", *overrides]) as cfg:
        assert cfg.init.mode == init_choice


def test_rllib_league_step5_experiment_composes():
    from tests.conftest import hydra_compose
    with hydra_compose(overrides=["+experiment=rllib_league_step5"]) as cfg:
        assert cfg.curriculum.dense_scale_schedule == [[0, 1.0], [3_000_000, 0.0]]
        assert cfg.curriculum.red_action_scale == 0.6
        assert cfg.league.snapshot_threshold_red == 0.5
        assert cfg.league.eval_enabled is True
        assert cfg.reward._target_.endswith("RewardStack")
