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
