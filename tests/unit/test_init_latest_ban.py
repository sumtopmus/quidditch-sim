"""Schema rejects `wandb://...:latest` in init.parent — committed configs
must pin a stable alias (`:prod` or `:<run_name>`) so lineage doesn't shift
silently under a moving alias."""
import pytest

from config_schema import InitConfig


def test_latest_alias_rejected_when_mode_pretrain() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:latest")


def test_latest_alias_rejected_when_mode_warm_start() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(mode="warm_start", parent="wandb://ppo_hoop_blue_4:latest")


def test_latest_alias_ok_when_mode_scratch() -> None:
    # scratch ignores `parent` anyway; no validation needed.
    InitConfig(mode="scratch", parent="wandb://anything:latest")


def test_stable_alias_accepted() -> None:
    InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:prod")
    InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:v3")
    InitConfig(mode="pretrain", parent="models/ppo_hoop_blue_4/best_model")


def test_fully_qualified_uri_latest_also_rejected() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(
            mode="pretrain",
            parent="wandb-artifact://shurioque/drone-quidditch/ppo_hoop_blue_4:latest",
        )
