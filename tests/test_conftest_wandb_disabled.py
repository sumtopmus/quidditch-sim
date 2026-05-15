"""Confirm conftest disables wandb so the suite never touches the network."""
import os


def test_wandb_mode_is_disabled() -> None:
    assert os.environ.get("WANDB_MODE") == "disabled"
