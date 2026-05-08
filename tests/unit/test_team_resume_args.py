"""Argparse contract for train_team_ppo.py: --resume and --warm-start are
mutually exclusive; --resume is captured into args.resume."""
from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest


@contextmanager
def _argv(*args: str):
    saved = sys.argv
    sys.argv = ["train_team_ppo.py", *args]
    try:
        yield
    finally:
        sys.argv = saved


def _parse():
    # Import inside the test so each call re-evaluates argparse defaults.
    from scripts.train_team_ppo import parse_args
    return parse_args()


def test_resume_is_captured() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--resume", "runs/x/y/checkpoints/ppo_1000_steps.zip"):
        args = _parse()
    assert args.resume == "runs/x/y/checkpoints/ppo_1000_steps.zip"
    assert args.warm_start == ""


def test_warm_start_and_resume_are_mutually_exclusive() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--resume", "ckpt.zip", "--warm-start", "model.zip"):
        with pytest.raises(SystemExit):
            _parse()


def test_resume_default_is_none() -> None:
    with _argv("--learner", "red_0", "--opponent", "beeline_blue"):
        args = _parse()
    assert args.resume is None


def test_missing_learner_without_resume_errors_in_main(monkeypatch, capsys) -> None:
    """Without --resume, --learner is still required."""
    from scripts import train_team_ppo

    with _argv("--opponent", "beeline_blue"):  # no --learner, no --resume
        with pytest.raises(SystemExit):
            train_team_ppo.main()
