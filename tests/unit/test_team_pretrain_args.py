"""Argparse contract for train_team_ppo.py: --pretrain is captured into
args.pretrain and is mutually exclusive with --resume and --warm-start."""
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
    from scripts.train_team_ppo import parse_args
    return parse_args()


def test_pretrain_is_captured() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--pretrain", "models/blue_v0/best_model"):
        args = _parse()
    assert args.pretrain == "models/blue_v0/best_model"
    assert args.resume is None
    assert args.warm_start == ""


def test_pretrain_default_is_none() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red"):
        args = _parse()
    assert args.pretrain is None


def test_pretrain_and_resume_are_mutually_exclusive() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--pretrain", "models/p/best_model",
               "--resume",   "runs/x/y/checkpoints/ppo_1000_steps.zip"):
        with pytest.raises(SystemExit):
            _parse()


def test_pretrain_and_warm_start_are_mutually_exclusive() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--pretrain",   "models/p/best_model",
               "--warm-start", "models/old/best_model.zip"):
        with pytest.raises(SystemExit):
            _parse()
