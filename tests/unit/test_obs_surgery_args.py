"""Argparse smoke tests for --obs-surgery on team + single-agent scripts."""
import sys
from importlib import import_module

import pytest


def _parse(module_name: str, argv: list[str]):
    mod = import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"{module_name}.py"] + argv
        return mod.parse_args()
    finally:
        sys.argv = old_argv


def test_team_pretrain_with_obs_surgery():
    args = _parse("scripts.train_team_ppo",
                  ["--learner", "blue_0", "--opponent", "beeline_red",
                   "--pretrain", "models/foo/best_model.zip", "--obs-surgery"])
    assert args.pretrain == "models/foo/best_model.zip"
    assert args.obs_surgery is True


def test_team_resume_with_obs_surgery_errors():
    with pytest.raises(SystemExit):
        _parse("scripts.train_team_ppo",
               ["--resume", "runs/x/y/checkpoints/ppo_1000_steps.zip",
                "--obs-surgery"])


def test_single_agent_pretrain_with_obs_surgery():
    args = _parse("scripts.train_ppo",
                  ["--pretrain", "models/foo/best_model.zip", "--obs-surgery"])
    assert args.pretrain == "models/foo/best_model.zip"
    assert args.obs_surgery is True


def test_obs_surgery_default_false():
    args = _parse("scripts.train_team_ppo",
                  ["--learner", "blue_0", "--opponent", "beeline_red",
                   "--pretrain", "models/foo/best_model.zip"])
    assert args.obs_surgery is False
