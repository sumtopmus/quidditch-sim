"""Smoke: eval_team Hydra entrypoint composes + runs 1 episode.

Uses the `scripted:beeline_blue` learner sentinel — eval_core supports it
so we don't need to load real weights for the smoke.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_team_hydra_runs_one_episode(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_team",
         "+eval_team=default",
         "+learner=blue", "learner.uri=scripted:beeline_blue",
         "opponent=beeline_red",
         "eval.n_episodes=1",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "win_rate:" in result.stdout
