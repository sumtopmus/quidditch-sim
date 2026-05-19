"""Smoke: eval_ppo Hydra entrypoint composes + runs 1 episode."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_ppo_hydra_runs_one_episode(tmp_path) -> None:
    # rand_start is the cheapest single-agent promoted model.
    model = "models/ppo_hoop_rand_start_20260505_174509/best_model"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_ppo",
         "+eval_ppo=default",
         f"eval_ppo.model_uri={model}",
         "eval_ppo.n_episodes=1",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "mean=" in result.stdout
