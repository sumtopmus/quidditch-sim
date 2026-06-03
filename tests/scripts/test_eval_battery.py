"""Smoke: eval_battery composes quick.yaml + runs 1-episode scenarios."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_battery_quick_produces_report(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_battery",
         "+eval_battery=quick",
         "eval_battery.candidate=models/ppo_hoop_blue_4_20260511_202612/best_model",
         f"hydra.run.dir={tmp_path}/run",
         "eval_battery.scenarios.0.n_episodes=1",
         "eval_battery.scenarios.1.n_episodes=1",
         "eval_battery.scenarios.2.n_episodes=1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "eval_report" / "summary.md").exists()
    assert (tmp_path / "run" / "eval_report" / "results.csv").exists()
    assert (tmp_path / "run" / "eval_report" / "per_episode.jsonl").exists()


def test_eval_battery_composes_default() -> None:
    """Verify Hydra can compose the battery configs without invoking the engine."""
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    cfg_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=["+eval_battery=default",
                       "eval_battery.candidate=models/x/best_model"],
        )
        assert cfg.eval_battery.candidate == "models/x/best_model"
        assert len(cfg.eval_battery.scenarios) == 5
        assert cfg.eval_battery.scenarios[0].opponent == "beeline_red"


def test_eval_battery_composes_quick() -> None:
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    cfg_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=["+eval_battery=quick",
                       "eval_battery.candidate=models/x/best_model"],
        )
        assert len(cfg.eval_battery.scenarios) == 3
