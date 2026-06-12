"""Step-5c: run_listing recognizes RLlib directory checkpoints."""
from __future__ import annotations

from pathlib import Path

from core.run_listing import list_runs, resolve_checkpoint


def _rllib_run(runs: Path, run_name: str, ts: str, ordinal: str) -> Path:
    ckpt = (runs / run_name / ts / "tune" / "trial_x"
            / f"checkpoint_{ordinal}")
    (ckpt / "learner_group" / "learner" / "rl_module" / "main_blue").mkdir(parents=True)
    return ckpt.resolve()


def test_list_runs_finds_rllib_checkpoint_dir(tmp_path):
    runs = tmp_path / "runs"
    latest = _rllib_run(runs, "rllib_league", "20260611_130631", "000030")
    _rllib_run(runs, "rllib_league", "20260611_130631", "000010")
    entries = list_runs(runs_dir=runs)
    assert len(entries) == 1
    assert entries[0].latest_checkpoint == latest


def test_resolve_checkpoint_returns_dir_when_no_zip(tmp_path):
    runs = tmp_path / "runs"
    latest = _rllib_run(runs, "rllib_league", "20260611_130631", "000030")
    trial_dir = runs / "rllib_league" / "20260611_130631"
    assert resolve_checkpoint(trial_dir) == latest
