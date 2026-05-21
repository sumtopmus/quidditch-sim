"""Behavior contract for core.run_listing.

list_runs(runs_dir=Path("runs")) -> list[RunEntry]
  One entry per `runs/<run_name>/`, with latest_trial = the lex-max
  YYYYMMDD_HHMMSS subdir, plus latest_checkpoint = the highest-step .zip
  inside <latest_trial>/checkpoints/.

resolve_trial(run_name, *, trial=None, runs_dir=Path("runs")) -> Path
  Helper: returns runs/<run_name>/<trial>/ for a given trial id, or the
  latest trial if trial is None.

resolve_checkpoint(trial_dir, *, ckpt=None) -> Path
  Helper: returns the requested checkpoint zip, or the highest-step zip
  if ckpt is None.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _seed_run(runs_dir: Path, run: str, trials: list[str], *,
              checkpoints_per_trial: dict[str, list[int]] | None = None) -> None:
    checkpoints_per_trial = checkpoints_per_trial or {}
    for trial in trials:
        td = runs_dir / run / trial
        td.mkdir(parents=True)
        (td / ".hydra").mkdir()
        (td / ".hydra" / "config.yaml").write_text(f"run_name: {run}\n")
        for step in checkpoints_per_trial.get(trial, []):
            cks = td / "checkpoints"
            cks.mkdir(exist_ok=True)
            (cks / f"ppo_hoop_{step}_steps.zip").write_bytes(b"\x50\x4b\x03\x04stub")


def test_list_runs_one_entry_per_run_dir(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "ppo_hoop_blue_5", ["20260514_120000", "20260515_010000"])
    _seed_run(tmp_path, "ppo_hoop_red_2", ["20260516_020000"])
    rows = list_runs(runs_dir=tmp_path)
    names = sorted([r.run_name for r in rows])
    assert names == ["ppo_hoop_blue_5", "ppo_hoop_red_2"]


def test_list_runs_latest_trial_is_lex_max(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000",
                                   "20260514_120000"])
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_trial.name == "20260515_080000"


def test_list_runs_latest_checkpoint_is_highest_step(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"],
              checkpoints_per_trial={"20260514_120000": [50_000, 200_000, 100_000]})
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_checkpoint is not None
    assert "200000" in row.latest_checkpoint.name


def test_list_runs_no_checkpoints_returns_none_checkpoint(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"])
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_checkpoint is None


def test_list_runs_filter_by_run_name(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"])
    _seed_run(tmp_path, "red_2", ["20260516_120000"])
    rows = list_runs(runs_dir=tmp_path, run_filter="blue_5")
    assert [r.run_name for r in rows] == ["blue_5"]


def test_resolve_trial_picks_latest_when_none(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000"])
    p = resolve_trial("blue_5", trial=None, runs_dir=tmp_path)
    assert p.name == "20260515_080000"


def test_resolve_trial_with_explicit_id(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000"])
    p = resolve_trial("blue_5", trial="20260513_120000", runs_dir=tmp_path)
    assert p.name == "20260513_120000"


def test_resolve_trial_unknown_raises(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    with pytest.raises(FileNotFoundError):
        resolve_trial("nope", runs_dir=tmp_path)


def test_resolve_checkpoint_highest_step(tmp_path: Path) -> None:
    from core.run_listing import resolve_checkpoint
    trial = tmp_path / "trial"
    cks = trial / "checkpoints"
    cks.mkdir(parents=True)
    for s in (50_000, 200_000, 100_000):
        (cks / f"ppo_hoop_{s}_steps.zip").write_bytes(b"stub")
    p = resolve_checkpoint(trial)
    assert "200000" in p.name


def test_resolve_checkpoint_explicit(tmp_path: Path) -> None:
    from core.run_listing import resolve_checkpoint
    trial = tmp_path / "trial"
    cks = trial / "checkpoints"
    cks.mkdir(parents=True)
    (cks / "ppo_hoop_50000_steps.zip").write_bytes(b"stub")
    p = resolve_checkpoint(trial, ckpt="ppo_hoop_50000_steps")
    assert p.name == "ppo_hoop_50000_steps.zip"
