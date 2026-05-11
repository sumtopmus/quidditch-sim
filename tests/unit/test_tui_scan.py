"""Unit tests for tui/state/scan.py — promoted models, runs, trials, checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tui.state import scan


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    # models/
    (tmp_path / "models" / "red_v1").mkdir(parents=True)
    (tmp_path / "models" / "red_v1" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "models" / "blue_v1").mkdir(parents=True)
    (tmp_path / "models" / "blue_v1" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "models" / "no_best").mkdir(parents=True)  # no best_model.zip — should be skipped

    # runs/
    (tmp_path / "runs" / "ppo_hoop" / "20260101_000000").mkdir(parents=True)
    (tmp_path / "runs" / "ppo_hoop" / "20260202_000000").mkdir(parents=True)
    (tmp_path / "runs" / "ppo_hoop" / "20260202_000000" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "runs" / "ppo_hoop_team" / "20260303_000000").mkdir(parents=True)

    # checkpoints in latest ppo_hoop trial — mix legacy (ppo_hoop_<step>) and
    # current (ppo_<step>) prefixes; both must be picked up by the regex.
    ckpt_dir = tmp_path / "runs" / "ppo_hoop" / "20260202_000000" / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "ppo_hoop_1000_steps.zip").write_bytes(b"x")   # legacy prefix
    (ckpt_dir / "ppo_5000_steps.zip").write_bytes(b"x")        # current prefix
    (ckpt_dir / "ppo_10000_steps.zip").write_bytes(b"x")       # current prefix

    # live progress file in one trial
    progress = {"schema_version": 1, "step": 5000}
    (tmp_path / "runs" / "ppo_hoop_team" / "20260303_000000" / "tui_progress.json"
     ).write_text(json.dumps(progress))

    return tmp_path


def test_promoted_models_lists_dirs_with_best_model(fixture_tree: Path) -> None:
    models = scan.promoted_models(fixture_tree / "models")
    names = [m.name for m in models]
    assert "red_v1" in names
    assert "blue_v1" in names
    assert "no_best" not in names


def test_run_names_sorted(fixture_tree: Path) -> None:
    names = scan.run_names(fixture_tree / "runs")
    assert names == ["ppo_hoop", "ppo_hoop_team"]


def test_trials_in_run_sorted_desc(fixture_tree: Path) -> None:
    trials = scan.trials_in_run("ppo_hoop", fixture_tree / "runs")
    names = [t.name for t in trials]
    assert names == ["20260202_000000", "20260101_000000"]
    assert trials[0].has_best_model is True
    assert trials[1].has_best_model is False


def test_checkpoints_sorted_desc_by_step(fixture_tree: Path) -> None:
    """The fixture mixes legacy `ppo_hoop_<step>_steps.zip` with current
    `ppo_<step>_steps.zip` filenames; both must be picked up and sorted
    together by step number."""
    ckpts = scan.checkpoints(
        "ppo_hoop", "20260202_000000", fixture_tree / "runs"
    )
    steps = [c.step for c in ckpts]
    assert steps == [10000, 5000, 1000]
    # The 1000-step file uses the legacy prefix — confirm it appears in the
    # returned path so consumers (the resume picker) can pass it as-is to
    # `--resume`, preserving filename-on-disk regardless of prefix.
    legacy = next(c for c in ckpts if c.step == 1000)
    assert legacy.path.name == "ppo_hoop_1000_steps.zip"


def test_live_trial_detection_uses_progress_file_mtime(fixture_tree: Path) -> None:
    progress = fixture_tree / "runs" / "ppo_hoop_team" / "20260303_000000" / "tui_progress.json"
    # File was just written by the fixture → mtime is now.
    trials = scan.trials_in_run("ppo_hoop_team", fixture_tree / "runs")
    assert trials[0].is_live is True

    # Backdate mtime by 60s — outside the 30s window.
    old = time.time() - 60
    import os
    os.utime(progress, (old, old))
    trials = scan.trials_in_run("ppo_hoop_team", fixture_tree / "runs")
    assert trials[0].is_live is False


def test_latest_trial_picks_newest_across_runs(fixture_tree: Path) -> None:
    latest = scan.latest_trial(fixture_tree / "runs")
    assert latest is not None
    assert latest.name == "20260303_000000"  # newest timestamp overall
