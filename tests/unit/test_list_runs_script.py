"""Unit tests for scripts/list_runs.py — output enumerates runs and models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(runs_root: Path, models_root: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "list_runs.py"),
         "--runs-root", str(runs_root), "--models-root", str(models_root)],
        capture_output=True, text=True,
    )


def test_list_runs_groups_by_config(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "ppo_hoop" / "20260101_000000").mkdir(parents=True)
    (runs / "ppo_hoop" / "20260202_000000").mkdir(parents=True)
    (runs / "ppo_hoop_team" / "20260303_000000").mkdir(parents=True)
    models = tmp_path / "models"
    (models / "red_v1").mkdir(parents=True)
    (models / "red_v1" / "best_model.zip").write_bytes(b"x")

    result = _run(runs, models)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "ppo_hoop/" in out
    assert "20260101_000000" in out
    assert "ppo_hoop_team/" in out
    assert "red_v1" in out
    assert "best_model.zip" in out


def test_list_runs_no_runs_no_models(tmp_path: Path) -> None:
    result = _run(tmp_path / "runs", tmp_path / "models")
    assert result.returncode == 0
    out = result.stdout
    assert "(none)" in out  # at least one "(none)" for missing runs and missing models
