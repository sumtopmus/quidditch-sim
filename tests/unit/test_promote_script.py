"""Unit tests for scripts/promote.py — file-copy semantics match the
former Make `promote` recipe (Makefile lines 103-122 pre-strip)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _make_trial_dir(tmp_path: Path, *, run: str, trial: str, with_info: bool = True,
                    with_config: bool = True) -> Path:
    trial_dir = tmp_path / "runs" / run / trial
    trial_dir.mkdir(parents=True)
    (trial_dir / "best_model.zip").write_bytes(b"fake-zip-data")
    if with_info:
        (trial_dir / "info.toml").write_text('name = "x"\n')
    if with_config:
        (trial_dir / "config_snapshot.toml").write_text('[training]\n')
    return trial_dir


def _run_promote(trial_dir: Path, models_root: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "promote.py"),
         "--trial", str(trial_dir), "--models-root", str(models_root)],
        capture_output=True, text=True,
    )


def test_promote_copies_three_files_with_flat_name(tmp_path: Path) -> None:
    trial = _make_trial_dir(tmp_path, run="ppo_hoop_red_1", trial="20260506_103058")
    models_root = tmp_path / "models"
    result = _run_promote(trial, models_root)
    assert result.returncode == 0, result.stderr
    dest = models_root / "ppo_hoop_red_1_20260506_103058"
    assert (dest / "best_model.zip").read_bytes() == b"fake-zip-data"
    assert (dest / "run_info.toml").exists()           # renamed from info.toml
    assert (dest / "config.toml").exists()             # renamed from config_snapshot.toml


def test_promote_missing_best_model_errors(tmp_path: Path) -> None:
    trial_dir = tmp_path / "runs" / "x" / "y"
    trial_dir.mkdir(parents=True)
    # no best_model.zip
    result = _run_promote(trial_dir, tmp_path / "models")
    assert result.returncode != 0
    assert "best_model.zip" in result.stderr or "best_model.zip" in result.stdout


def test_promote_missing_optional_files_succeeds(tmp_path: Path) -> None:
    trial = _make_trial_dir(tmp_path, run="r", trial="t",
                            with_info=False, with_config=False)
    models_root = tmp_path / "models"
    result = _run_promote(trial, models_root)
    assert result.returncode == 0, result.stderr
    dest = models_root / "r_t"
    assert (dest / "best_model.zip").exists()
    assert not (dest / "run_info.toml").exists()
    assert not (dest / "config.toml").exists()
