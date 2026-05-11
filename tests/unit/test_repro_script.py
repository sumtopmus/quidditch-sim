"""Unit tests for scripts/repro.py — config restoration semantics."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(model_dir: Path, config_target: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "repro.py"),
         "--model-dir", str(model_dir),
         "--config-target", str(config_target)],
        capture_output=True, text=True,
    )


def test_repro_copies_config_to_target(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "red_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text('[training]\nlr = 5e-5\n')
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode == 0, result.stderr
    assert target.read_text() == '[training]\nlr = 5e-5\n'


def test_repro_warns_on_legacy_env_toml(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "old_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text('[training]\n')
    (model_dir / "env.toml").write_text('[env]\n')
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode == 0
    assert "env.toml" in (result.stdout + result.stderr)


def test_repro_missing_config_errors(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "broken"
    model_dir.mkdir(parents=True)
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode != 0
    assert "config.toml" in (result.stderr + result.stdout)
