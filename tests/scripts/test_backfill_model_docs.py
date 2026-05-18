"""Tests for scripts/backfill_model_docs.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _make_fake_model_dir(root: Path, name: str) -> Path:
    """Build a minimal models/<name>/ for backfill to render against."""
    d = root / "models" / name
    (d / ".hydra").mkdir(parents=True)
    (d / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({
        "run_name": name,
        "description": "",
        "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1},
        "init": {"mode": "scratch"},
        "trainer": {"lr": 1e-3, "total_timesteps": 1000},
        "env": {"learner_id": "drone_0"},
        "curriculum": {"name": "fixed_start"},
        "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack", "terms": []},
    })))
    return d


def _run_backfill(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "WANDB_MODE": "disabled"}
    return subprocess.run(
        [sys.executable, "-m", "scripts.backfill_model_docs"] + args,
        cwd=repo_root, capture_output=True, text=True, env=env,
    )


def test_backfill_writes_model_doc(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    _make_fake_model_dir(tmp_path, "fake_model_a")
    result = _run_backfill(repo_root, ["--root", str(tmp_path / "models")])
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models" / "fake_model_a" / "MODEL.md").exists()


def test_backfill_is_idempotent_without_force(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    d = _make_fake_model_dir(tmp_path, "fake_model_a")
    md = d / "MODEL.md"
    md.write_text("ORIGINAL")
    result = _run_backfill(repo_root, ["--root", str(tmp_path / "models")])
    assert result.returncode == 0
    # Existing file untouched
    assert md.read_text() == "ORIGINAL"


def test_backfill_force_overwrites(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    d = _make_fake_model_dir(tmp_path, "fake_model_a")
    md = d / "MODEL.md"
    md.write_text("ORIGINAL")
    result = _run_backfill(repo_root, ["--root", str(tmp_path / "models"), "--force"])
    assert result.returncode == 0
    assert md.read_text() != "ORIGINAL"
    assert "fake_model_a" in md.read_text()
