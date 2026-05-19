"""Smoke + exit-code contract for dsim obs-preflight."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def _make_parent(tmp: Path, obs: str, n_stack: int) -> Path:
    d = tmp / "parent"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "p", "obs": {"name": obs, "n_stack": n_stack},
    }), d / ".hydra" / "config.yaml")
    return d


def test_compatible_exits_zero(tmp_path: Path) -> None:
    p = _make_parent(tmp_path, "DUEL_V2_WORLD", 3)
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(p),
        "--child-obs", "DUEL_V2_WORLD", "--child-n-stack", "3",
    ])
    assert result.exit_code == 0
    assert "compatible" in result.output


def test_surgery_required_exits_one(tmp_path: Path) -> None:
    p = _make_parent(tmp_path, "DUEL_V1_BODY", 1)
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(p),
        "--child-obs", "DUEL_V2_WORLD", "--child-n-stack", "3",
    ])
    assert result.exit_code == 1
    assert "surgery required" in result.output


def test_missing_parent_exits_two(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(tmp_path / "nope"),
        "--child-obs", "DUEL_V2_WORLD",
    ])
    assert result.exit_code == 2
