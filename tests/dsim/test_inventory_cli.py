"""Smoke: dsim inventory renders + --json round-trips."""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def test_inventory_table_renders(tmp_path: Path) -> None:
    d = tmp_path / "models" / "ppo_hoop_blue_4_20260511_202612"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "ppo_hoop_blue_4",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
        "init": {"mode": "scratch"},
    }), d / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create({"final_steps": 1, "parent_chain_total": 1}),
                   d / ".hydra" / "meta.yaml")

    runner = CliRunner()
    result = runner.invoke(app, ["inventory", "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0, result.output
    assert "blue_4" in result.output
    assert "DUEL_V2_WORLD" in result.output


def test_inventory_json_round_trips(tmp_path: Path) -> None:
    d = tmp_path / "models" / "ppo_hoop_blue_4_20260511_202612"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "ppo_hoop_blue_4",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
    }), d / ".hydra" / "config.yaml")

    runner = CliRunner()
    result = runner.invoke(app, ["inventory", "--json",
                                 "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 1
    assert rows[0]["short_name"] == "blue_4"
