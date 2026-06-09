"""dsim campaign-status — JSON verdict; telemetry monkeypatched (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from core.campaign.telemetry import CampaignTelemetry
from dsim.cli import app
from dsim.commands import campaign_status as cmd


def _patch_telemetry(monkeypatch, rows, summary=None):
    def fake(run_path, **kwargs):
        return CampaignTelemetry(rows=rows, summary=summary or {})
    monkeypatch.setattr(cmd, "read_telemetry", fake)


def test_status_no_rules_reports_no_kill(monkeypatch):
    _patch_telemetry(monkeypatch,
                     [{"_step": 100, "_runtime": 1.0, "eval/success_rate": 0.5}])
    result = CliRunner().invoke(app, ["campaign-status", "ent/proj/x"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["kill"] is False
    assert out["latest"]["eval/success_rate"] == 0.5


def test_status_with_rules_reports_kill(monkeypatch, tmp_path: Path):
    _patch_telemetry(monkeypatch,
                     [{"_step": 3_000_000, "_runtime": 9.0,
                       "eval/success_rate": 0.04}])
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [
        {"type": "floor_at_step", "metric": "eval/success_rate",
         "min": 0.10, "at_step": 3_000_000}]}))
    result = CliRunner().invoke(
        app, ["campaign-status", "ent/proj/x", "--rules", str(rules)])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["kill"] is True
    assert out["reasons"][0]["rule"] == "floor_at_step"
