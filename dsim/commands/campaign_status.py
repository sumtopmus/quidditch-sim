"""dsim campaign-status — read-only W&B telemetry + kill-rule verdict as JSON.

Used by the run-campaign controller skill as the deterministic, cheap
monitor tick.  Prints a JSON object: latest metrics, kill bool, reasons,
and the run summary.  Never mutates anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from core.campaign.killrules import evaluate
from core.campaign.telemetry import read_telemetry


def run(
    wandb_run: str = typer.Argument(
        ..., help="W&B run path (entity/project/run_id)"),
    rules: Path = typer.Option(
        None, "--rules", help="JSON file of kill-rules (see campaign spec)"),
) -> None:
    """Print latest telemetry + kill-rule verdict for a W&B run as JSON."""
    tel = read_telemetry(wandb_run)
    rule_list: list[dict] = []
    baselines: dict[str, list] = {}
    if rules is not None:
        spec = json.loads(rules.read_text())
        rule_list = spec.get("rules", [])
        for b in spec.get("baseline_runs", []):
            baselines[b] = read_telemetry(b).rows
    verdict = evaluate(tel.rows, rule_list, baselines=baselines)
    typer.echo(json.dumps({
        "wandb_run": wandb_run,
        "kill": verdict.kill,
        "reasons": [{"rule": r.rule, "detail": r.detail} for r in verdict.reasons],
        "latest": verdict.latest,
        "summary": tel.summary,
    }, indent=2, default=str))
