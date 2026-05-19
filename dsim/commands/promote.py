"""dsim promote — promote a run's best model to canonical/vendored."""
from __future__ import annotations

from pathlib import Path

import typer

from core.promote import promote_run
from core.run_listing import resolve_trial


def run(
    run_name: str = typer.Argument(..., help="Run name"),
    trial: str = typer.Option(None, "--trial"),
    alias: str = typer.Option("prod", "--alias"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
) -> None:
    """Promote a run's best model to canonical/vendored."""
    trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    result = promote_run(trial_dir, alias=alias, models_root=models_dir)
    typer.echo(f"promoted {result.run_name} ({result.wandb_version}, alias={result.wandb_alias})")
    typer.echo(f"  copied {len(result.copied_files)} files into {result.target_dir}")
    typer.echo("git add models/ && git commit -m 'model: promote ...' to vendor.")
