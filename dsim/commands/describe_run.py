"""dsim describe-run — display the existing MODEL.md for a run.

Does NOT regenerate (per design decision 2026-05-18).  If MODEL.md is
absent, exit 2 with a pointer to scripts.render_model_doc.
"""
from __future__ import annotations

from pathlib import Path

import typer

from core.run_listing import resolve_trial


def run(
    run_name: str = typer.Argument(..., help="Run name (e.g. ppo_hoop_blue_5)"),
    trial: str | None = typer.Option(None, "--trial",
                                     help="Specific trial; default: latest"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
) -> None:
    """Display the existing MODEL.md for a run."""
    # Prefer models/<run_name>/MODEL.md (promoted); fall back to runs/.
    model_md = models_dir / run_name / "MODEL.md"
    if model_md.exists():
        typer.echo(model_md.read_text())
        return

    try:
        trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)

    candidate = trial_dir / "MODEL.md"
    if not candidate.exists():
        typer.echo(
            f"error: no MODEL.md under {trial_dir}\n"
            f"  generate one: python -m scripts.render_model_doc --run-dir {trial_dir}",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(candidate.read_text())
