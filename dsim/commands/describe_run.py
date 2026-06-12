"""dsim describe-run — display the existing MODEL.md for a run.

Does NOT regenerate (per design decision 2026-05-18).  If MODEL.md is
absent, exit 2 with a pointer to scripts.render_model_doc.
"""
from __future__ import annotations

import json as _json
from pathlib import Path

import typer

from core.run_listing import resolve_trial


def _print_rllib_provenance(model_dir: Path) -> bool:
    """Surface an RLlib promoted model's checkpoint format + module ids from its
    committed _wandb_metadata.json. Returns True iff it printed (an RLlib model);
    SB3 / unpromoted dirs are a no-op so existing describe-run output is unchanged.
    """
    meta_path = model_dir / "_wandb_metadata.json"
    if not meta_path.exists():
        return False
    try:
        meta = _json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    if meta.get("checkpoint_format") != "rllib":
        return False
    typer.echo("checkpoint format: rllib")
    mods = meta.get("module_ids") or []
    if mods:
        typer.echo(f"modules:           {', '.join(mods)}")
    return True


def run(
    run_name: str = typer.Argument(..., help="Run name (e.g. ppo_hoop_blue_5)"),
    trial: str | None = typer.Option(None, "--trial",
                                     help="Specific trial; default: latest"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
) -> None:
    """Display the existing MODEL.md for a run."""
    # A promoted model dir may be passed directly (path) or by name; prefer the
    # direct path when it carries a committed _wandb_metadata.json.
    direct = Path(run_name)
    if direct.is_dir() and (direct / "_wandb_metadata.json").exists():
        model_dir = direct
    else:
        model_dir = models_dir / run_name

    # RLlib provenance (Step 5c): print the checkpoint format + module ids when
    # this is a promoted RLlib model, alongside / ahead of MODEL.md.
    printed_provenance = _print_rllib_provenance(model_dir)

    # Prefer models/<run_name>/MODEL.md (promoted); fall back to runs/.
    model_md = model_dir / "MODEL.md"
    if model_md.exists():
        typer.echo(model_md.read_text())
        return
    if printed_provenance:
        # An RLlib model without a MODEL.md is still fully described by its
        # provenance — success, not the "no MODEL.md" error.
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
