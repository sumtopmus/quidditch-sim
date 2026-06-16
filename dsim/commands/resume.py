"""dsim resume — resume a training run from its latest checkpoint."""
from __future__ import annotations

from pathlib import Path

import typer

from core.run_listing import resolve_checkpoint, resolve_trial


def run(
    run_name: str = typer.Argument(..., help="Run name"),
    trial: str = typer.Option(None, "--trial",
                              help="Specific trial; default: latest"),
    ckpt: str = typer.Option(None, "--ckpt",
                             help="Specific checkpoint; default: highest-step"),
    exp: str = typer.Option(None, "--exp",
                            help="conf/experiment override; default: same as parent's"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    """Resume a training run from its latest checkpoint."""
    trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    ckpt_path = resolve_checkpoint(trial_dir, ckpt=ckpt)

    # Read the parent trial's hydra-choices to determine which experiment
    # YAML to compose with on resume.
    from omegaconf import OmegaConf
    parent_exp = exp
    if parent_exp is None:
        hydra_yaml_path = trial_dir / ".hydra" / "hydra.yaml"
        if hydra_yaml_path.exists():
            hydra_meta = OmegaConf.load(hydra_yaml_path)
            choices = (hydra_meta.hydra.runtime.choices
                       if hasattr(hydra_meta, "hydra") else {})
            parent_exp = choices.get("experiment") if hasattr(choices, "get") else None
    if parent_exp is None:
        typer.echo(f"error: cannot infer experiment for {run_name}; pass --exp",
                   err=True)
        raise typer.Exit(code=2)

    typer.echo(
        f"error: `dsim resume` is not yet supported on the RLlib path.\n"
        f"  RLlib runs resume via ray.tune Tuner.restore, which is not wired\n"
        f"  into scripts/train.py yet (tracked as a follow-up). Start a fresh\n"
        f"  run with:  make train EXP={parent_exp}",
        err=True,
    )
    raise typer.Exit(code=2)
