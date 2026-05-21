"""dsim resume — resume a training run from its latest checkpoint."""
from __future__ import annotations

import subprocess
import sys
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

    cmd = [
        sys.executable, "-m", "scripts.train",
        f"+experiment={parent_exp}",
        "init=resume",
        f"init.parent_run={run_name}",
        f"run_name={run_name}",
    ]
    typer.echo(f"$ {' '.join(cmd)}")
    raise typer.Exit(code=subprocess.call(cmd))
