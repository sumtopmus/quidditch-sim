"""dsim sweep: create / agent / agents — thin wrappers over wandb CLI."""
from __future__ import annotations

import subprocess

import typer

app = typer.Typer(help="W&B sweep controller subcommands.", no_args_is_help=True)


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Sweep YAML name (sweeps/<name>.yaml)"),
) -> None:
    """Create a W&B sweep controller from sweeps/<name>.yaml."""
    cmd = ["wandb", "sweep", f"sweeps/{name}.yaml"]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("agent")
def agent(
    sweep_id: str = typer.Argument(...),
) -> None:
    """Run one W&B sweep agent."""
    cmd = ["wandb", "agent", sweep_id]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("agents")
def agents(
    sweep_id: str = typer.Argument(...),
    n: int = typer.Option(1, "--n", "-n", help="Number of parallel agents"),
) -> None:
    """Run N parallel W&B sweep agents."""
    procs = [
        subprocess.Popen(["wandb", "agent", sweep_id])
        for _ in range(n)
    ]
    rcs = [p.wait() for p in procs]
    raise typer.Exit(code=max(rcs) if rcs else 0)
