"""dsim lineage — walk a run's pretrain ancestry."""
from __future__ import annotations

import typer

from core.lineage import walk_chain_local, walk_chain_wandb


def run(
    target: str = typer.Option(..., "--target",
                               help="Filesystem path or wandb:// URI"),
    local: bool = typer.Option(False, "--local", help="Walker A only"),
    both: bool = typer.Option(False, "--both",
                              help="Both walkers side-by-side"),
) -> None:
    """Walk a run's pretrain ancestry."""

    def render(chain) -> None:
        for i, n in enumerate(chain):
            ind = "  " * i
            sfx = "  (truncated)" if n.truncated else ""
            typer.echo(f"{ind}{n.name}  steps={n.final_steps}  "
                       f"chain={n.parent_chain_total}{sfx}")

    if local:
        render(walk_chain_local(target))
        return
    if both:
        typer.echo("--- local ---"); render(walk_chain_local(target))
        typer.echo("\n--- wandb ---")
        try:
            render(walk_chain_wandb(target))
        except Exception as e:
            typer.echo(f"(wandb failed: {e})")
        return
    if target.startswith(("wandb://", "wandb-artifact://")):
        try:
            render(walk_chain_wandb(target))
            return
        except Exception as e:
            typer.echo(f"(wandb failed: {e}; falling back to local)")
    render(walk_chain_local(target))
