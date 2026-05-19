"""dsim inventory — list promoted models with obs spec, parent, chain total."""
from __future__ import annotations

import json as json_module
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.inventory import inventory


def run(
    json_out: bool = typer.Option(False, "--json", help="Output as JSON instead of a table"),
    include_cache: bool = typer.Option(False, "--include-cache",
                                       help="Include models/.cache/ entries"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir",
                                    help="Override the models/ directory"),
) -> None:
    """List promoted models."""
    rows = inventory(models_dir=models_dir, include_cache=include_cache)
    if json_out:
        out = [
            {
                "name": r.name, "short_name": r.short_name,
                "obs_spec": r.obs_spec, "n_stack": r.n_stack,
                "parent": r.parent, "parent_chain_total": r.parent_chain_total,
                "final_steps": r.final_steps, "source": r.source,
                "path": str(r.path),
                "wandb_alias": r.wandb_alias, "wandb_version": r.wandb_version,
                "has_model_doc": r.has_model_doc,
            }
            for r in rows
        ]
        typer.echo(json_module.dumps(out, indent=2))
        return

    table = Table(title=f"Inventory ({len(rows)} models)", show_lines=False)
    table.add_column("Short name")
    table.add_column("Obs spec")
    table.add_column("n_stack", justify="right")
    table.add_column("Chain steps", justify="right")
    table.add_column("Source")
    table.add_column("W&B")
    table.add_column("Doc")
    for r in rows:
        wandb_str = f"{r.wandb_alias}:{r.wandb_version}" if r.wandb_alias else "—"
        table.add_row(
            r.short_name, r.obs_spec, str(r.n_stack),
            f"{r.parent_chain_total:,}", r.source,
            wandb_str, "yes" if r.has_model_doc else "—",
        )
    Console().print(table)
