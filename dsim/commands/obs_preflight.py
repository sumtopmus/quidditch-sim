"""dsim obs-preflight — check parent ↔ child obs-spec compat without loading weights.

Exit codes:
    0 — compatible (no surgery needed)
    1 — incompatible but surgery would resolve (warm_start required)
    2 — incompatible AND the diff suggests something else is wrong
       (e.g. unknown spec name, missing parent .hydra/)
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from core.obs_compat import preflight


def run(
    parent: str = typer.Option(..., "--parent",
                               help="Parent: filesystem path or wandb:// URI"),
    child_obs: str = typer.Option(..., "--child-obs",
                                  help="Child obs spec name (e.g. DUEL_V2_WORLD)"),
    child_n_stack: int = typer.Option(1, "--child-n-stack",
                                      help="Child frame-stack depth"),
) -> None:
    """Obs-spec compatibility preflight (no model load)."""
    try:
        report = preflight(parent, child_obs, child_n_stack)
    except (FileNotFoundError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)

    cons = Console()
    cons.print(
        f"parent: [bold]{report.parent_spec_name}[/] x n_stack={report.parent_n_stack}"
        f"  ->  child: [bold]{report.child_spec_name}[/] x n_stack={report.child_n_stack}"
    )

    table = Table(show_lines=False)
    table.add_column("Block")
    table.add_column("Dim", justify="right")
    table.add_column("Parent frame")
    table.add_column("Child frame")
    table.add_column("Status")
    glyph = {"matched": "ok",
             "frame_changed": "frame_changed",
             "removed": "removed",
             "added": "added"}
    for d in report.diff:
        table.add_row(d.block, str(d.dim),
                      d.parent_frame or "—",
                      d.child_frame or "—",
                      glyph.get(d.status, d.status))
    cons.print(table)

    if report.compatible:
        cons.print("[green]compatible[/] — init.mode=pretrain will load cleanly.")
        raise typer.Exit(code=0)
    cons.print("[yellow]surgery required[/] — set init.mode=warm_start.")
    raise typer.Exit(code=1)
