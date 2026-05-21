"""dsim list-runs — enumerate runs/ with latest trial + checkpoint."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.run_listing import list_runs


def run(
    run_filter: str = typer.Option(None, "--run",
                                   help="Filter by substring of run name"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    """Enumerate runs/ with latest trial + checkpoint."""
    rows = list_runs(runs_dir=runs_dir, run_filter=run_filter)
    table = Table(title=f"Runs ({len(rows)})")
    table.add_column("Run name")
    table.add_column("Latest trial")
    table.add_column("Latest checkpoint")
    for r in rows:
        ck = r.latest_checkpoint.name if r.latest_checkpoint else "—"
        table.add_row(r.run_name, r.latest_trial.name, ck)
    Console().print(table)
