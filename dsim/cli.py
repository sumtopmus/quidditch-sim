"""Top-level Typer app for dsim.

Subcommands are registered here as direct commands on `app`.  Each
subcommand module exposes a `run` callable; this file just wires the
mapping from CLI name → `run` function.  `sweep` is the exception: it's
a multi-command sub-app (create/agent/agents) and is mounted via
add_typer.
"""
from __future__ import annotations

import typer

from dsim.commands import describe_run as _describe_run_cmd
from dsim.commands import inventory as _inventory_cmd
from dsim.commands import lineage as _lineage_cmd
from dsim.commands import list_runs as _list_runs_cmd
from dsim.commands import obs_preflight as _obs_preflight_cmd
from dsim.commands import obs_specs as _obs_specs_cmd
from dsim.commands import promote as _promote_cmd
from dsim.commands import resume as _resume_cmd
from dsim.commands import sweep as _sweep_cmd

app = typer.Typer(
    name="dsim",
    help="Drone-sim inspection and dispatch CLI.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


# Single-command modules: register via app.command(name=...).
app.command(name="inventory")(_inventory_cmd.run)
app.command(name="obs-preflight")(_obs_preflight_cmd.run)
app.command(name="obs-specs")(_obs_specs_cmd.run)
app.command(name="describe-run")(_describe_run_cmd.run)
app.command(name="lineage")(_lineage_cmd.run)
app.command(name="list-runs")(_list_runs_cmd.run)
app.command(name="resume")(_resume_cmd.run)
app.command(name="promote")(_promote_cmd.run)

# Multi-command sub-app: sweep create / agent / agents.
app.add_typer(_sweep_cmd.app, name="sweep")


def main() -> None:
    """Entry point for `dsim` script (defined in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
