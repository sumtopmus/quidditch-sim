"""Top-level Typer app for dsim.

Subcommands are registered here as direct commands on `app`.  Each
subcommand module exposes a `run` callable; this file just wires the
mapping from CLI name → `run` function.  Keeps subcommand modules
simple (no per-module Typer instance).
"""
from __future__ import annotations

import typer

from dsim.commands import describe_run as _describe_run_cmd
from dsim.commands import inventory as _inventory_cmd
from dsim.commands import obs_preflight as _obs_preflight_cmd
from dsim.commands import obs_specs as _obs_specs_cmd

app = typer.Typer(
    name="dsim",
    help="Drone-sim inspection and dispatch CLI.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


app.command(name="inventory")(_inventory_cmd.run)
app.command(name="obs-preflight")(_obs_preflight_cmd.run)
app.command(name="obs-specs")(_obs_specs_cmd.run)
app.command(name="describe-run")(_describe_run_cmd.run)


def main() -> None:
    """Entry point for `dsim` script (defined in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
