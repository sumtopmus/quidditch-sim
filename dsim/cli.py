"""Top-level Typer app for dsim.

Subcommands are registered here as they're implemented (see dsim/commands/).
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="dsim",
    help="Drone-sim inspection and dispatch CLI.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root() -> None:
    """Root callback — Typer needs at least one command/callback to mount."""


# Subcommands are added in later tasks.  Each subcommand module exposes a
# Typer sub-app object and is wired here.


def main() -> None:
    """Entry point for `dsim` script (defined in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
