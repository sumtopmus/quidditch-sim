"""Smoke test: dsim CLI mounts and --help renders without raising.

If a subcommand's import fails at module level (e.g. missing import,
typer decorator typo), this catches it before deployment.
"""
from __future__ import annotations

from typer.testing import CliRunner

from dsim.cli import app


def test_dsim_help_renders_without_error() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_dsim_module_invocation_works() -> None:
    """`python -m dsim --help` must also work, not just the installed entry point."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "dsim", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
