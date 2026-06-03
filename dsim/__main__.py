"""`python -m dsim` entry point — delegates to the Typer app in dsim.cli."""
from dsim.cli import app

if __name__ == "__main__":
    app()
