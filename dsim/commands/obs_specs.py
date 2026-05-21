"""dsim obs-specs — pretty-print every entry in SPEC_BY_NAME."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from envs.quidditch.obs_spec import SPEC_BY_NAME


def run() -> None:
    """Pretty-print the canonical obs spec catalog."""
    cons = Console()
    for name, spec in SPEC_BY_NAME.items():
        cons.print(f"\n[bold]{name}[/] ({spec.dim}-d)")
        t = Table(show_lines=False)
        t.add_column("Slot")
        t.add_column("Block")
        t.add_column("Dim", justify="right")
        t.add_column("Frame")
        t.add_column("Notes")
        off = 0
        for b in spec.blocks:
            t.add_row(f"{off}:{off + b.dim}", b.name, str(b.dim),
                      b.frame or "—", b.notes or "")
            off += b.dim
        cons.print(t)
