"""dsim obs-specs — walk conf/obs/*.yaml + pretty-print each spec."""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from envs.quidditch.obs_spec import build_spec_from_block_names


def run() -> None:
    """Pretty-print the canonical obs spec catalog from conf/obs/*.yaml."""
    cons = Console()
    repo_root = Path(__file__).resolve().parents[2]
    obs_dir = repo_root / "conf" / "obs"
    if not obs_dir.exists():
        cons.print(f"[red]no conf/obs/ under {repo_root}[/]")
        return

    for yaml_path in sorted(obs_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text()) or {}
        name = str(data.get("name", yaml_path.stem))
        blocks = data.get("blocks") or []
        n_stack = data.get("n_stack", 1)
        if not blocks:
            cons.print(f"\n[bold]{name}[/] — [yellow]no blocks: field in {yaml_path.name}[/]")
            continue
        spec = build_spec_from_block_names(list(blocks))
        cons.print(f"\n[bold]{name}[/] ({spec.dim}-d, n_stack={n_stack})  "
                   f"[dim]conf/obs/{yaml_path.name}[/]")
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
