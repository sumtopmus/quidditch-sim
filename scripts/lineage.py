"""CLI wrapper around core.lineage walkers.

For new code, prefer:
    dsim lineage --target X [--local|--both]
This script remains for back-compat with the `python -m scripts.lineage`
invocation pattern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.lineage import LineageNode, walk_chain_local, walk_chain_wandb


def _render(chain: list[LineageNode]) -> str:
    """Aligned table; collapses consecutive resume segments into one row."""
    if not chain:
        return "(empty chain)"

    collapsed: list[dict] = []
    for n in chain:
        steps = n.parent_chain_total if n.parent_chain_total is not None else n.final_steps
        seg = {
            "name": n.name, "version": n.version or "",
            "init": n.init_mode or "",
            "obs": n.obs_spec or "",
            "steps": steps,
            "truncated": n.truncated,
        }
        if (collapsed and seg["name"] == collapsed[-1]["name"]
                and seg["init"] == "resume"):
            prev = collapsed[-1]
            prev["resume_count"] = prev.get("resume_count", 0) + 1
            prev["resume_steps"] = prev.get("resume_steps", 0) + (seg["steps"] or 0)
        else:
            collapsed.append(seg)

    rows: list[list[str]] = []
    for seg in collapsed:
        name = seg["name"]
        if seg.get("resume_count"):
            name = f"{name} (resumed x{seg['resume_count']}, +{seg['resume_steps']:,} steps)"
        steps = seg["steps"]
        steps_str = f"{steps:,}" if isinstance(steps, int) else "?"
        if seg["truncated"]:
            name = f"{name}  (truncated)"
        rows.append([name, seg["version"], seg["init"], seg["obs"], steps_str])

    headers = ["run", "version", "init", "obs", "steps"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", help="Filesystem path or wandb:// URI")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="walker A (local) only")
    g.add_argument("--both", action="store_true",
                   help="both walkers side-by-side")
    args = p.parse_args()

    if args.local:
        print(_render(walk_chain_local(args.target)))
        return 0

    if args.both:
        print("--- local ---")
        print(_render(walk_chain_local(args.target)))
        print("\n--- wandb ---")
        try:
            print(_render(walk_chain_wandb(args.target)))
        except Exception as e:
            print(f"(wandb walker failed: {e})", file=sys.stderr)
        return 0

    if args.target.startswith(("wandb://", "wandb-artifact://")):
        try:
            print(_render(walk_chain_wandb(args.target)))
            return 0
        except Exception as e:
            print(f"(wandb failed: {e}; falling back to local)", file=sys.stderr)
    print(_render(walk_chain_local(args.target)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
