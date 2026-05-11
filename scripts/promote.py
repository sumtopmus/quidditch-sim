"""Promote a trial's best_model.zip (+ optional metadata) into models/<flat-name>/.

Port of the former `make promote` recipe. Callable from the TUI's Manage →
promote action and from the shell for one-offs.

Usage:
    python scripts/promote.py --trial runs/<run>/<trial>  [--models-root models]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def promote(trial_dir: Path, models_root: Path) -> Path:
    if not trial_dir.is_dir():
        raise SystemExit(f"ERROR: trial directory not found: {trial_dir}")
    src_best = trial_dir / "best_model.zip"
    if not src_best.is_file():
        raise SystemExit(
            f"ERROR: {src_best} not found — has training produced a best_model.zip yet?"
        )

    # Flat name: strip the "runs/" prefix if present, then replace path separators
    # with underscores so "ppo_hoop_red_1/20260506_103058" → "ppo_hoop_red_1_20260506_103058".
    parts = trial_dir.parts
    try:
        runs_ix = parts.index("runs")
        rel_parts = parts[runs_ix + 1:]
    except ValueError:
        rel_parts = parts[-2:]  # fall back to last two segments
    flat_name = "_".join(rel_parts)

    dest = models_root / flat_name
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_best, dest / "best_model.zip")
    info_src = trial_dir / "info.toml"
    if info_src.is_file():
        shutil.copy2(info_src, dest / "run_info.toml")
    config_src = trial_dir / "config_snapshot.toml"
    if config_src.is_file():
        shutil.copy2(config_src, dest / "config.toml")

    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Promote a trial's best model into models/.")
    p.add_argument("--trial", required=True, type=Path,
                   help="Path to the trial directory (e.g. runs/ppo_hoop/20260506_103058)")
    p.add_argument("--models-root", default=Path("models"), type=Path,
                   help="Models directory (default: models)")
    args = p.parse_args(argv)

    dest = promote(args.trial, args.models_root)
    print()
    print(f"  Trial:    {args.trial}")
    print(f"  Promoted  →  {dest}/")
    print()
    print("  To commit:")
    print(f"    git add {dest}")
    print(f"    git commit -m 'model: promote {dest.name} best model'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
