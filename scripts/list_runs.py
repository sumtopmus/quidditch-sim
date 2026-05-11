"""List training runs and promoted models — port of the former `make list-runs` recipe.

Output format mirrors the original shell version:

  === runs/ ===
    <config_name>/
      <trial_name>
      <trial_name>
  === models/ ===
    <model_dir>/
      <file>
      <file>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def list_runs(runs_root: Path, models_root: Path) -> int:
    print(f"=== {runs_root}/ ===")
    if not runs_root.is_dir() or not any(runs_root.iterdir()):
        print("  (none)")
    else:
        for cfg in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            print(f"  {cfg.name}/")
            trials = sorted(
                (t for t in cfg.iterdir() if t.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
            for t in trials:
                print(f"    {t.name}")

    print()
    print(f"=== {models_root}/ ===")
    if not models_root.is_dir() or not any(p for p in models_root.iterdir() if p.is_dir()):
        print("  (none — run 'make promote' after a successful training run)")
    else:
        for m in sorted(p for p in models_root.iterdir() if p.is_dir()):
            print(f"  {m}/")
            for child in sorted(m.iterdir()):
                print(f"      {child.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List training runs and promoted models.")
    p.add_argument("--runs-root", default=Path("runs"), type=Path)
    p.add_argument("--models-root", default=Path("models"), type=Path)
    args = p.parse_args(argv)
    return list_runs(args.runs_root, args.models_root)


if __name__ == "__main__":
    sys.exit(main())
