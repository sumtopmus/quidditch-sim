"""One-shot: render MODEL.md for every dir under models/ (legacy backfill).

Idempotent — skips dirs that already have MODEL.md unless `--force` is set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m scripts.backfill_model_docs` and direct invocation alike.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._render_model_doc import render_model_doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("models"),
                   help="Root dir to scan (default: ./models)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing MODEL.md files")
    args = p.parse_args()

    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 1

    rendered, skipped, failed = 0, 0, 0
    for model_dir in sorted(args.root.iterdir()):
        if not model_dir.is_dir() or not (model_dir / ".hydra" / "config.yaml").exists():
            continue
        md = model_dir / "MODEL.md"
        if md.exists() and not args.force:
            print(f"[skip] {model_dir.name}: MODEL.md exists (use --force to overwrite)")
            skipped += 1
            continue
        try:
            doc = render_model_doc(model_dir)
            md.write_text(doc)
            print(f"[done] {model_dir.name}: wrote MODEL.md")
            rendered += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {model_dir.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nrendered {rendered}, skipped {skipped}, failed {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
