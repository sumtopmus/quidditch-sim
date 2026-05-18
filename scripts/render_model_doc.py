"""CLI entrypoint: render MODEL.md for a run dir.

Usage:
    python -m scripts.render_model_doc --run-dir runs/ppo_hoop_test/20260516_120000
    make describe-run RUN_NAME=ppo_hoop_test                   # latest trial auto-resolved
    make describe-run RUN_NAME=ppo_hoop_test TRIAL=20260516_120000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m scripts.render_model_doc` and direct invocation alike.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._render_model_doc import render_model_doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Path to a run dir containing .hydra/{config,meta,hydra}.yaml")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    doc = render_model_doc(run_dir)
    out_path = run_dir / "MODEL.md"
    out_path.write_text(doc)
    print(f"wrote {out_path} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
