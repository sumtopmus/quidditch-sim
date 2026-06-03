"""CLI wrapper around core.promote.promote_run / promote_run_dir.

For new code, prefer:
    dsim promote <run-name> [--alias prod]
This script remains for back-compat with `python -m scripts.promote`
and existing tests that import the lower-level helpers under their
historical names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-export legacy names: tests/scripts/test_promote.py imports promote_run_dir,
# _resolve_entity_project, _find_run_artifact from scripts.promote.  The
# implementation lives in core.promote now; re-importing here preserves the
# public surface without duplication.
from core.promote import (  # noqa: F401
    PromoteResult,
    _find_run_artifact,
    _resolve_entity_project,
    _resolve_run_name,
    promote_run,
    promote_run_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Promote a run's best_model to canonical.")
    p.add_argument("run_dir", help="runs/<run_name>/<timestamp>/")
    p.add_argument("--models-root", default="models", help="defaults to models/")
    p.add_argument("--alias", default="prod",
                   help="W&B alias to set (default: prod)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    result = promote_run(run_dir, alias=args.alias,
                         models_root=Path(args.models_root))

    print(f"  Run:      {run_dir}")
    print(f"  Wandb:    {result.run_name}:{result.wandb_version}  "
          f"(alias: {result.wandb_alias})")
    print(f"  Vendored: {result.target_dir}")
    print(f"  Copied:   {', '.join(result.copied_files)}")
    print("")
    print("  To use as a pretrain parent in a new experiment YAML:")
    print(f"    init:")
    print(f"      parent: wandb://{result.run_name}:prod")
    print("")
    print("  To vendor this checkpoint into the repo:")
    print(f"    git add {result.target_dir} && git commit -m 'model: promote {result.run_name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
