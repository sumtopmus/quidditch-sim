"""Restore config/training.toml from a promoted model's config.toml.

Port of the former `make repro MODEL=<name>` recipe.

Usage:
    python scripts/repro.py --model-dir models/<name>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def repro(model_dir: Path, config_target: Path) -> int:
    src = model_dir / "config.toml"
    if not src.is_file():
        raise SystemExit(
            f"ERROR: {src} not found — model promoted before config snapshots were added?"
        )
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, config_target)
    print(f"Restored {config_target} from {src}")

    env_src = model_dir / "env.toml"
    if env_src.is_file():
        print(
            f"NOTE: {env_src} is from an older promote format; its [env] section is "
            "now part of config/training.toml — verify the values match."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Restore config/training.toml from a promoted model.")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Path to the promoted model directory (e.g. models/red_v1)")
    p.add_argument("--config-target", default=Path("config/training.toml"), type=Path)
    args = p.parse_args(argv)
    return repro(args.model_dir, args.config_target)


if __name__ == "__main__":
    sys.exit(main())
