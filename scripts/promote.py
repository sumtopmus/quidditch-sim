"""Promote a training run's best model to canonical / vendored status.

Two-step:

  1. Wandb side: look up the artifact logged by this run, add aliases
     `prod` and `<run_name>` (mutable aliases that move as new versions
     get promoted), persist via art.save().

  2. Repo side: copy best_model.zip + .hydra/ → models/<run_name>/, write
     `_wandb_metadata.json` pinning the IMMUTABLE version (`v3`, not `prod`).
     Print a hint reminding the user to `git add && git commit` if they
     want to vendor the checkpoint.

Usage:
    python -m scripts.promote runs/ppo_hoop_blue_5/20260514_120000

Or via the Makefile:
    make promote RUN_NAME=ppo_hoop_blue_5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb
from omegaconf import OmegaConf


def _resolve_run_name(run_dir: Path) -> str:
    """Read run_name from the run's .hydra/config.yaml."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return str(cfg.run_name)


def _find_run_artifact(run_name: str, timestamp: str):
    """Find the wandb artifact logged by run_id=<run_name>_<timestamp>.

    The simple case: `<run_name>:latest` points to the just-finished run
    (this is the universal case immediately after training completes, since
    `log_run_artifact` aliases its output as `:latest`).
    """
    api = wandb.Api()
    return api.artifact(f"{run_name}:latest")


def promote_run_dir(run_dir: Path, run_name: str, models_root: Path) -> None:
    """Two-step promote: alias the artifact, copy + pin into models/."""
    run_dir = Path(run_dir).resolve()
    src = run_dir / "best_model.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — was eval triggered, or did training crash early?"
        )

    timestamp = run_dir.name
    art = _find_run_artifact(run_name, timestamp)

    # Mutable aliases: add `prod` + `<run_name>` if not present.
    aliases = list(art.aliases)
    for alias in ("prod", run_name):
        if alias not in aliases:
            aliases.append(alias)
    art.aliases = aliases
    art.save()

    # Repo side: copy + pin.
    dest = Path(models_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "best_model.zip")
    hydra_src = run_dir / ".hydra"
    if hydra_src.exists():
        hydra_dest = dest / ".hydra"
        if hydra_dest.exists():
            shutil.rmtree(hydra_dest)
        shutil.copytree(hydra_src, hydra_dest)
    def _str_or_none(v):
        return v if isinstance(v, str) else None
    metadata = {
        "name":     run_name,
        "version":  art.version,
        "entity":   _str_or_none(getattr(art, "entity", None)),
        "project":  _str_or_none(getattr(art, "project", None)),
        "aliases":  list(art.aliases),
        "logged_by_run_id": f"{run_name}_{timestamp}",
    }
    (dest / "_wandb_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"")
    print(f"  Run:      {run_dir}")
    print(f"  Wandb:    {run_name}:{art.version}  (aliases: {sorted(art.aliases)})")
    print(f"  Vendored: {dest}")
    print(f"")
    print(f"  To use as a pretrain parent in a new experiment YAML:")
    print(f"    init:")
    print(f"      parent: wandb://{run_name}:prod")
    print(f"")
    print(f"  To vendor this checkpoint into the repo:")
    print(f"    git add {dest} && git commit -m 'model: promote {run_name}'")


def main() -> None:
    p = argparse.ArgumentParser(description="Promote a run's best_model to canonical.")
    p.add_argument("run_dir", help="runs/<run_name>/<timestamp>/")
    p.add_argument("--models-root", default="models", help="defaults to models/")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_name = _resolve_run_name(run_dir)
    promote_run_dir(run_dir=run_dir, run_name=run_name,
                    models_root=Path(args.models_root))


if __name__ == "__main__":
    main()
