"""One-shot upload of legacy promoted models to wandb.

For each `models/<run_name>/` (post-Hydra migration: has .hydra/config.yaml
+ best_model.zip), this script:

  1. Reads the run's metadata from .hydra/{config,meta}.yaml.
  2. Translates legacy filesystem-path parent references to wandb URIs.
  3. Builds and logs a wandb artifact with aliases `prod`, `<run_name>`, `v0`.
  4. Writes `_wandb_metadata.json` into models/<run_name>/ pinning v0.

Idempotent: skips dirs that already have _wandb_metadata.json.

Usage:
    python -m scripts.upload_legacy_models                    # all dirs in models/
    python -m scripts.upload_legacy_models models/ppo_hoop_red_1
    python -m scripts.upload_legacy_models --project drone-quidditch ...

Requires a live wandb connection (it's a migration script, run once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb
import yaml


# Map legacy filesystem-path parents to wandb URIs.  Each entry is keyed by
# the literal `init.parent` string found in the legacy config and maps to
# the wandb URI that means the same thing post-migration.
LEGACY_PARENT_REWRITES: dict[str, str] = {
    "models/ppo_hoop_fixed_start_20260504_023051/best_model":
        "wandb://ppo_hoop_fixed_start_20260504_023051:prod",
    "models/ppo_hoop_rand_start_20260505_174509/best_model":
        "wandb://ppo_hoop_rand_start_20260505_174509:prod",
}


def _rewrite_parent_uri(legacy_path: str | None) -> str | None:
    """Convert legacy filesystem `init.parent` to a wandb URI if known."""
    if legacy_path is None:
        return None
    if legacy_path in LEGACY_PARENT_REWRITES:
        return LEGACY_PARENT_REWRITES[legacy_path]
    if legacy_path.startswith(("wandb://", "wandb-artifact://")):
        return legacy_path
    return legacy_path  # leave path as-is for un-rewritten legacy edges


def upload_one(model_dir: Path, project: str = "drone-quidditch") -> bool:
    """Upload one legacy model.  Returns True if skipped (idempotent)."""
    model_dir = Path(model_dir).resolve()
    if (model_dir / "_wandb_metadata.json").exists():
        print(f"[skip] {model_dir.name}: _wandb_metadata.json already present")
        return True

    cfg_path = model_dir / ".hydra" / "config.yaml"
    meta_path = model_dir / ".hydra" / "meta.yaml"
    if not cfg_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"{model_dir} missing .hydra/{{config,meta}}.yaml — "
            f"run scripts/migrate_legacy_models.py first"
        )
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    meta = yaml.safe_load(meta_path.read_text()) or {}

    run_name = cfg.get("run_name", model_dir.name)
    obs = cfg.get("obs", {})
    init = cfg.get("init", {})
    final = meta.get("final_stats", {})

    rewritten_parent = _rewrite_parent_uri(init.get("parent"))

    # Open a wandb run so log_artifact has a producer (the artifact lineage
    # DAG needs each artifact to be `logged_by` *something*).  job_type
    # marks it as a one-shot migration.
    run = wandb.init(
        project=project,
        name=f"upload_legacy:{run_name}",
        id=f"upload_legacy_{run_name}",
        job_type="upload-legacy",
        tags=["upload-legacy", obs.get("name", "?")],
        resume="allow",
        config={"source": "upload_legacy_models", "model_dir": str(model_dir)},
    )

    art = wandb.Artifact(
        name=run_name,
        type="model",
        metadata={
            "obs_spec":           obs.get("name"),
            "n_stack":            int(obs.get("n_stack", 1)),
            "init_mode":          init.get("mode", "scratch"),
            "parent_uri":         rewritten_parent,
            "parent_chain_total": int(meta.get("parent_chain_total", 0)),
            "best_eval_reward":   final.get("best_eval_reward"),
            "git_hash":           meta.get("git_hash", "<legacy>"),
            "legacy_dir":         str(model_dir),
        },
    )
    art.add_file(str(model_dir / "best_model.zip"))
    art.add_dir(str(model_dir / ".hydra"), name=".hydra")

    aliases = ["latest", "prod", run_name, "v0"]
    run.log_artifact(art, aliases=aliases)
    art.wait()

    def _str_or_none(v):
        return v if isinstance(v, str) else None
    metadata = {
        "name":     run_name,
        "version":  art.version,
        "entity":   _str_or_none(getattr(art, "entity", None)),
        "project":  project,
        "aliases":  aliases,
        "logged_by_run_id": _str_or_none(getattr(run, "id", None)),
    }
    (model_dir / "_wandb_metadata.json").write_text(json.dumps(metadata, indent=2))

    wandb.finish()
    print(f"[done] {model_dir.name} → {run_name}:{art.version} (aliases: {aliases})")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Upload legacy promoted models to wandb.")
    p.add_argument("dirs", nargs="*", default=[],
                   help="models/<name>/ dirs to upload (default: all under models/)")
    p.add_argument("--project", default="drone-quidditch")
    args = p.parse_args()

    if args.dirs:
        targets = [Path(d) for d in args.dirs]
    else:
        targets = sorted(
            d for d in Path("models").iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    for d in targets:
        try:
            upload_one(d, project=args.project)
        except Exception as e:
            print(f"[error] {d.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
