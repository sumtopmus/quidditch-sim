"""One-shot migration: legacy promoted-model dirs → Hydra layout.

Each `models/<name>/` is expected to contain:
  - best_model.zip
  - run_info.toml (legacy run-state record)
  - config.toml   (legacy config snapshot — optional, some old models lack it)

After migration:
  - models/<name>/.hydra/config.yaml   (synthesized minimal Hydra-shaped config)
  - models/<name>/.hydra/meta.yaml     (git_hash, parent_chain_total, init_mode, ...)
  - run_info.toml and config.toml are NOT deleted — they remain as audit.

Idempotent: skips if .hydra/ already exists.

Usage:
    python scripts/migrate_legacy_models.py models/ppo_hoop_blue_4_20260511_202612
    python scripts/migrate_legacy_models.py models/*  # migrate all

The obs-spec → name mapping uses LEGACY_SPECS (hand-curated, matches the
removed scripts/backfill_obs_spec.py).  If a model's [obs] block dim/n_stack
matches a canonical ObsSpec exactly, that name is recorded in the new
config.yaml; otherwise the dim+n_stack are recorded and an explicit
LEGACY_SPECS entry must be added by hand.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.quidditch.obs_spec import SPEC_BY_NAME


# Hand-curated mapping from promoted model dir names to their obs spec name.
# Mirrors the legacy scripts/backfill_obs_spec.py LEGACY_SPECS.
LEGACY_SPECS: dict[str, str] = {
    "ppo_hoop_fixed_start_20260430_224234":     "SIMPLE_ENV_OBS",
    "ppo_hoop_fixed_start_20260504_023051":     "SIMPLE_ENV_OBS",
    "ppo_hoop_rand_start_20260430_234354":      "SIMPLE_ENV_OBS",
    "ppo_hoop_rand_start_20260505_174509":      "SIMPLE_ENV_OBS",
    "ppo_hoop_red_1_20260506_103058":           "DUEL_V1_BODY",
    "ppo_hoop_blue_1_20260507_194423":          "DUEL_V1_BODY",
    "ppo_hoop_blue_4_20260511_202612":          "DUEL_V2_WORLD",
}


def _resolve_obs_name(model_dir: Path, info: dict) -> str:
    """Resolve the canonical obs-spec name for this promoted model."""
    if model_dir.name in LEGACY_SPECS:
        return LEGACY_SPECS[model_dir.name]
    obs = info.get("obs", {})
    target_dim = obs.get("dim")
    for name, spec in SPEC_BY_NAME.items():
        if spec.dim == target_dim:
            return name
    raise ValueError(
        f"Could not resolve obs spec for {model_dir.name}: dim={target_dim}.  "
        f"Add an entry to LEGACY_SPECS."
    )


def _synthesize_hydra_config(model_dir: Path, info: dict) -> dict:
    """Produce a minimal .hydra/config.yaml shape from legacy info.toml."""
    obs_name = _resolve_obs_name(model_dir, info)
    n_stack = int(info.get("obs", {}).get("n_stack", 1))
    return {
        "run_name": info.get("run", {}).get("name", model_dir.name),
        "seed": 42,
        "obs": {"name": obs_name, "n_stack": n_stack},
        # init.mode reflects how this run was launched.
        "init": {
            "mode": "pretrain" if "pretrain" in info else "scratch",
            "parent": info.get("pretrain", {}).get("parent"),
            "parent_run": None,
            "parent_checkpoint": None,
            "obs_surgery": False,
        },
    }


def _synthesize_meta_yaml(info: dict) -> dict:
    """Produce a .hydra/meta.yaml shape from legacy info.toml."""
    pretrain = info.get("pretrain", {})
    steps_trained = info.get("run", {}).get("steps_trained")
    completed = int(steps_trained) if isinstance(steps_trained, int) else 0
    return {
        "git_hash": "<legacy>",
        "parent_chain_total": int(pretrain.get("total_steps", 0)),
        "init_mode": "pretrain" if pretrain else "scratch",
        "parent_path": pretrain.get("parent"),
        "final_stats": {
            "wall_time_s": None,
            "completed_steps": completed,
            "best_eval_reward": None,
            "peak_eval_step": None,
        },
    }


def migrate_one(model_dir: Path) -> None:
    """Migrate a single promoted-model dir.  Idempotent."""
    model_dir = Path(model_dir).resolve()
    hydra_dir = model_dir / ".hydra"
    if hydra_dir.exists():
        print(f"[skip] {model_dir.name}: .hydra/ already exists")
        return
    info_path = model_dir / "run_info.toml"
    if not info_path.exists():
        info_path = model_dir / "info.toml"
    if not info_path.exists():
        raise FileNotFoundError(f"No run_info.toml or info.toml under {model_dir}")
    info = tomllib.loads(info_path.read_text())
    hydra_dir.mkdir()
    cfg = _synthesize_hydra_config(model_dir, info)
    meta = _synthesize_meta_yaml(info)
    (hydra_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (hydra_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"[done] {model_dir.name} → {hydra_dir}/")


def main():
    p = argparse.ArgumentParser(description="Migrate legacy promoted models to Hydra layout.")
    p.add_argument("models", nargs="+", help="model directories to migrate")
    args = p.parse_args()
    for m in args.models:
        try:
            migrate_one(Path(m))
        except Exception as e:
            print(f"[error] {m}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
