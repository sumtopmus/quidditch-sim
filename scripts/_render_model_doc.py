"""Pure renderer for per-model MODEL.md spec sheets.

Reads .hydra/{config,meta,hydra}.yaml + _wandb_metadata.json from a run dir
and returns a Markdown string.  No filesystem writes, no wandb calls.

See docs/superpowers/specs/2026-05-16-model-doc-generator-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _load_run_context(run_dir: Path) -> dict[str, Any]:
    """Gather config + meta + hydra-choices + wandb-meta into one dict.

    Required: `.hydra/config.yaml`.  All other inputs optional; missing ones
    surface as `None` in the returned ctx.
    """
    hdir = run_dir / ".hydra"
    cfg_path = hdir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"required input missing: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    meta_path = hdir / "meta.yaml"
    meta = (
        OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
        if meta_path.exists() else None
    )

    hydra_yaml_path = hdir / "hydra.yaml"
    hydra_yaml = (
        OmegaConf.to_container(OmegaConf.load(hydra_yaml_path), resolve=True)
        if hydra_yaml_path.exists() else None
    )

    wandb_meta_path = run_dir / "_wandb_metadata.json"
    wandb_meta = (
        json.loads(wandb_meta_path.read_text()) if wandb_meta_path.exists() else None
    )

    return {
        "cfg": cfg,
        "meta": meta,
        "hydra_yaml": hydra_yaml,
        "wandb_meta": wandb_meta,
        "run_dir": run_dir,
    }
