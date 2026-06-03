"""Loader for a run's persisted context: .hydra/{config,meta,hydra}.yaml +
_wandb_metadata.json.

Public API:
    load_run_context(run_dir) -> dict

Used by:
    - scripts._render_model_doc (model-doc rendering; re-imports this)
    - core.inventory             (per-model row construction)
    - core.obs_compat            (parent obs spec lookup without loading weights)

No filesystem writes, no wandb calls.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def load_run_context(run_dir: Path) -> dict[str, Any]:
    """Gather config + meta + hydra-choices + wandb-meta into one dict.

    Required: `.hydra/config.yaml`.  All other inputs optional; missing ones
    surface as `None` in the returned ctx.
    """
    hdir = Path(run_dir) / ".hydra"
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
    # resolve=False: hydra.yaml carries interpolations like `${run_name}` in
    # hydra.sweep.dir that reference the parent config's scope and fail to
    # resolve standalone.
    hydra_yaml = (
        OmegaConf.to_container(OmegaConf.load(hydra_yaml_path), resolve=False)
        if hydra_yaml_path.exists() else None
    )

    wandb_meta_path = Path(run_dir) / "_wandb_metadata.json"
    wandb_meta: dict[str, Any] | None = None
    if wandb_meta_path.exists():
        try:
            wandb_meta = json.loads(wandb_meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not read %s: %s", wandb_meta_path, e)
            wandb_meta = None

    return {
        "run_dir": Path(run_dir),
        "cfg": cfg,
        "meta": meta,
        "hydra_yaml": hydra_yaml,
        "wandb_meta": wandb_meta,
    }


# Legacy alias: scripts._render_model_doc historically named this
# `_load_run_context`; keep the leading-underscore name working.
_load_run_context = load_run_context
