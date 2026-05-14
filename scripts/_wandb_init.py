"""init_wandb — single call site for wandb.init.

Owns: run identity (name/id/group), tag derivation, config snapshot, mode/
entity resolution.  Called from scripts/train.py (after Hydra composes
cfg) and from scripts/eval_team.py when `WANDB=1` is set.

In WANDB_MODE=disabled, wandb.init returns a no-op stub; tests rely on
this (conftest.py sets the env var globally).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


# Map opponent _target_ class names to short tag-friendly strings.
_OPP_SHORT_NAMES = {
    "BeelineRed":          "beeline_red",
    "BeelineBlue":         "beeline_blue",
    "IntercepterBlue":     "intercepter_blue",
    "ZeroOpponent":        "zero",
    "FrozenPolicyOpponent": "frozen",
    "MixtureOpponent":     "mixture",
}


def _opponent_short_name(opp_cfg) -> str | None:
    """Last segment of cfg.opponent._target_ → short tag (or None for single-agent)."""
    target = opp_cfg.get("_target_") if isinstance(opp_cfg, (dict, DictConfig)) else None
    if not target:
        return None
    cls = target.rsplit(".", 1)[-1]
    return _OPP_SHORT_NAMES.get(cls, cls.lower())


def _resolve_job_type(role: str) -> str:
    """Promote `role` to sweep-train when WANDB_SWEEP_ID is set.

    `wandb agent` sets WANDB_SWEEP_ID automatically; reading it lets the
    dashboard filter "manual training" vs "sweep children" without any
    code in the sweep YAML to remember to set a flag.
    """
    if role == "train" and os.environ.get("WANDB_SWEEP_ID"):
        return "sweep-train"
    return role


def _build_tags(cfg: DictConfig) -> list[str]:
    """Derive filter tags from cfg.  Empty/None values are dropped."""
    tags: list[str] = []
    learner = cfg.env.get("learner_id") if "learner_id" in cfg.env else None
    if learner:
        tags.append(str(learner))
    if cfg.obs.get("name"):
        tags.append(str(cfg.obs.name))
    if cfg.init.get("mode"):
        tags.append(str(cfg.init.mode))
    opp = _opponent_short_name(cfg.get("opponent"))
    if opp:
        tags.append(opp)
    extra = list(cfg.wandb.get("tags_extra", []))
    tags.extend(str(t) for t in extra if t)
    return tags


def init_wandb(cfg: DictConfig, run_dir: Path, role: str) -> Any:
    """Initialize the wandb run for this Hydra-composed cfg.

    `role` ∈ {"train", "eval"}.  Promoted to "sweep-train" automatically
    when WANDB_SWEEP_ID is in the environment.

    Returns the wandb run object (or a disabled-mode stub).
    """
    timestamp = Path(run_dir).name
    run_id = f"{cfg.run_name}_{timestamp}"

    entity = os.environ.get("WANDB_ENTITY")
    if cfg.wandb.get("entity_override"):
        entity = str(cfg.wandb.entity_override)

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg.wandb.project),
        entity=entity,
        name=run_id,
        id=run_id,
        dir=str(run_dir),
        group=str(cfg.run_name),
        job_type=_resolve_job_type(role),
        tags=_build_tags(cfg),
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        mode=os.environ.get("WANDB_MODE", "online"),
        resume="allow",
        notes=str(cfg.wandb.get("notes", "")),
    )
