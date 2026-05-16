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


def _section_header(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    wandb_meta = ctx["wandb_meta"]
    run_dir = ctx["run_dir"]

    # run_dir basename is <timestamp> (under runs/<name>/<ts>/) or <name>
    # (under models/<name>/).  Join with run_name unless the basename already
    # IS the run_name (e.g., the vendored models/<name>/ layout).
    timestamp = run_dir.name
    title = (
        f"# MODEL: {cfg.run_name}_{timestamp}"
        if timestamp != cfg.run_name else f"# MODEL: {cfg.run_name}"
    )

    status = "promoted" if wandb_meta else "run-only"
    git = meta.get("git_hash", "(unknown)") if meta else "(unknown)"

    lines = [
        title,
        "",
        f"**Status:** {status}  ·  **Git:** `{git}`",
    ]
    if wandb_meta:
        name = wandb_meta["name"]
        version = wandb_meta["version"]
        aliases = ", ".join(wandb_meta.get("aliases", []))
        lines.append(f"**W&B:** `wandb://{name}:prod` ({version}, aliases: {aliases})")
    return "\n".join(lines)


def _section_summary(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    hydra_yaml = ctx["hydra_yaml"]

    description = ""
    if hasattr(cfg, "get"):
        description = (cfg.get("description") or "").strip()
    elif hasattr(cfg, "description"):
        description = (cfg.description or "").strip()
    if description:
        return "## Summary\n\n" + description

    # Auto-template path.
    learner_id = "drone_0"
    if hasattr(cfg, "env") and cfg.env is not None and hasattr(cfg.env, "get"):
        learner_id = cfg.env.get("learner_id", "drone_0")

    init_mode = "scratch"
    parent = None
    if hasattr(cfg, "init") and cfg.init is not None:
        init_mode = cfg.init.mode
        if hasattr(cfg.init, "get"):
            parent = cfg.init.get("parent", None)

    obs_name = cfg.obs.name
    n_stack = cfg.obs.n_stack
    total_steps = int(cfg.trainer.total_timesteps)
    lr = cfg.trainer.lr
    curriculum = "(unknown)"
    if hasattr(cfg, "curriculum") and cfg.curriculum is not None and hasattr(cfg.curriculum, "get"):
        curriculum = cfg.curriculum.get("name", "(unknown)")

    # Opponent shorthand: read `spec` field if present, else _target_ class name.
    opp_short = None
    opp = cfg.get("opponent", None) if hasattr(cfg, "get") else None
    if opp is not None:
        if hasattr(opp, "get"):
            opp_short = opp.get("spec", None) or opp.get("_target_", "").rsplit(".", 1)[-1]
        else:
            opp_short = str(opp)

    # Reward group choice from hydra.yaml; fall back if absent.
    reward_choice = "(unknown)"
    if hydra_yaml:
        reward_choice = (
            hydra_yaml.get("hydra", {})
            .get("runtime", {})
            .get("choices", {})
            .get("reward", "(unknown)")
        )

    parts = [f"{learner_id} learner trained from {init_mode}"]
    if parent:
        parts.append(f"(parent: {parent})")
    if opp_short:
        parts.append(f"against {opp_short}")
    parts.append(f"on {obs_name} × n_stack={n_stack}")
    parts.append(f"reward stack {reward_choice}")
    parts.append(f"lr={lr}")
    parts.append(curriculum)
    parts.append(f"{total_steps:,} steps.")

    body = ", ".join(parts)
    return "## Summary\n\n" + body


def _section_lineage(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    init_mode = cfg.init.mode if hasattr(cfg, "init") and cfg.init is not None else "scratch"
    if init_mode == "scratch":
        return "## Lineage\n\n- **init mode:** `scratch` — no parent"

    parent = None
    if hasattr(cfg.init, "get"):
        parent = cfg.init.get("parent", None)
    chain_total = meta.get("parent_chain_total", None) if meta else None
    this_total = int(cfg.trainer.total_timesteps)

    lines = ["## Lineage", "", f"- **init mode:** `{init_mode}`"]
    if parent:
        lines.append(f"- **parent:** `{parent}`")
    if chain_total is not None:
        lines.append(
            f"- **parent chain total:** {chain_total:,} steps "
            f"(this run is {this_total:,} of that)"
        )
    return "\n".join(lines)
