"""Promote a training run's best model to canonical / vendored status.

Two-step (unchanged from the original scripts/promote.py):

  1. Wandb side: alias the artifact this run logged with `prod` + `<run_name>`,
     save() to persist.
  2. Repo side: copy best_model.zip + .hydra/ + MODEL.md (if present) into
     models/<run_name>/, write _wandb_metadata.json pinning the IMMUTABLE
     version (e.g. `v3`, not the alias `prod`).

Public API:
    promote_run(run_dir, *, alias="prod") -> PromoteResult

The script `scripts/promote.py` is a thin CLI wrapper around this; it
also re-exports the lower-level helpers (`promote_run_dir`,
`_resolve_entity_project`, `_find_run_artifact`) for back-compat with
existing tests/callers.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import wandb
from omegaconf import OmegaConf


@dataclass(frozen=True)
class PromoteResult:
    run_name: str
    wandb_version: str
    wandb_alias: str
    target_dir: Path
    copied_files: list[str]


def _resolve_run_name(run_dir: Path) -> str:
    """Read run_name from the run's .hydra/config.yaml."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return str(cfg.run_name)


def _resolve_entity_project(run_dir: Path) -> tuple[str | None, str]:
    """Resolve wandb (entity, project) for artifact lookups.

    Order: WANDB_PROJECT / WANDB_ENTITY env vars → cfg.wandb in
    .hydra/config.yaml → drone-quidditch fallback.
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    wandb_cfg = cfg.get("wandb") if hasattr(cfg, "get") else None
    wandb_cfg = wandb_cfg or {}
    project = (
        os.environ.get("WANDB_PROJECT")
        or (wandb_cfg.get("project") if hasattr(wandb_cfg, "get") else None)
        or "drone-quidditch"
    )
    entity = (
        os.environ.get("WANDB_ENTITY")
        or (wandb_cfg.get("entity_override") if hasattr(wandb_cfg, "get") else None)
        or None
    )
    return entity, str(project)


def _find_run_artifact(run_name: str, timestamp: str,
                       entity: str | None, project: str):
    """Find the wandb artifact logged by run_id=<run_name>_<timestamp>.

    Looks up `<run_name>:latest`, which is the universal post-training alias
    set by log_run_artifact.  Fully qualified `entity/project/...` to avoid
    falling through to wandb's `uncategorized/` workspace.
    """
    api = wandb.Api()
    qualified = (
        f"{entity}/{project}/{run_name}:latest"
        if entity else f"{project}/{run_name}:latest"
    )
    return api.artifact(qualified)


def promote_run_dir(run_dir: Path, run_name: str, models_root: Path) -> PromoteResult:
    """Two-step promote: alias the artifact, copy + pin into models/."""
    run_dir = Path(run_dir).resolve()
    src = run_dir / "best_model.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — was eval triggered, or did training crash early?"
        )

    timestamp = run_dir.name
    entity, project = _resolve_entity_project(run_dir)
    art = _find_run_artifact(run_name, timestamp, entity=entity, project=project)

    aliases = list(art.aliases)
    for alias in ("prod", run_name):
        if alias not in aliases:
            aliases.append(alias)
    art.aliases = aliases
    art.save()

    dest = Path(models_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    shutil.copy2(src, dest / "best_model.zip")
    copied.append("best_model.zip")
    hydra_src = run_dir / ".hydra"
    if hydra_src.exists():
        hydra_dest = dest / ".hydra"
        if hydra_dest.exists():
            shutil.rmtree(hydra_dest)
        shutil.copytree(hydra_src, hydra_dest)
        copied.append(".hydra/")
    src_doc = run_dir / "MODEL.md"
    if src_doc.exists():
        shutil.copy2(src_doc, dest / "MODEL.md")
        copied.append("MODEL.md")

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
    copied.append("_wandb_metadata.json")

    return PromoteResult(
        run_name=run_name,
        wandb_version=art.version,
        wandb_alias="prod",
        target_dir=dest,
        copied_files=copied,
    )


def promote_run(run_dir: Path, *, alias: str = "prod",
                models_root: Path = Path("models")) -> PromoteResult:
    """Importable shape used by dsim promote + the TUI Promote task."""
    run_dir = Path(run_dir)
    run_name = _resolve_run_name(run_dir)
    result = promote_run_dir(run_dir=run_dir, run_name=run_name,
                             models_root=Path(models_root))
    if alias != "prod":
        # Caller-supplied alias is the "primary" alias; ensure it's in the list.
        # (promote_run_dir always adds `prod` + run_name; we add the user's
        # alias here if it's something else, like `staging`.)
        # Re-fetch the artifact to update.
        entity, project = _resolve_entity_project(run_dir)
        art = _find_run_artifact(run_name, run_dir.name,
                                 entity=entity, project=project)
        if alias not in art.aliases:
            art.aliases = list(art.aliases) + [alias]
            art.save()
        result = PromoteResult(
            run_name=result.run_name,
            wandb_version=result.wandb_version,
            wandb_alias=alias,
            target_dir=result.target_dir,
            copied_files=result.copied_files,
        )
    return result
