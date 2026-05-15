"""Artifact registry helpers.

Two functions:

  - resolve_parent(uri_or_path, models_root=Path("models")) → Path
      Maps `wandb://run:alias` (or `wandb-artifact://entity/project/run:alias`)
      to a local checkpoint path.  Filesystem paths are returned as-is.
      Wandb URIs are: alias-resolved → cache-checked → downloaded if needed.
      Always calls wandb.use_artifact on the current run for lineage.

  - log_run_artifact(run, run_dir, cfg, parent_chain_total, best_eval_reward)
      Called at the end of every training run.  Logs best_model.zip + .hydra/
      as `<cfg.run_name>:latest` (plus auto-versioned :v<N>).

The committed `models/<run>/_wandb_metadata.json` pins the immutable
version (e.g. v3) — never the alias.  If a later promote shifts :prod to a
newer version, the resolver detects the mismatch and falls through to
downloading the new version instead of silently serving the stale
committed checkpoint.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


def _resolve_default_entity_project() -> tuple[str | None, str]:
    """Compute default (entity, project) for `wandb.Api().artifact(...)` calls.

    Order of precedence:
      1. WANDB_ENTITY / WANDB_PROJECT env vars (let users repoint without
         touching code or config).
      2. The live wandb.run (when resolve_parent fires inside a training
         run, the run's entity/project is the right default).
      3. Hardcoded fallback: project=drone-quidditch, entity=None.

    Without explicit qualification, wandb.Api() looks up the user's
    *workspace default* project — often `uncategorized` — which never
    contains training artifacts.
    """
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if wandb.run is not None:
        entity = entity or wandb.run.entity
        project = project or wandb.run.project
    return entity, project or "drone-quidditch"


@dataclass(frozen=True)
class _WandbURI:
    """Parsed wandb:// or wandb-artifact:// URI."""
    entity: str | None
    project: str | None
    name: str
    alias: str

    def short_form(self) -> str:
        return f"wandb://{self.name}:{self.alias}"

    def for_api(self, default_entity: str | None, default_project: str | None) -> str:
        """Build the wandb-API artifact path with the strongest qualification
        the inputs allow.

        wandb.Api().artifact accepts three positional forms:
          - "entity/project/name:alias"   (fully qualified)
          - "project/name:alias"          (entity falls back to workspace default)
          - "name:alias"                  (entity AND project fall back to workspace defaults)
        The two-arg form is critical here: when the user has only a workspace-
        default entity set (no WANDB_ENTITY env), we still want to pin the
        project so the lookup doesn't route to `uncategorized/`.  Emitting the
        bare `name:alias` form whenever entity is None is the bug that caused
        promote/lineage/resolve to all fail with "project 'uncategorized'
        not found under entity 'X'" — workspace-default entity + workspace-
        default project is the wrong cross product for our artifacts.
        """
        ent = self.entity or default_entity
        proj = self.project or default_project
        if ent and proj:
            return f"{ent}/{proj}/{self.name}:{self.alias}"
        if proj:
            return f"{proj}/{self.name}:{self.alias}"
        return f"{self.name}:{self.alias}"


def _parse_wandb_uri(uri: str) -> _WandbURI:
    """Parse `wandb://run:alias` or `wandb-artifact://entity/project/run:alias`."""
    if uri.startswith("wandb-artifact://"):
        body = uri[len("wandb-artifact://"):]
        parts = body.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid wandb-artifact URI: {uri!r} "
                "(expected wandb-artifact://entity/project/name:alias)"
            )
        entity, project, name_and_alias = parts
    elif uri.startswith("wandb://"):
        entity = None
        project = None
        name_and_alias = uri[len("wandb://"):]
    else:
        raise ValueError(f"Not a wandb URI: {uri!r}")

    if ":" not in name_and_alias:
        raise ValueError(f"Invalid wandb URI {uri!r}: missing alias (e.g. `:prod`)")
    name, alias = name_and_alias.rsplit(":", 1)
    if not name or not alias:
        raise ValueError(f"Invalid wandb URI {uri!r}: empty name or alias")
    return _WandbURI(entity=entity, project=project, name=name, alias=alias)


def _is_wandb_uri(s: str) -> bool:
    return s.startswith(("wandb://", "wandb-artifact://"))


def _committed_metadata(committed_dir: Path) -> dict | None:
    """Read `_wandb_metadata.json` if present."""
    p = committed_dir / "_wandb_metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def resolve_parent(
    uri_or_path: str | Path,
    models_root: Path = Path("models"),
) -> Path:
    """Resolve a parent reference to a local checkpoint Path.

    Filesystem paths (anything not starting with `wandb://` or
    `wandb-artifact://`) are returned as-is.  Wandb URIs go through:

      1. Resolve alias → immutable version via wandb.Api().artifact(uri).
      2. If models/<name>/_wandb_metadata.json pins the same version,
         return the committed path.
      3. Else download via artifact.download(root=models/.cache/<name>_v<N>/).
      4. In both cases, if wandb.run is active, call use_artifact for lineage.

    Returns the path to the loadable best_model.zip (or whatever the
    artifact wrapped — convention is best_model.zip).
    """
    s = str(uri_or_path)
    if not _is_wandb_uri(s):
        return Path(s)

    parsed = _parse_wandb_uri(s)
    api = wandb.Api()
    # Resolve alias → immutable version.  Default entity/project come from
    # env vars or the active wandb.run — NOT wandb's workspace default,
    # which routes lookups to `uncategorized/` and silently misses our
    # artifacts.
    default_entity, default_project = _resolve_default_entity_project()
    art = api.artifact(parsed.for_api(default_entity=default_entity,
                                       default_project=default_project))
    version = art.version

    # Lineage edge: only if there's a live training run.
    if wandb.run is not None:
        wandb.run.use_artifact(art)

    # Cache hit on committed dir?
    committed = Path(models_root) / parsed.name
    meta = _committed_metadata(committed)
    if meta is not None and meta.get("version") == version and meta.get("name") == parsed.name:
        cp = committed / "best_model.zip"
        if cp.exists():
            return cp
        # Metadata pins but best_model.zip is missing — fall through to download.

    # Download into the gitignored cache.
    cache_dir = Path(models_root) / ".cache" / f"{parsed.name}_{version}"
    art.download(root=str(cache_dir))
    return cache_dir / "best_model.zip"


def log_run_artifact(
    run: Any,
    run_dir: Path,
    cfg: Any,
    parent_chain_total: int,
    best_eval_reward: float | None,
) -> None:
    """Log this run's model + .hydra/ as a wandb artifact.

    Source-file preference: best_model.zip (written by EvalCallback after a
    successful eval) → final_model.zip (always written by train.py's
    `finally:` block).  Either way the artifact's internal name is
    `best_model.zip` so downstream `resolve_parent` keeps working without a
    special case.  Metadata field `model_kind` ∈ {"best", "final"} records
    the source so a final-only run is auditable (no proven eval signal).

    Aliased `:latest` automatically; further aliases (`:prod`, `:<run_name>`)
    are added by scripts/promote.py.  No-op when wandb.run is None or in
    WANDB_MODE=disabled.
    """
    if run is None or getattr(run, "disabled", False):
        return
    run_dir = Path(run_dir)
    best = run_dir / "best_model.zip"
    final = run_dir / "final_model.zip"
    hydra_dir = run_dir / ".hydra"

    if best.exists():
        model_path, model_kind = best, "best"
    elif final.exists():
        model_path, model_kind = final, "final"
    else:
        # Training crashed before either file landed.
        return

    art = wandb.Artifact(
        name=str(cfg.run_name),
        type="model",
        metadata={
            "obs_spec":           str(cfg.obs.name),
            "n_stack":            int(cfg.obs.n_stack),
            "learner_id":         cfg.env.get("learner_id"),
            "init_mode":          str(cfg.init.mode),
            "parent_uri":         cfg.init.parent,
            "parent_chain_total": int(parent_chain_total),
            "best_eval_reward":   best_eval_reward,
            "model_kind":         model_kind,
        },
    )
    art.add_file(str(model_path), name="best_model.zip")
    if hydra_dir.exists():
        art.add_dir(str(hydra_dir), name=".hydra")
    run.log_artifact(art, aliases=["latest"])
