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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


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
        """The full entity/project/name:alias string the wandb API needs."""
        ent = self.entity or default_entity
        proj = self.project or default_project
        if ent is None or proj is None:
            # Fall through; wandb.Api() will use defaults from env + workspace.
            return f"{self.name}:{self.alias}"
        return f"{ent}/{proj}/{self.name}:{self.alias}"


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
    # Resolve alias → immutable version.  wandb.Api() pulls defaults from
    # env/workspace when entity/project are absent in our shorthand URIs.
    art = api.artifact(parsed.for_api(default_entity=None, default_project=None))
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
    """Log this run's best_model + .hydra/ as a wandb artifact.

    Aliased `:latest` automatically; further aliases (`:prod`, `:<run_name>`)
    are added by scripts/promote.py.  No-op when wandb.run is None or in
    WANDB_MODE=disabled.
    """
    if run is None or getattr(run, "disabled", False):
        return
    best = Path(run_dir) / "best_model.zip"
    hydra_dir = Path(run_dir) / ".hydra"
    if not best.exists():
        # No best_model emitted (e.g., training crashed before first eval).
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
        },
    )
    art.add_file(str(best))
    if hydra_dir.exists():
        art.add_dir(str(hydra_dir), name=".hydra")
    run.log_artifact(art, aliases=["latest"])
