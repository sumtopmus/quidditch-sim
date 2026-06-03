"""Walk a run's pretrain ancestry — importable walkers + URI parsing helpers.

Two walkers, same return shape (list[LineageNode], oldest-first by historical
convention; walk_chain_local emits child-first then reverses to match).

  walk_chain_local(start)  -> list[LineageNode]
    Reads parent links from each ancestor's .hydra/config.yaml `init.parent`.
    Offline-survivable; truncates if an intermediate path is missing.

  walk_chain_wandb(target) -> list[LineageNode]
    Uses wandb.Api().artifact().logged_by().used_artifacts() for the native
    artifact DAG.  Richer (sees un-vendored intermediates) but needs
    network + credentials.

CLI dispatch lives in scripts/lineage.py (a thin wrapper) and
dsim/commands/lineage.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class LineageNode:
    name: str
    path: Path | None        # local path if known
    parent: str | None       # parent URI/path (None ⇒ scratch)
    final_steps: int | None
    parent_chain_total: int | None
    version: str | None = None  # wandb version, only walker B fills it
    init_mode: str | None = None
    obs_spec: str | None = None
    truncated: bool = False  # True when chain was cut due to missing intermediate


def walk_chain_local(start_path: Path | str) -> list[LineageNode]:
    """Walk back via init.parent in each .hydra/config.yaml, ending at scratch.

    Returns child-first (start_path at index 0).  If an intermediate parent
    can't be located on disk, the last successfully-visited node is marked
    `truncated=True`.
    """
    start = Path(start_path)
    if start.is_file():
        start = start.parent

    chain: list[LineageNode] = []
    cursor: Path | None = start
    safety = 64
    visited: set[Path] = set()

    while cursor is not None and safety > 0:
        safety -= 1
        cursor_r = cursor.resolve()
        if cursor_r in visited:
            break
        visited.add(cursor_r)
        node = _node_from_path(cursor)
        if node is None:
            # Couldn't load the current dir at all (no .hydra/config.yaml);
            # mark prior node as truncated and stop.
            if chain:
                chain[-1] = _mark_truncated(chain[-1])
            break
        chain.append(node)
        next_cursor = _next_parent_path(node)
        # Truncation: node declared a parent we can't locally walk to (either
        # the path doesn't exist, or it's a wandb:// URI which walker A
        # cannot follow).  Mark the appended node and stop.
        if node.parent is not None and next_cursor is None:
            chain[-1] = _mark_truncated(chain[-1])
            break
        cursor = next_cursor
    return chain


def _mark_truncated(n: LineageNode) -> LineageNode:
    return LineageNode(
        name=n.name, path=n.path, parent=n.parent,
        final_steps=n.final_steps,
        parent_chain_total=n.parent_chain_total,
        version=n.version, init_mode=n.init_mode, obs_spec=n.obs_spec,
        truncated=True,
    )


def walk_chain_wandb(target_uri: str) -> list[LineageNode]:
    """Walk an artifact DAG via wandb.Api.

    Mirrors the historical scripts.lineage.walk_chain_wandb walker (which
    used a queue + visited-set) but yields LineageNode rows instead of
    plain dicts.  Returns oldest-first.
    """
    import wandb
    from scripts._artifact_io import _parse_wandb_uri, _resolve_default_entity_project

    api = wandb.Api()

    parsed = _parse_wandb_uri(target_uri)
    default_entity, default_project = _resolve_default_entity_project()
    qualified = parsed.for_api(default_entity=default_entity,
                                default_project=default_project)

    nodes: list[LineageNode] = []
    seen: set[str] = set()
    initial = api.artifact(qualified)
    queue = [initial]

    while queue:
        art = queue.pop(0)
        key = f"{art.name}:{getattr(art, 'version', '')}"
        if key in seen:
            continue
        seen.add(key)

        meta = art.metadata or {}
        run = art.logged_by()
        nodes.append(LineageNode(
            name=str(art.name).split(":", 1)[0],
            path=None,
            parent=meta.get("parent_uri"),
            final_steps=None,
            parent_chain_total=meta.get("parent_chain_total"),
            version=getattr(art, "version", None),
            init_mode=meta.get("init_mode"),
            obs_spec=meta.get("obs_spec"),
            truncated=False,
        ))

        if run is None:
            continue
        for used in run.used_artifacts():
            queue.append(used)

    nodes.reverse()
    return nodes


def _node_from_path(p: Path) -> LineageNode | None:
    cfg_path = p / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    init = cfg.get("init") if hasattr(cfg, "get") else None
    parent: str | None = None
    init_mode: str | None = None
    if init is not None and hasattr(init, "get"):
        parent_val = init.get("parent")
        parent = str(parent_val) if parent_val else None
        init_mode_val = init.get("mode")
        init_mode = str(init_mode_val) if init_mode_val else None

    obs = cfg.get("obs") if hasattr(cfg, "get") else None
    obs_spec: str | None = None
    if obs is not None and hasattr(obs, "get"):
        obs_spec_val = obs.get("name")
        obs_spec = str(obs_spec_val) if obs_spec_val else None

    meta_path = p / ".hydra" / "meta.yaml"
    meta = (OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
            if meta_path.exists() else None) or {}
    return LineageNode(
        name=str(cfg.get("run_name", p.name)),
        path=p.resolve(),
        parent=parent,
        final_steps=meta.get("final_steps"),
        parent_chain_total=meta.get("parent_chain_total"),
        version=None,
        init_mode=init_mode,
        obs_spec=obs_spec,
        truncated=False,
    )


def _next_parent_path(node: LineageNode) -> Path | None:
    if node.parent is None:
        return None
    if node.parent.startswith(("wandb://", "wandb-artifact://")):
        return None
    p = Path(node.parent)
    if p.is_file():
        p = p.parent
    return p if p.exists() else None
