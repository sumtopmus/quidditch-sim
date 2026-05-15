"""Walk a trial's pretrain ancestry — two walkers + CLI dispatch.

Walker A (local-only): reads `_wandb_metadata.json` chains in `models/<run>/`.
  Works offline; truncates on missing parents.

Walker B (wandb API): uses `artifact.logged_by().used_artifacts()` for the
  native artifact DAG.  Richer (sees un-vendored intermediates) but needs
  network + credentials.

CLI dispatch:
    python -m scripts.lineage <target>           # default: B, falls back to A
    python -m scripts.lineage --local <target>   # A only
    python -m scripts.lineage --both <target>    # side-by-side
Targets accepted: filesystem path (runs/.../<trial> or models/<run>),
                  wandb URI (wandb://run:alias or wandb-artifact://...).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def _load_metadata(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_uri_to_local_name(uri: str) -> str | None:
    """`wandb://x:y` → `x`; filesystem path → its parent dir name."""
    if uri.startswith(("wandb://", "wandb-artifact://")):
        # Strip the scheme and any qualifier, then split off the alias.
        body = uri.split("://", 1)[1]
        name_and_alias = body.rsplit("/", 1)[-1]
        return name_and_alias.split(":", 1)[0]
    p = Path(uri)
    # `models/<name>/best_model` or `models/<name>/`
    if "best_model" in p.name:
        return p.parent.name
    return p.name


# ── Walker A: local-only ─────────────────────────────────────────────────────
def walk_chain_local(
    start_dir: Path | str,
    models_root: Path | str = Path("models"),
) -> list[dict[str, Any]]:
    """Walk `_wandb_metadata.json` chains.  Returns oldest-first."""
    models_root = Path(models_root)
    cur = Path(start_dir).resolve()
    chain: list[dict] = []
    visited: set[Path] = set()

    while cur is not None and cur not in visited:
        visited.add(cur)
        meta = _load_metadata(cur / "_wandb_metadata.json")
        if meta is None:
            print(f"WARN: no _wandb_metadata.json at {cur} — chain truncated",
                  file=sys.stderr)
            break

        # Read .hydra for richer info (steps, obs_spec, init_mode).
        hydra_cfg_path = cur / ".hydra" / "config.yaml"
        hydra_meta_path = cur / ".hydra" / "meta.yaml"
        hydra_cfg = yaml.safe_load(hydra_cfg_path.read_text()) if hydra_cfg_path.exists() else {}
        hydra_meta = yaml.safe_load(hydra_meta_path.read_text()) if hydra_meta_path.exists() else {}
        final = hydra_meta.get("final_stats", {})

        chain.append({
            "name":      meta.get("name", cur.name),
            "version":   meta.get("version"),
            "dir":       cur,
            "init_mode": (hydra_cfg.get("init") or {}).get("mode"),
            "obs_spec":  (hydra_cfg.get("obs") or {}).get("name"),
            "steps":     final.get("completed_steps"),
            "parent_chain_total": hydra_meta.get("parent_chain_total"),
        })

        parent_uri = meta.get("parent_uri")
        if not parent_uri:
            break
        parent_name = _resolve_uri_to_local_name(parent_uri)
        if parent_name is None:
            break
        next_dir = (models_root / parent_name).resolve()
        if not next_dir.exists():
            print(f"WARN: parent {parent_uri} not in models/ — chain truncated",
                  file=sys.stderr)
            break
        cur = next_dir

    chain.reverse()
    return chain


# ── Walker B: wandb API ──────────────────────────────────────────────────────
def walk_chain_wandb(uri: str) -> list[dict[str, Any]]:
    """Walk artifact lineage via the wandb API.  Returns oldest-first."""
    import wandb
    from scripts._artifact_io import _parse_wandb_uri, _resolve_default_entity_project

    api = wandb.Api()

    chain: list[dict] = []
    seen: set[str] = set()

    # Qualify the URI with entity/project before handing to api.artifact() —
    # bare `wandb://name:alias` (or `wandb-artifact://e/p/name:alias`) is
    # not a scheme wandb understands; without qualification it falls back
    # to the workspace-default project (`uncategorized`) and silently
    # misses our artifacts.
    parsed = _parse_wandb_uri(uri)
    default_entity, default_project = _resolve_default_entity_project()
    qualified = parsed.for_api(default_entity=default_entity,
                                default_project=default_project)

    # We hold artifact objects directly (not URIs) so once we resolve the
    # entry point we can hop via `run.used_artifacts()` without going back
    # to api.artifact() — which means the walker tolerates partial mocks
    # in tests and avoids a network round-trip per node.
    initial = api.artifact(qualified)
    queue: list = [initial]

    while queue:
        art = queue.pop(0)
        key = f"{art.name}:{getattr(art, 'version', '')}"
        if key in seen:
            continue
        seen.add(key)

        meta = art.metadata or {}
        run = art.logged_by()
        run_id = getattr(run, "id", None) if run is not None else None
        chain.append({
            "name":      art.name,
            "version":   art.version,
            "init_mode": meta.get("init_mode"),
            "obs_spec":  meta.get("obs_spec"),
            "steps":     meta.get("parent_chain_total"),
            "logged_by": run_id,
        })

        if run is None:
            continue
        for used in run.used_artifacts():
            queue.append(used)

    # Oldest-first.
    chain.reverse()
    return chain


# ── Dispatch ─────────────────────────────────────────────────────────────────
def walk_dispatch(
    target: str,
    models_root: Path | str = Path("models"),
    prefer: str = "wandb",
) -> list[dict[str, Any]]:
    """Pick a walker.  prefer ∈ {"wandb", "local"}; wandb falls back to local on error."""
    is_uri = target.startswith(("wandb://", "wandb-artifact://"))

    if prefer == "local" or not is_uri:
        if is_uri:
            # Resolve URI → local dir.
            name = _resolve_uri_to_local_name(target)
            start = Path(models_root) / (name or target)
        else:
            start = Path(target)
        return walk_chain_local(start, models_root=models_root)

    try:
        return walk_chain_wandb(target)
    except Exception as e:
        print(f"WARN: wandb walk failed ({e}); falling back to local", file=sys.stderr)
        name = _resolve_uri_to_local_name(target)
        start = Path(models_root) / (name or target)
        return walk_chain_local(start, models_root=models_root)


# ── Render ───────────────────────────────────────────────────────────────────
def render(chain: list[dict]) -> str:
    """Render a chain as an aligned table.  Collapses resume segments."""
    if not chain:
        return "(empty chain)"

    # Collapse: consecutive segments with the same name + init_mode=resume
    # get merged into one row with `(resumed ×N, +M steps)`.
    collapsed: list[dict] = []
    for seg in chain:
        if (
            collapsed
            and seg["name"] == collapsed[-1]["name"]
            and seg.get("init_mode") == "resume"
        ):
            prev = collapsed[-1]
            prev["resume_count"] = prev.get("resume_count", 0) + 1
            prev["resume_steps"] = prev.get("resume_steps", 0) + (seg.get("steps") or 0)
        else:
            collapsed.append(dict(seg))

    rows: list[list[str]] = []
    for seg in collapsed:
        name = seg["name"]
        if seg.get("resume_count"):
            name = f"{name} (resumed ×{seg['resume_count']}, +{seg['resume_steps']:,} steps)"
        steps = seg.get("steps")
        steps_str = f"{steps:,}" if isinstance(steps, int) else "?"
        rows.append([
            name,
            str(seg.get("version", "")),
            str(seg.get("init_mode") or ""),
            str(seg.get("obs_spec") or ""),
            steps_str,
        ])

    headers = ["run", "version", "init", "obs", "steps"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk a run's pretrain ancestry.")
    p.add_argument("target",
                   help="filesystem path (runs/<run>/<trial> or models/<run>) "
                        "or wandb URI (wandb://run:alias)")
    p.add_argument("--local", action="store_true",
                   help="local-only walker (skip wandb API)")
    p.add_argument("--both", action="store_true",
                   help="run both walkers and print side-by-side")
    p.add_argument("--models-root", default="models")
    args = p.parse_args()

    if args.both:
        local_chain = walk_dispatch(args.target,
                                     models_root=args.models_root, prefer="local")
        wandb_chain = walk_dispatch(args.target,
                                     models_root=args.models_root, prefer="wandb")
        print("── local walker ───────────────────────────────────────")
        print(render(local_chain))
        print()
        print("── wandb walker ───────────────────────────────────────")
        print(render(wandb_chain))
        return

    prefer = "local" if args.local else "wandb"
    chain = walk_dispatch(args.target, models_root=args.models_root, prefer=prefer)
    print(render(chain))


if __name__ == "__main__":
    main()
