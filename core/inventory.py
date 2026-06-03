"""Read-only model catalog.

Walks `models/*/` (committed-vendored) and optionally `models/.cache/*/`
(wandb downloads), reads each model's `.hydra/{config,meta}.yaml` +
`_wandb_metadata.json` via `core.run_context.load_run_context`, and returns
a sorted list of `ModelInfo` rows.

Public API:
    inventory(models_dir=Path("models"), include_cache=False) -> list[ModelInfo]
    load_model_doc(info) -> str | None
    load_run_context(info) -> dict        # re-export for callers that want the raw dict

No W&B network calls; offline-survivable.  Used by `dsim inventory`, the TUI
inventory pane, and `core.obs_compat.preflight` (parent lookup).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.run_context import load_run_context as _load_run_context

MODELS_DIR = Path("models")
_CACHE_SUBDIR = ".cache"


@dataclass(frozen=True)
class ModelInfo:
    name: str                  # e.g. "ppo_hoop_blue_4_20260511_202612"
    short_name: str            # e.g. "blue_4"
    obs_spec: str              # e.g. "DUEL_V2_WORLD"; "?" when missing
    n_stack: int               # 1 if not declared
    parent: str | None         # init.parent URI or path; None for scratch
    parent_chain_total: int    # 0 when meta.yaml missing
    final_steps: int | None
    source: Literal["vendored", "cache"]
    path: Path                 # absolute path to the model dir
    wandb_alias: str | None    # "prod" / "<run_name>" / None
    wandb_version: str | None  # "v<N>" pinned in _wandb_metadata.json
    has_model_doc: bool        # True if MODEL.md exists on disk


def inventory(
    models_dir: Path = MODELS_DIR,
    include_cache: bool = False,
) -> list[ModelInfo]:
    """Enumerate vendored (and optionally cached) promoted models."""
    models_dir = Path(models_dir)
    rows: list[ModelInfo] = []

    if models_dir.exists():
        rows.extend(_scan(models_dir, source="vendored", skip_dirs={_CACHE_SUBDIR}))

    if include_cache:
        cache_dir = models_dir / _CACHE_SUBDIR
        if cache_dir.exists():
            rows.extend(_scan(cache_dir, source="cache"))

    def _sort_key(r: ModelInfo) -> tuple[str, int]:
        ts = _trail_timestamp_int(r.name)
        return (r.short_name, -ts)

    rows.sort(key=_sort_key)
    return rows


def load_model_doc(info: ModelInfo) -> str | None:
    """Return the contents of `<model_dir>/MODEL.md`, or None if absent."""
    p = info.path / "MODEL.md"
    if not p.exists():
        return None
    return p.read_text()


def load_run_context(info: ModelInfo) -> dict[str, Any]:
    """Re-load the full `core.run_context.load_run_context` dict for an info row."""
    return _load_run_context(info.path)


def _scan(
    base: Path,
    *,
    source: Literal["vendored", "cache"],
    skip_dirs: set[str] | None = None,
) -> list[ModelInfo]:
    skip_dirs = skip_dirs or set()
    out: list[ModelInfo] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in skip_dirs:
            continue
        if not (d / ".hydra" / "config.yaml").exists():
            continue
        try:
            ctx = _load_run_context(d)
        except Exception:
            continue
        out.append(_row_from_ctx(d, ctx, source=source))
    return out


def _row_from_ctx(
    d: Path,
    ctx: dict[str, Any],
    *,
    source: Literal["vendored", "cache"],
) -> ModelInfo:
    cfg = ctx["cfg"]
    meta = ctx.get("meta") or {}
    wandb_meta = ctx.get("wandb_meta") or {}

    obs_block = cfg.get("obs") if hasattr(cfg, "get") else None
    if obs_block is None:
        obs_spec = "?"
        n_stack = 1
    else:
        obs_spec = str(obs_block.get("name", "?")) if hasattr(obs_block, "get") else "?"
        n_stack = int(obs_block.get("n_stack", 1)) if hasattr(obs_block, "get") else 1

    init_block = cfg.get("init") if hasattr(cfg, "get") else None
    parent: str | None = None
    if init_block is not None and hasattr(init_block, "get"):
        parent_val = init_block.get("parent")
        parent = str(parent_val) if parent_val else None

    return ModelInfo(
        name=d.name,
        short_name=_short_name(d.name),
        obs_spec=obs_spec,
        n_stack=n_stack,
        parent=parent,
        parent_chain_total=int(meta.get("parent_chain_total", 0) or 0),
        final_steps=(int(meta["final_steps"]) if meta.get("final_steps") is not None else None),
        source=source,
        path=d.resolve(),
        wandb_alias=wandb_meta.get("alias") if isinstance(wandb_meta, dict) else None,
        wandb_version=wandb_meta.get("version") if isinstance(wandb_meta, dict) else None,
        has_model_doc=(d / "MODEL.md").exists(),
    )


def _short_name(full: str) -> str:
    """`ppo_hoop_blue_4_20260511_202612` -> `blue_4` (drops `ppo_hoop_` prefix
    and `_YYYYMMDD_HHMMSS` trailing timestamp).
    """
    s = full
    if s.startswith("ppo_hoop_"):
        s = s[len("ppo_hoop_"):]
    parts = s.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit() \
            and len(parts[2]) == 6 and parts[2].isdigit():
        s = parts[0]
    return s


def _trail_timestamp_int(name: str) -> int:
    """Concatenated `YYYYMMDDHHMMSS` int, or 0 if not parseable."""
    parts = name.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit() \
            and len(parts[2]) == 6 and parts[2].isdigit():
        return int(parts[1] + parts[2])
    return 0
