"""Obs-spec compatibility preflight — function-shaped, no sys.exit, no print.

Resolves a parent run's obs spec via .hydra/config.yaml (Hydra-era) WITHOUT
loading the model weights, and runs an obs-block diff against the child spec.
Returns a structured PreflightReport so callers (dsim obs-preflight CLI, the
TUI experiment picker, scripts/train.py's pretrain guard rail) can decide how
to render.

Public API:
    preflight(parent_uri, child_obs_name, child_n_stack=1) -> PreflightReport

`child_obs_name` is matched (case-insensitive) against the `name:` field of
`conf/obs/*.yaml`; that YAML's `blocks: [...]` list resolves to an ObsSpec
via `build_spec_from_block_names`.

For wandb:// URIs, downloads ONLY .hydra/ from the artifact (not the
weights) via scripts._artifact_io.resolve_parent(metadata_only=True).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from core.run_context import load_run_context
from envs.quidditch.obs_spec import (
    ObsSpec,
    build_spec_from_block_names,
)

DiffStatus = Literal["matched", "frame_changed", "removed", "added"]


@dataclass(frozen=True)
class ObsBlockDiff:
    block: str
    dim: int
    parent_frame: str | None
    child_frame: str | None
    status: DiffStatus


@dataclass(frozen=True)
class PreflightReport:
    compatible: bool                # would NOT strict-raise
    surgery_required: bool          # init.mode=warm_start would be needed
    diff: list[ObsBlockDiff]
    parent_spec_name: str
    child_spec_name: str
    parent_n_stack: int
    child_n_stack: int


def preflight(
    parent_uri: str,
    child_obs_name: str,
    child_n_stack: int = 1,
) -> PreflightReport:
    """Run obs-compat check between parent_uri and (child_obs_name, child_n_stack).

    parent_uri may be:
      - a local path to a run dir (containing .hydra/config.yaml)
      - a local path to a best_model.zip (the parent dir is used)
      - a wandb:// URI (only .hydra/ is downloaded, not weights)
    """
    parent_dir = _resolve_parent_dir_metadata_only(parent_uri)
    ctx = load_run_context(parent_dir)
    parent_cfg = ctx["cfg"]
    parent_obs = parent_cfg.get("obs") if hasattr(parent_cfg, "get") else None
    if not parent_obs:
        raise FileNotFoundError(
            f"parent {parent_dir}/.hydra/config.yaml has no `obs` block — "
            "cannot preflight"
        )

    parent_spec_name = str(parent_obs.get("name", "?"))
    parent_n_stack = int(parent_obs.get("n_stack", 1))

    parent_spec = _resolve_parent_spec(parent_obs, parent_spec_name)
    child_spec = _spec_by_name(child_obs_name)

    diff = _build_diff(parent_spec, child_spec)
    n_stack_ok = parent_n_stack == child_n_stack
    all_matched = all(d.status == "matched" for d in diff)
    compatible = n_stack_ok and all_matched
    surgery_required = not compatible

    return PreflightReport(
        compatible=compatible,
        surgery_required=surgery_required,
        diff=diff,
        parent_spec_name=parent_spec_name,
        child_spec_name=child_obs_name,
        parent_n_stack=parent_n_stack,
        child_n_stack=child_n_stack,
    )


def _resolve_parent_dir_metadata_only(parent_uri: str) -> Path:
    """Return the parent run dir, downloading only .hydra/ for wandb URIs."""
    if parent_uri.startswith(("wandb://", "wandb-artifact://")):
        from scripts._artifact_io import resolve_parent
        return resolve_parent(parent_uri, metadata_only=True)

    p = Path(parent_uri)
    # Normalize the "best_model" / "best_model.zip" / run-dir conventions.
    if p.is_file():
        p = p.parent
    elif not p.exists() and p.with_suffix(".zip").is_file():
        p = p.parent
    elif not p.exists():
        raise FileNotFoundError(f"no such parent: {p}")
    if not (p / ".hydra" / "config.yaml").exists():
        raise FileNotFoundError(f"no .hydra/config.yaml under {p}")
    return p


def _resolve_parent_spec(parent_obs, parent_spec_name: str) -> ObsSpec:
    """Build the parent's ObsSpec.

    Preference order:
      1. `parent_obs.blocks` if present (post-2026-05-18 schema).
      2. Lookup parent_spec_name in conf/obs/*.yaml by `name:` field.
    """
    blocks = parent_obs.get("blocks") if hasattr(parent_obs, "get") else None
    if blocks:
        return build_spec_from_block_names(list(blocks))
    return _spec_by_name(parent_spec_name)


def _spec_by_name(name: str) -> ObsSpec:
    """Find the conf/obs/*.yaml whose `name:` field matches (case-insensitive)
    and build its ObsSpec.  Raises KeyError if no match."""
    repo_root = Path(__file__).resolve().parents[1]
    obs_dir = repo_root / "conf" / "obs"
    if not obs_dir.exists():
        raise KeyError(f"no conf/obs/ directory under {repo_root}")
    target = name.upper()
    for yaml_path in sorted(obs_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text()) or {}
        if str(data.get("name", "")).upper() == target:
            blocks = data.get("blocks")
            if not blocks:
                raise KeyError(
                    f"{yaml_path} has no `blocks:` field — cannot resolve {name!r}"
                )
            return build_spec_from_block_names(list(blocks))
    available = sorted(
        str((yaml.safe_load(p.read_text()) or {}).get("name", ""))
        for p in obs_dir.glob("*.yaml")
    )
    raise KeyError(f"unknown obs spec name {name!r} (known: {available})")


def _build_diff(parent: ObsSpec, child: ObsSpec) -> list[ObsBlockDiff]:
    """Column-by-column block alignment between two specs.

    Walks both specs in parallel.  Same-name-same-dim-same-frame is `matched`;
    same-name with different dim or frame is `frame_changed`; child-only is
    `added`; parent-only is `removed`.
    """
    out: list[ObsBlockDiff] = []
    p_by_name = {b.name: b for b in parent.blocks}
    c_by_name = {b.name: b for b in child.blocks}

    seen: set[str] = set()
    for cb in child.blocks:
        pb = p_by_name.get(cb.name)
        if pb is None:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=None, child_frame=cb.frame,
                status="added",
            ))
            continue
        seen.add(cb.name)
        if pb.dim != cb.dim or pb.frame != cb.frame:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=pb.frame, child_frame=cb.frame,
                status="frame_changed",
            ))
        else:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=pb.frame, child_frame=cb.frame,
                status="matched",
            ))
    for pb in parent.blocks:
        if pb.name in seen:
            continue
        if pb.name not in c_by_name:
            out.append(ObsBlockDiff(
                block=pb.name, dim=pb.dim,
                parent_frame=pb.frame, child_frame=None,
                status="removed",
            ))
    return out


def obs_modes_compatible(parent_mode: str, child_mode: str) -> bool:
    """flat↔dict is never compatible: a CTDE (dict) run can only init=scratch.

    Same-mode is compatible at this coarse level; block-level compat is still
    judged by preflight() for flat↔flat, and (future) actor-spec-only for
    dict↔dict."""
    return parent_mode == child_mode
