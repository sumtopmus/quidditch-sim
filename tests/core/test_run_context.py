"""Behavior contract for core.run_context.load_run_context.

Re-uses the same fixtures as tests/scripts/test_render_model_doc.py; the
old `scripts._render_model_doc._load_run_context` re-imports from core so
both call sites get the same loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _write_hydra_dir(run_dir: Path, *, cfg: dict, meta: dict | None = None) -> None:
    hdir = run_dir / ".hydra"
    hdir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), hdir / "config.yaml")
    if meta is not None:
        OmegaConf.save(OmegaConf.create(meta), hdir / "meta.yaml")


def test_load_run_context_reads_hydra_config(tmp_path: Path) -> None:
    from core.run_context import load_run_context

    _write_hydra_dir(
        tmp_path,
        cfg={"run_name": "x", "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3}},
        meta={"final_steps": 1_000_000, "parent_chain_total": 5_000_000},
    )
    ctx = load_run_context(tmp_path)

    assert ctx["cfg"]["run_name"] == "x"
    assert ctx["cfg"]["obs"]["name"] == "DUEL_V2_WORLD"
    assert ctx["meta"]["final_steps"] == 1_000_000


def test_load_run_context_missing_config_raises(tmp_path: Path) -> None:
    from core.run_context import load_run_context
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_run_context(tmp_path)


def test_load_run_context_optional_fields_default_to_none(tmp_path: Path) -> None:
    from core.run_context import load_run_context
    _write_hydra_dir(tmp_path, cfg={"run_name": "x"})
    ctx = load_run_context(tmp_path)
    assert ctx["meta"] is None
    assert ctx.get("wandb_meta") is None


def test_scripts_render_model_doc_uses_core_loader() -> None:
    """The legacy `_load_run_context` symbol on scripts._render_model_doc
    must still work — it's now a re-import from core.run_context."""
    from scripts import _render_model_doc as legacy
    from core.run_context import load_run_context
    assert legacy._load_run_context is load_run_context
