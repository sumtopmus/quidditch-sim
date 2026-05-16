"""Tests for scripts/_render_model_doc.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts._render_model_doc import _load_run_context


def _write_run_fixture(
    run_dir: Path,
    *,
    config: dict,
    meta: dict | None = None,
    hydra_yaml: dict | None = None,
    wandb_meta: dict | None = None,
) -> Path:
    """Write a minimal run-dir fixture under `run_dir/.hydra/`."""
    hdir = run_dir / ".hydra"
    hdir.mkdir(parents=True, exist_ok=True)
    (hdir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    if meta is not None:
        (hdir / "meta.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(meta)))
    if hydra_yaml is not None:
        (hdir / "hydra.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(hydra_yaml)))
    if wandb_meta is not None:
        (run_dir / "_wandb_metadata.json").write_text(json.dumps(wandb_meta))
    return run_dir


def test_load_run_context_reads_all_inputs(tmp_path: Path):
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={"run_name": "ppo_hoop_test", "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1}},
        meta={"git_hash": "abc123", "final_stats": {"best_eval_reward": 7.91}},
        hydra_yaml={"hydra": {"runtime": {"choices": {"reward": "team_v2"}}}},
        wandb_meta={"name": "ppo_hoop_test", "version": "v0", "aliases": ["prod"]},
    )
    ctx = _load_run_context(run_dir)
    assert ctx["cfg"].run_name == "ppo_hoop_test"
    assert ctx["meta"]["git_hash"] == "abc123"
    assert ctx["hydra_yaml"]["hydra"]["runtime"]["choices"]["reward"] == "team_v2"
    assert ctx["wandb_meta"]["name"] == "ppo_hoop_test"
    assert ctx["run_dir"] == run_dir


def test_load_run_context_raises_when_config_missing(tmp_path: Path):
    """config.yaml is the required input — everything depends on it."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_run_context(run_dir)


def test_load_run_context_tolerates_missing_optional_inputs(tmp_path: Path):
    """meta.yaml, hydra.yaml, _wandb_metadata.json are all optional."""
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={"run_name": "ppo_hoop_test", "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1}},
    )
    ctx = _load_run_context(run_dir)
    assert ctx["cfg"].run_name == "ppo_hoop_test"
    assert ctx["meta"] is None
    assert ctx["hydra_yaml"] is None
    assert ctx["wandb_meta"] is None
