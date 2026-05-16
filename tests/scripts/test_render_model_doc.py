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


from scripts._render_model_doc import _section_header


def _ctx_for_section(**overrides) -> dict:
    """Build a baseline ctx covering the union of fields all sections need.
    Tests override just what they care about.
    """
    cfg = OmegaConf.create({
        "run_name": "ppo_hoop_test",
        "description": "",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
        "init": {"mode": "scratch", "parent": None},
        "trainer": {"lr": 3e-4, "total_timesteps": 10_000_000,
                     "n_envs": 8, "batch_size": 256, "n_epochs": 10,
                     "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0,
                     "clip_range": 0.2},
        "env": {"learner_id": "blue_0",
                 "team_cfg": {"episode_seconds": 30.0, "tag_radius": 0.3,
                              "crash_vel_thr": 1.0, "midpoint_alpha": 0.5,
                              "crash_aftermath_seconds": 0.0}},
        "opponent": {"_target_": "envs.quidditch.opponents.BeelineRed"},
        "curriculum": {"name": "fixed_start"},
        "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                    "terms": []},  # empty terms ok for non-reward sections
    })
    ctx = {
        "cfg": cfg,
        "meta": {"git_hash": "abc1234", "final_stats": {
            "best_eval_reward": 7.91, "best_step": 9_500_000,
            "completed_steps": 10_000_000, "wall_clock_seconds": 1923.0,
            "model_kind": "best"}},
        "hydra_yaml": {"hydra": {"runtime": {"choices": {"reward": "team_v2"}}}},
        "wandb_meta": {"name": "ppo_hoop_test", "version": "v0",
                        "aliases": ["latest", "prod", "ppo_hoop_test"],
                        "entity": "gridcom", "project": "drone-quidditch"},
        "run_dir": Path("runs/ppo_hoop_test/20260516_120000"),
    }
    ctx.update(overrides)
    return ctx


def test_section_header_promoted_run():
    """When _wandb_metadata.json is present, status is 'promoted' and the W&B
    line renders below the header."""
    out = _section_header(_ctx_for_section())
    assert "# MODEL: ppo_hoop_test_20260516_120000" in out
    assert "promoted" in out
    assert "abc1234" in out
    assert "wandb://ppo_hoop_test:prod" in out
    assert "v0" in out


def test_section_header_run_only_when_wandb_meta_absent():
    """When _wandb_metadata.json is None, status is 'run-only' and the W&B
    line is omitted entirely."""
    out = _section_header(_ctx_for_section(wandb_meta=None))
    assert "run-only" in out
    assert "wandb://" not in out
    assert "**W&B:**" not in out
