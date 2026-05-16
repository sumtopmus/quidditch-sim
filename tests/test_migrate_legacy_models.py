"""Tests for migrate_legacy_models.py."""
from __future__ import annotations

from pathlib import Path

import yaml


def _make_fake_promoted_model(model_dir: Path) -> None:
    """Create a model_dir with the legacy run_info.toml + config.toml shape."""
    model_dir.mkdir(parents=True)
    (model_dir / "best_model.zip").write_bytes(b"")
    (model_dir / "run_info.toml").write_text("""
[run]
name = "test_run"
trial = "20260101_120000"
started = "2026-01-01T12:00:00"
elapsed = "1h00m00s"
finished = "2026-01-01T13:00:00"
steps_trained = 10000000

[obs]
dim = 25
n_stack = 3
slots = [
  {name = "ang_vel",          dim = 3, frame = "body"},
  {name = "ang_pos",          dim = 3, frame = "body"},
  {name = "lin_vel",          dim = 3, frame = "body"},
  {name = "lin_pos",          dim = 3, frame = "world"},
  {name = "unit_to_goal",     dim = 3, frame = "world"},
  {name = "vec_to_hoop",      dim = 3, frame = "world"},
  {name = "opp_pos_rel",      dim = 3, frame = "world"},
  {name = "opp_vel_rel",      dim = 3, frame = "world"},
  {name = "closing_rate",     dim = 1},
]

[pretrain]
parent       = "models/parent/best_model"
parent_steps = 5000000
total_steps  = 15000000
""")
    (model_dir / "config.toml").write_text("""
[training]
run_name = "test_run"
total_timesteps = 10000000

[training.ppo]
lr = 3e-4
""")


def test_migrate_creates_hydra_dir(tmp_path: Path):
    from scripts.migrate_legacy_models import migrate_one
    model = tmp_path / "models" / "test_model"
    _make_fake_promoted_model(model)
    migrate_one(model)
    assert (model / ".hydra" / "config.yaml").exists()
    assert (model / ".hydra" / "meta.yaml").exists()


def test_migrate_obs_name_resolved_from_dim(tmp_path: Path):
    from scripts.migrate_legacy_models import migrate_one
    model = tmp_path / "models" / "test_model"
    _make_fake_promoted_model(model)
    migrate_one(model)
    cfg = yaml.safe_load((model / ".hydra" / "config.yaml").read_text())
    assert cfg["obs"]["name"] == "DUEL_V2_WORLD"
    assert cfg["obs"]["n_stack"] == 3


def test_migrate_idempotent(tmp_path: Path):
    from scripts.migrate_legacy_models import migrate_one
    model = tmp_path / "models" / "test_model"
    _make_fake_promoted_model(model)
    migrate_one(model)
    first_cfg = (model / ".hydra" / "config.yaml").read_text()
    migrate_one(model)
    second_cfg = (model / ".hydra" / "config.yaml").read_text()
    assert first_cfg == second_cfg


def test_migrate_meta_yaml_has_parent_chain_total(tmp_path: Path):
    from scripts.migrate_legacy_models import migrate_one
    model = tmp_path / "models" / "test_model"
    _make_fake_promoted_model(model)
    migrate_one(model)
    meta = yaml.safe_load((model / ".hydra" / "meta.yaml").read_text())
    # parent_chain_total = pretrain.total_steps = 15M
    assert meta["parent_chain_total"] == 15_000_000
    assert meta["init_mode"] == "pretrain"
    assert meta["parent_path"] == "models/parent/best_model"
