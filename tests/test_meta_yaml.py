"""Tests for meta.yaml read/write and parent-chain walking."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_write_meta_yaml_creates_file(tmp_path: Path):
    from scripts._train_common import write_meta_yaml
    run_dir = tmp_path / "runs" / "test" / "20260513_120000"
    (run_dir / ".hydra").mkdir(parents=True)
    write_meta_yaml(
        run_dir,
        git_hash="deadbeef",
        parent_chain_total=12345,
        init_mode="pretrain",
        parent_path="models/foo/best_model",
    )
    meta = yaml.safe_load((run_dir / ".hydra" / "meta.yaml").read_text())
    assert meta["git_hash"] == "deadbeef"
    assert meta["parent_chain_total"] == 12345
    assert meta["init_mode"] == "pretrain"
    assert meta["parent_path"] == "models/foo/best_model"
    assert "final_stats" not in meta  # not yet appended


def test_append_meta_yaml_final_stats(tmp_path: Path):
    from scripts._train_common import (
        write_meta_yaml, append_meta_yaml_final_stats,
    )
    run_dir = tmp_path / "runs" / "test" / "20260513_120000"
    (run_dir / ".hydra").mkdir(parents=True)
    write_meta_yaml(run_dir, git_hash="aa", parent_chain_total=0,
                     init_mode="scratch", parent_path=None)
    append_meta_yaml_final_stats(run_dir, wall_time_s=123.4,
                                   completed_steps=1_000_000,
                                   best_eval_reward=8.5)
    meta = yaml.safe_load((run_dir / ".hydra" / "meta.yaml").read_text())
    assert meta["final_stats"]["wall_time_s"] == 123.4
    assert meta["final_stats"]["completed_steps"] == 1_000_000
    assert meta["final_stats"]["best_eval_reward"] == 8.5
    # Initial fields preserved
    assert meta["git_hash"] == "aa"


def test_read_parent_chain_total_from_hydra(tmp_path: Path):
    """A 2-level chain: child reads parent's meta.yaml total, computes
    child_total = parent_total + this_run_steps."""
    from scripts._train_common import read_parent_chain_total_from_hydra
    parent_dir = tmp_path / "runs" / "parent" / "20260101_120000"
    (parent_dir / ".hydra").mkdir(parents=True)
    (parent_dir / ".hydra" / "meta.yaml").write_text(yaml.safe_dump({
        "git_hash": "p",
        "parent_chain_total": 0,
        "init_mode": "scratch",
        "parent_path": None,
        "final_stats": {"completed_steps": 5_000_000},
    }))
    # Path that looks like a "parent" reference would: best_model.zip inside parent dir.
    parent_best = parent_dir / "best_model.zip"
    parent_best.write_bytes(b"")  # dummy file
    total = read_parent_chain_total_from_hydra(str(parent_best))
    # parent contributed 5M on top of its own ancestry (0).
    assert total == 5_000_000
