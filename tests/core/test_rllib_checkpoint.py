"""Step-5c RLlib checkpoint-directory mechanics."""
from __future__ import annotations

from pathlib import Path

import core.rllib_checkpoint as RC


def _make_ckpt(root: Path, rel: str, modules=("main_red", "main_blue")) -> Path:
    ckpt = root / rel
    rl = ckpt / "learner_group" / "learner" / "rl_module"
    for m in modules:
        (rl / m).mkdir(parents=True)
    return ckpt


def test_find_latest_checkpoint_dir_picks_highest_ordinal(tmp_path):
    run = tmp_path / "runs" / "rllib_league" / "20260611_130631"
    _make_ckpt(run, "tune/trial_abc/checkpoint_000010")
    latest = _make_ckpt(run, "tune/trial_abc/checkpoint_000030")
    _make_ckpt(run, "tune/trial_abc/checkpoint_000020")
    assert RC.find_latest_checkpoint_dir(run) == latest.resolve()


def test_find_latest_checkpoint_dir_none_when_absent(tmp_path):
    assert RC.find_latest_checkpoint_dir(tmp_path) is None


def test_is_rllib_checkpoint_detects_module_layout(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010")
    assert RC.is_rllib_checkpoint(ckpt) is True
    assert RC.is_rllib_checkpoint(tmp_path / "nope") is False
    (tmp_path / "best_model.zip").write_bytes(b"x")
    assert RC.is_rllib_checkpoint(tmp_path / "best_model.zip") is False


def test_module_subpath(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010")
    p = RC.module_subpath(ckpt, "main_blue")
    assert p == ckpt / "learner_group" / "learner" / "rl_module" / "main_blue"
    assert p.is_dir()


def test_module_ids_lists_population(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010",
                      modules=("main_red", "main_blue", "blue_pop_v1"))
    assert RC.module_ids(ckpt) == {"main_red", "main_blue", "blue_pop_v1"}
