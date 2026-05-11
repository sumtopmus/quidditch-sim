"""Unit tests for tui/process/progress.py — tailing tui_progress.json."""
from __future__ import annotations

import json
from pathlib import Path

from tui.process.progress import read_snapshot, ProgressSnapshot, SCHEMA_VERSION_SUPPORTED


def _write_snapshot(p: Path, **overrides) -> None:
    base = {
        "schema_version": 1,
        "ts": 1.0, "run_name": "x", "trial": "y",
        "kind": "single", "learner": None, "opponent": None,
        "step": 100, "total_steps": 1000, "fps": 100.0, "elapsed_sec": 1.0,
        "ep_rew_mean": 0.5, "ep_len_mean": 200.0,
        "best_so_far": {"reward": 0.5, "step": 100},
        "recent_rewards": [0.5],
    }
    base.update(overrides)
    p.write_text(json.dumps(base))


def test_read_snapshot_returns_dataclass(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    _write_snapshot(p)
    snap = read_snapshot(p)
    assert isinstance(snap, ProgressSnapshot)
    assert snap.step == 100
    assert snap.total_steps == 1000


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_snapshot(tmp_path / "absent.json") is None


def test_partial_json_returns_none_without_raising(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    p.write_text('{"schema_version": 1, "step": ')  # truncated mid-write
    assert read_snapshot(p) is None


def test_newer_schema_version_returns_none_gracefully(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    _write_snapshot(p, schema_version=SCHEMA_VERSION_SUPPORTED + 1)
    assert read_snapshot(p) is None
