"""Unit tests for core/training/tui_progress_callback.py — JSON schema, atomicity."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.training.tui_progress_callback import TUIProgressCallback


def _fake_model(num_timesteps: int, ep_rewards: list[float]) -> MagicMock:
    m = MagicMock()
    m.num_timesteps = num_timesteps
    # SB3 stores recent episodes here; each entry is a dict with 'r' (return) and 'l' (length)
    m.ep_info_buffer = [{"r": r, "l": 100.0} for r in ep_rewards]
    m.start_time = time.time_ns() - 60_000_000_000  # 60 seconds ago, ns precision
    return m


def test_callback_writes_schema_v1_json(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=5000, ep_rewards=[1.0, 1.2, 1.5, 1.8])
    cb._on_step()

    p = tmp_path / "tui_progress.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["schema_version"] == 1
    assert data["step"] == 5000
    assert data["total_steps"] == 1_000_000
    assert data["kind"] == "single"
    assert data["learner"] is None
    assert data["opponent"] is None
    assert data["ep_rew_mean"] == pytest.approx((1.0 + 1.2 + 1.5 + 1.8) / 4)
    assert isinstance(data["recent_rewards"], list)
    assert data["elapsed_sec"] >= 59  # ~60s


def test_callback_team_kind_records_learner_and_opponent(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=10_000_000,
        kind="team",
        learner="blue_0",
        opponent_spec="frozen:models/red_v1/best_model",
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=12345, ep_rewards=[2.0, 2.1])
    cb._on_step()

    data = json.loads((tmp_path / "tui_progress.json").read_text())
    assert data["kind"] == "team"
    assert data["learner"] == "blue_0"
    assert data["opponent"] == "frozen:models/red_v1/best_model"


def test_callback_throttles_writes_by_write_every(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1000,
    )
    p = tmp_path / "tui_progress.json"

    cb.model = _fake_model(num_timesteps=500, ep_rewards=[1.0])
    cb._on_step()
    assert not p.exists(), "should not write before write_every threshold"

    cb.model = _fake_model(num_timesteps=1000, ep_rewards=[1.0])
    cb._on_step()
    assert p.exists(), "should write at multiples of write_every"


def test_callback_atomic_write_via_temp_rename(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=100, ep_rewards=[1.0])
    cb._on_step()
    # No .tmp left behind after a successful write.
    assert not (tmp_path / "tui_progress.json.tmp").exists()
    assert (tmp_path / "tui_progress.json").exists()


def test_callback_recent_rewards_bounded_to_16(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    # Feed 25 distinct ep_rew_mean values by repeatedly stepping.
    for i in range(25):
        cb.model = _fake_model(num_timesteps=(i + 1) * 100,
                               ep_rewards=[float(i)])
        cb._on_step()

    data = json.loads((tmp_path / "tui_progress.json").read_text())
    assert len(data["recent_rewards"]) == 16
    # last value is the most recent ep_rew_mean (24.0 since the buffer has one ep with r=24)
    assert data["recent_rewards"][-1] == pytest.approx(24.0)
