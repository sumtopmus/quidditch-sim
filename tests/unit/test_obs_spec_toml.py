"""TOML round-trip for the [obs] block in run_info.toml."""
import tomllib
from pathlib import Path

import pytest

from envs.quidditch import obs_spec
from scripts._train_common import format_obs_block, read_obs_spec


def _parse(text: str) -> dict:
    return tomllib.loads(text)


def test_format_obs_block_minimum_fields():
    text = format_obs_block(obs_spec.SIMPLE_ENV_OBS, n_stack=1)
    parsed = _parse("[run]\nname='x'\n" + text)
    assert parsed["obs"]["dim"] == 16
    assert parsed["obs"]["n_stack"] == 1
    slots = parsed["obs"]["slots"]
    assert len(slots) == 6
    assert slots[0] == {"name": "ang_vel", "dim": 3, "frame": "body"}
    last = slots[-1]
    assert last["name"] == "signed_dist_norm"
    assert "frame" not in last
    assert "notes" in last


def test_format_obs_block_augmented_with_n_stack():
    text = format_obs_block(obs_spec.AUGMENTED_OBS, n_stack=3)
    parsed = _parse("[run]\nname='x'\n" + text)
    assert parsed["obs"]["dim"] == 25
    assert parsed["obs"]["n_stack"] == 3
    assert len(parsed["obs"]["slots"]) == 9


def test_read_obs_spec_round_trip(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname='x'\n" + format_obs_block(obs_spec.TEAM_ENV_OBS, n_stack=1))
    spec, n_stack = read_obs_spec(info)
    assert spec == obs_spec.TEAM_ENV_OBS
    assert n_stack == 1


def test_read_obs_spec_round_trip_augmented(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname='x'\n" + format_obs_block(obs_spec.AUGMENTED_OBS, n_stack=3))
    spec, n_stack = read_obs_spec(info)
    assert spec == obs_spec.AUGMENTED_OBS
    assert n_stack == 3


def test_read_obs_spec_returns_none_when_block_absent(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname='x'\n")
    assert read_obs_spec(info) is None


import argparse
from datetime import datetime

from scripts._train_common import write_run_info


def test_write_run_info_emits_obs_block(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = argparse.Namespace(run_name="foo", config=None)
    write_run_info(
        run_dir,
        config={"training": {}},
        args=args,
        obs_spec=obs_spec.AUGMENTED_OBS,
        n_stack=3,
        started=datetime(2026, 5, 12, 12, 0, 0),
    )
    parsed = tomllib.loads((run_dir / "info.toml").read_text())
    assert parsed["obs"]["dim"] == 25
    assert parsed["obs"]["n_stack"] == 3
    assert len(parsed["obs"]["slots"]) == 9


def test_write_run_info_omits_obs_block_when_spec_none(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = argparse.Namespace(run_name="foo", config=None)
    write_run_info(run_dir, config={"training": {}}, args=args,
                   started=datetime(2026, 5, 12, 12, 0, 0))
    text = (run_dir / "info.toml").read_text()
    assert "[obs]" not in text
