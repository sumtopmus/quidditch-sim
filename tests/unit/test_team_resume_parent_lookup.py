"""Resolving --learner / --opponent from a parent trial's info.toml."""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_parent_trial(tmp_path: Path, *,
                       learner: str = "blue_0",
                       opponent: str = "frozen:models/red/best_model") -> Path:
    trial = tmp_path / "ppo_hoop_blue_1" / "20260507_194423"
    (trial / "checkpoints").mkdir(parents=True)
    (trial / "info.toml").write_text(
        '[run]\n'
        'name = "ppo_hoop_blue_1"\n'
        'trial = "20260507_194423"\n'
        '\n'
        '[extra]\n'
        f'learner = "{learner}"\n'
        f'opponent_spec = "{opponent}"\n'
        'warm_start_from = ""\n'
    )
    ckpt = trial / "checkpoints" / "ppo_10000000_steps.zip"
    ckpt.write_bytes(b"")  # placeholder — only path-resolution is tested here
    return ckpt


def test_lookup_returns_extra_block_values(tmp_path: Path) -> None:
    from scripts.train_team_ppo import _read_parent_extra

    ckpt = _make_parent_trial(tmp_path)
    learner, opponent = _read_parent_extra(str(ckpt))
    assert learner == "blue_0"
    assert opponent == "frozen:models/red/best_model"


def test_lookup_returns_none_if_no_info_toml(tmp_path: Path) -> None:
    from scripts.train_team_ppo import _read_parent_extra

    bare = tmp_path / "ppo_hoop_x" / "20260101_000000" / "checkpoints"
    bare.mkdir(parents=True)
    ckpt = bare / "ppo_500_steps.zip"
    ckpt.write_bytes(b"")
    learner, opponent = _read_parent_extra(str(ckpt))
    assert learner is None
    assert opponent is None
