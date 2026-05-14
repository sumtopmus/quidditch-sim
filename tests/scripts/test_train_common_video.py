"""build_callbacks accepts an optional video_env_fn that, when provided,
appends a VideoRecorderCallback configured from the shared
[training.callbacks.video] block."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.quidditch.simple_env import QuidditchSimpleEnv
from scripts._train_common import build_callbacks
from scripts.callbacks import VideoRecorderCallback


_BASE_CFG: dict = {
    "training": {
        "eval": {"eval_freq_steps": 1000, "n_eval_episodes": 1},
        "callbacks": {
            "checkpoint_freq_steps": 1000,
            "video_every_n_evals": 4,
            "video_fps": 20,
            "video": {
                "grid": True,
                "cells": ["south", "east", "top", "fixed"],
                "cell_width": 64,
                "cell_height": 48,
            },
        },
    }
}


def test_build_callbacks_without_video(tmp_path: Path) -> None:
    cbs = build_callbacks(
        run_dir=tmp_path,
        eval_env_fn=lambda: QuidditchSimpleEnv(),
        config=_BASE_CFG,
        n_envs=1,
    )
    assert any(isinstance(c, CheckpointCallback) for c in cbs)
    assert any(isinstance(c, EvalCallback) for c in cbs)
    assert not any(isinstance(c, VideoRecorderCallback) for c in cbs)


def test_build_callbacks_with_video(tmp_path: Path) -> None:
    cbs = build_callbacks(
        run_dir=tmp_path,
        eval_env_fn=lambda: QuidditchSimpleEnv(),
        config=_BASE_CFG,
        n_envs=1,
        video_env_fn=lambda: QuidditchSimpleEnv(render_mode="rgb_array"),
    )
    video_cbs = [c for c in cbs if isinstance(c, VideoRecorderCallback)]
    assert len(video_cbs) == 1

    vc = video_cbs[0]
    assert vc.grid is True
    assert vc.grid_cams == ("south", "east", "top", "fixed")
    assert vc.cell_width == 64
    assert vc.cell_height == 48
    assert vc.fps == 20
    # record_freq = (eval_freq_steps / n_envs) * video_every_n_evals = 1000 * 4 = 4000
    assert vc.record_freq == 4000
