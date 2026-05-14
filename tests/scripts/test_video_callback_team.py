"""VideoRecorderCallback captures render cells from a team env wrapped in
OpponentControlledEnv — the same way it does for QuidditchSimpleEnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from scripts.callbacks import VideoRecorderCallback


def _make_team_env() -> OpponentControlledEnv:
    team = QuidditchTeamEnv(cfg=TeamConfig(), render_mode="rgb_array")
    return OpponentControlledEnv(
        team,
        learner_id="red_0",
        opponent=from_spec("beeline_blue"),
    )


def test_capture_cells_team_grid(tmp_path: Path) -> None:
    cb = VideoRecorderCallback(
        env_fn=_make_team_env,
        video_dir=str(tmp_path),
        record_freq=1,
        sim_hz=120,
        grid=True,
        grid_cams=("south", "east", "top", "fixed"),
        cell_width=64,
        cell_height=48,
    )
    env = _make_team_env()
    env.reset(seed=0)

    cells = cb._capture_cells(env)

    assert cells is not None
    assert len(cells) == 4
    for cell in cells:
        assert cell.shape == (48, 64, 3)
        assert cell.dtype == np.uint8
        # Renders are not all-black — every cam sees the arena.
        assert int(cell.sum()) > 0


def test_capture_cells_team_single_cam(tmp_path: Path) -> None:
    cb = VideoRecorderCallback(
        env_fn=_make_team_env,
        video_dir=str(tmp_path),
        record_freq=1,
        sim_hz=120,
        grid=False,
    )
    env = _make_team_env()
    env.reset(seed=0)

    cells = cb._capture_cells(env)

    assert cells is not None
    assert len(cells) == 1
    assert cells[0].ndim == 3 and cells[0].shape[2] == 3
    assert cells[0].dtype == np.uint8
