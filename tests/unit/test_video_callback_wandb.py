"""VideoRecorderCallback emits per-cam wandb.Video objects, not TB Videos."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_log_to_wandb_emits_per_cam_video() -> None:
    """_log_to_wandb fans out to one wandb.log call with per-cam keys."""
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = True
    cb.grid = True
    cb.fps = 20
    cb.grid_cams = ("south", "east", "top", "fixed")
    cb.model = MagicMock()
    cb.model.num_timesteps = 12345

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    per_cam = {
        "south":  list(frames),
        "east":   list(frames),
        "top":    list(frames),
        "fixed":  list(frames),
    }

    with patch("wandb.log") as mock_log, patch("wandb.Video") as mock_video:
        mock_video.side_effect = lambda *a, **k: MagicMock()
        cb._log_to_wandb(per_cam)

    # Exactly one wandb.log call, with the four cam-keyed entries plus the step.
    assert mock_log.call_count == 1
    payload = mock_log.call_args.args[0] if mock_log.call_args.args else mock_log.call_args.kwargs["data"]
    keys = set(payload.keys())
    assert keys == {"eval/video/south", "eval/video/east", "eval/video/top", "eval/video/fixed"}
    # Step passed via kwarg.
    assert mock_log.call_args.kwargs["step"] == 12345


def test_log_to_wandb_single_cam_mode() -> None:
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = True
    cb.grid = False
    cb.fps = 20
    cb.grid_cams = ("south", "east", "top", "fixed")
    cb.model = MagicMock()
    cb.model.num_timesteps = 42

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    per_cam = {"fixed": frames}

    with patch("wandb.log") as mock_log, patch("wandb.Video") as mock_video:
        mock_video.side_effect = lambda *a, **k: MagicMock()
        cb._log_to_wandb(per_cam)

    payload = mock_log.call_args.args[0] if mock_log.call_args.args else mock_log.call_args.kwargs["data"]
    assert set(payload.keys()) == {"eval/video"}


def test_log_to_wandb_noop_when_moviepy_missing() -> None:
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = False
    cb.grid = True
    cb.fps = 20
    cb.grid_cams = ("south",)
    cb.model = MagicMock()

    with patch("wandb.log") as mock_log:
        cb._log_to_wandb({"south": [np.zeros((10, 10, 3), dtype=np.uint8)]})

    assert mock_log.call_count == 0
