"""Custom SB3 callbacks for QuidditchSimpleEnv training."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Video


class ResumeProgressCallback(BaseCallback):
    """Progress bar that shows absolute step counts when resuming (e.g. 10.8M → 20M)."""

    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self._total = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        try:
            from tqdm.rich import tqdm
        except ImportError:
            from tqdm import tqdm
        self.pbar = tqdm(total=self._total, initial=self.model.num_timesteps, unit="step")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.n = self.model.num_timesteps
            self.pbar.update(0)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.n = self._total
            self.pbar.update(0)
            self.pbar.close()


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


class VideoRecorderCallback(BaseCallback):
    """Record one deterministic episode as an MP4 every *record_freq* calls.

    record_freq should be in per-env steps (divide target total steps by n_envs),
    matching the convention used for CheckpointCallback / EvalCallback.

    By default writes a 2x2 grid stitching four named cameras together
    (South / East / Top / chase-cam) at 1920x1080.  Pass ``grid=False`` to
    fall back to the env's single-cam ``render()`` output (the cinematic
    "Fixed" cam at 640x480).  Grid mode bypasses ``env.render()`` and reaches
    into ``env._quad.render_grid(...)`` directly — same precedent as
    eval_ppo.py reaching into ``env._quad`` for the live viewer.

    Cam choices: hoop sits at +X — South works well as the wide side view
    (broadcast convention puts the goal on the right of frame).  Either
    East or West can be paired with it: West frames the approach from the
    drone-start side, East frames the goal head-on from behind the hoop.
    North would mirror the hoop to the LEFT and is usually avoided.

    Each recorded episode is also logged to TensorBoard under ``eval/video``
    via SB3's ``Video`` log type — appears in TB's "Images" tab.  TB embedding
    needs ``moviepy`` (used by torch's SummaryWriter.add_video); if it's
    missing the mp4 still gets written and we just warn once.

    Requires: pip install imageio imageio-ffmpeg moviepy
    """

    DEFAULT_GRID_CAMS: tuple[str, str, str, str] = (
        "South", "East", "Top", "drone_tpv",
    )

    def __init__(
        self,
        env_fn: Callable,
        video_dir: str,
        record_freq: int,
        fps: int = 20,
        sim_hz: int = 120,
        verbose: int = 1,
        *,
        grid: bool = True,
        grid_cams: tuple[str, str, str, str] | None = None,
        cell_width: int = 960,
        cell_height: int = 540,
    ) -> None:
        super().__init__(verbose)
        self.env_fn = env_fn
        self.video_dir = video_dir
        self.record_freq = record_freq
        self.fps = fps
        self.frame_stride = max(1, round(sim_hz / fps))
        self.grid = grid
        self.grid_cams = tuple(grid_cams) if grid_cams else self.DEFAULT_GRID_CAMS
        self.cell_width = cell_width
        self.cell_height = cell_height
        os.makedirs(video_dir, exist_ok=True)

        # Lazy imports so training still works if optional video deps are missing.
        # imageio writes the on-disk mp4; moviepy is required by torch's
        # SummaryWriter.add_video to embed the video in TB events.
        try:
            import imageio  # noqa: F401

            self._imageio_ok = True
        except ImportError:
            self._imageio_ok = False
            print(f"{_ts()} ⚠️  [VideoRecorder] imageio not found. "
                  "Install with: pip install imageio imageio-ffmpeg")
        try:
            import moviepy  # noqa: F401

            self._moviepy_ok = True
        except ImportError:
            self._moviepy_ok = False
            print(f"{_ts()} ⚠️  [VideoRecorder] moviepy not found — TB video "
                  "logging disabled. Install with: pip install moviepy")

    def _capture_frame(self, env) -> np.ndarray | None:
        """Render one frame in the configured mode (single-cam or 2x2 grid)."""
        if self.grid:
            return env._quad.render_grid(
                self.grid_cams, self.cell_width, self.cell_height
            )
        return env.render()

    def _on_step(self) -> bool:
        if not self._imageio_ok:
            return True
        if self.n_calls % self.record_freq != 0:
            return True

        import imageio

        env = self.env_fn()
        frames: list[np.ndarray] = []

        obs, _ = env.reset()
        done = False
        step_idx = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if step_idx % self.frame_stride == 0:
                frame = self._capture_frame(env)
                if frame is not None:
                    frames.append(frame)
            step_idx += 1
            done = terminated or truncated

        env.close()

        if not frames:
            return True

        path = os.path.join(self.video_dir, f"step_{self.model.num_timesteps:08d}.mp4")
        with imageio.v2.get_writer(path, fps=self.fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]

        self._log_to_tensorboard(frames)

        if self.verbose:
            print(f"{_ts()} 🎬 [VideoRecorder] {len(frames)} frames → {path}")
        return True

    def _log_to_tensorboard(self, frames: list[np.ndarray]) -> None:
        """Embed the recorded episode in TB under ``eval/video``.

        SB3's ``Video`` wraps a (N, T, C, H, W) uint8 tensor; the underlying
        torch ``SummaryWriter.add_video`` swallows missing-moviepy as a print
        and returns None, so we gate on the init-time probe.
        """
        if not self._moviepy_ok:
            return
        import torch

        # (T, H, W, 3) uint8  →  (1, T, 3, H, W) uint8
        tensor = torch.from_numpy(
            np.stack(frames).transpose(0, 3, 1, 2)[None].copy()
        )
        self.logger.record(
            "eval/video",
            Video(tensor, fps=self.fps),
            exclude=("stdout", "log", "json", "csv"),
        )
        # Force-flush so the video shows up at this exact timestep instead
        # of waiting for the next rollout-end dump.
        self.logger.dump(self.model.num_timesteps)
