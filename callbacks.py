"""Custom SB3 callbacks for QuidditchHoopEnv training."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class VideoRecorderCallback(BaseCallback):
    """Record one deterministic episode as an MP4 every *record_freq* calls.

    record_freq should be in per-env steps (divide target total steps by n_envs),
    matching the convention used for CheckpointCallback / EvalCallback.

    Requires: pip install imageio imageio-ffmpeg
    """

    def __init__(
        self,
        env_fn: Callable,
        video_dir: str,
        record_freq: int,
        fps: int = 20,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.env_fn = env_fn
        self.video_dir = video_dir
        self.record_freq = record_freq
        self.fps = fps
        os.makedirs(video_dir, exist_ok=True)

        # Lazy import so training still works if imageio is missing
        try:
            import imageio  # noqa: F401
            self._imageio_ok = True
        except ImportError:
            self._imageio_ok = False
            print(
                "[VideoRecorder] WARNING: imageio not found. "
                "Install with: pip install imageio imageio-ffmpeg"
            )

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
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated

        env.close()

        if not frames:
            return True

        total_steps = self.n_calls * self.model.n_envs
        path = os.path.join(self.video_dir, f"step_{total_steps:08d}.mp4")
        with imageio.get_writer(path, fps=self.fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)

        if self.verbose:
            print(
                f"[VideoRecorder] {len(frames)} frames → {path}"
            )
        return True
