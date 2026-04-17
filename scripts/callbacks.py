"""Custom SB3 callbacks for QuidditchSimpleEnv training."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


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
            print(f"{_ts()} ⚠️  [VideoRecorder] imageio not found. "
                  "Install with: pip install imageio imageio-ffmpeg")

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

        path = os.path.join(self.video_dir, f"step_{self.model.num_timesteps:08d}.mp4")
        with imageio.v2.get_writer(path, fps=self.fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]

        if self.verbose:
            print(f"{_ts()} 🎬 [VideoRecorder] {len(frames)} frames → {path}")
        return True
