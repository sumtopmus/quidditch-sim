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
        self.pbar = tqdm(
            total=self._total, initial=self.model.num_timesteps, unit="step"
        )

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

    By default writes a 2x2 grid stitching four named cameras together at
    1920x1080.  Pass ``grid=False`` to fall back to the env's single-cam
    ``render()`` output (the cinematic "fixed" cam at 640x480).  Grid mode
    bypasses ``env.render()`` and reaches into ``env._world.render_cells(...)``
    directly — works for any env (single-agent QuidditchSimpleEnv or team
    QuidditchTeamEnv wrapped in OpponentControlledEnv) that exposes ``_world``.

    Cam choices: hoop sits at +X — south works well as the wide side view
    (broadcast convention puts the goal on the right of frame).  Either
    east or west can be paired with it: west frames the approach from the
    drone-start side, east frames the goal head-on from behind the hoop.
    north would mirror the hoop to the LEFT and is usually avoided.

    Each recorded episode is also logged to W&B.  In grid mode every cell
    becomes its own ``wandb.Video`` under ``eval/video/<cam>`` (the
    dashboard renders four independent media panels).  In single-cam mode
    there's one clip under ``eval/video``.  Either way the on-disk mp4
    still contains the stitched grid.  wandb.Video encodes mp4 in-memory
    via moviepy<2.0 (pinned because moviepy 2.x removed moviepy.editor);
    missing moviepy disables wandb uploads but on-disk mp4 writes are
    unaffected.

    Requires: pip install imageio imageio-ffmpeg "moviepy<2.0"
    """

    DEFAULT_GRID_CAMS: tuple[str, str, str, str] = (
        "south",
        "east",
        "top",
        "tpv",
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
        # imageio writes the on-disk mp4; moviepy is required by wandb.Video to
        # encode the in-memory mp4 it ships to the dashboard.
        try:
            import imageio  # noqa: F401

            self._imageio_ok = True
        except ImportError:
            self._imageio_ok = False
            print(
                f"{_ts()} ⚠️  [VideoRecorder] imageio not found. "
                "Install with: pip install imageio imageio-ffmpeg"
            )
        try:
            import moviepy  # noqa: F401

            self._moviepy_ok = True
        except ImportError:
            self._moviepy_ok = False
            print(
                f"{_ts()} ⚠️  [VideoRecorder] moviepy not found — wandb video "
                "logging disabled. Install with: pip install 'moviepy<2.0'"
            )

    def _capture_cells(self, env) -> list[np.ndarray] | None:
        """Capture per-cam frames for one timestep.

        Grid mode returns one (cell_h × cell_w × 3) array per cam in
        ``self.grid_cams``.  Single-cam mode returns a 1-element list
        wrapping ``env.render()`` (the cinematic "fixed" cam).  Stored
        unstitched so each cam can be logged to TB independently; the
        on-disk mp4 stitches them at write-time.
        """
        if self.grid:
            return env._world.render_cells(
                self.grid_cams, self.cell_width, self.cell_height
            )
        frame = env.render()
        return [frame] if frame is not None else None

    def _on_step(self) -> bool:
        if not self._imageio_ok:
            return True
        if self.n_calls % self.record_freq != 0:
            return True

        import imageio

        env = self.env_fn()
        # Per-cam streams: one list of frames per cam name.  In grid mode the
        # cam names are self.grid_cams; in single-cam mode there's just one
        # entry tagged "fixed".
        cam_keys = self.grid_cams if self.grid else ("fixed",)
        per_cam: dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}

        obs, _ = env.reset()
        done = False
        step_idx = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if step_idx % self.frame_stride == 0:
                cells = self._capture_cells(env)
                if cells is not None:
                    for name, cell in zip(cam_keys, cells):
                        per_cam[name].append(cell)
            step_idx += 1
            done = terminated or truncated

        env.close()

        n_frames = len(next(iter(per_cam.values())))
        if n_frames == 0:
            return True

        # On-disk mp4: stitch per-step cells back into the 2x2 grid (or pass
        # through the single-cam frames unchanged).
        if self.grid:
            mp4_frames = [
                self._stitch_2x2([per_cam[c][i] for c in self.grid_cams])
                for i in range(n_frames)
            ]
        else:
            mp4_frames = per_cam["fixed"]

        path = os.path.join(self.video_dir, f"step_{self.model.num_timesteps:08d}.mp4")
        with imageio.v2.get_writer(path, fps=self.fps, macro_block_size=None) as writer:
            for frame in mp4_frames:
                writer.append_data(frame)  # type: ignore[attr-defined]

        self._log_to_wandb(per_cam)

        if self.verbose:
            print(f"{_ts()} 🎬 [VideoRecorder] {n_frames} frames → {path}")
        return True

    @staticmethod
    def _stitch_2x2(cells: list[np.ndarray]) -> np.ndarray:
        """Row-major 2x2 stitch — same scheme as World.render_grid."""
        top = np.hstack([cells[0], cells[1]])
        bottom = np.hstack([cells[2], cells[3]])
        return np.vstack([top, bottom])

    def _log_to_wandb(self, per_cam: dict[str, list[np.ndarray]]) -> None:
        """Log each cam stream as a separate wandb.Video.

        Grid mode emits one entry per cam under ``eval/video/<cam>`` (with
        the cam name lowercased for tag consistency); single-cam mode emits
        one entry under ``eval/video``.  wandb.Video accepts ``(T, H, W, C)``
        uint8 arrays and encodes the mp4 in-memory.

        In WANDB_MODE=disabled, wandb.log is a no-op — this method runs but
        nothing leaves the process.  We still gate on _moviepy_ok because
        wandb.Video uses moviepy for the in-memory mp4 encode (same dep as
        the on-disk grid writer; one probe suffices).
        """
        if not self._moviepy_ok:
            return
        import wandb

        payload: dict[str, "wandb.Video"] = {}
        for name, frames in per_cam.items():
            stack = np.stack(frames)  # (T, H, W, 3) uint8
            tag = f"eval/video/{name.lower()}" if self.grid else "eval/video"
            payload[tag] = wandb.Video(stack, fps=self.fps, format="mp4")
        wandb.log(payload, step=int(self.model.num_timesteps))
