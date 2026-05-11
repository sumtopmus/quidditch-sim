"""Shared training infrastructure used by train_team_ppo.py.

Owns:
  - run-dir creation with timestamps
  - info.toml / config snapshot writers
  - the standard callback set (checkpoint + eval)
  - TOML config loader

Env-specific code (QuidditchSimpleEnv vs QuidditchTeamEnv + opponent) lives
in the respective entry scripts.  train_ppo.py keeps its own copies of these
helpers (untouched) so the single-agent canary commit stays bit-identical.
"""
from __future__ import annotations

import argparse
import shutil
import tomllib
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a TOML training config and return as a dict.  Raises if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.  Run `make configs` to copy templates/training.toml."
        )
    with p.open("rb") as fh:
        return tomllib.load(fh)


def make_run_dir(*, run_name: str, runs_root: str | Path = "runs") -> Path:
    """Create runs/<run_name>/<YYYYMMDD_HHMMSS>/ and return it."""
    trial = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(runs_root) / run_name / trial
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    return out


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def write_run_info(
    run_dir: Path,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    started: datetime | None = None,
    elapsed_s: float | None = None,
    steps_trained: int | None = None,
) -> None:
    """Write a human-readable info.toml + snapshot the config."""
    started = started or datetime.now()
    if elapsed_s is not None:
        elapsed_line = f'elapsed       = "{_fmt_elapsed(elapsed_s)}"'
        finished_line = (
            f'finished      = '
            f'"{(started + timedelta(seconds=elapsed_s)).isoformat(timespec="seconds")}"'
        )
    else:
        elapsed_line = 'elapsed       = "in progress"'
        finished_line = 'finished      = "in progress"'
    steps_line = (
        f"steps_trained = {steps_trained}"
        if steps_trained is not None
        else 'steps_trained = "in progress"'
    )

    extra_block = ""
    if extra:
        extra_block = "\n[extra]\n"
        for k, v in extra.items():
            if isinstance(v, str):
                extra_block += f'{k} = "{v}"\n'
            else:
                extra_block += f"{k} = {v}\n"

    resume_block = ""
    if resume:
        resume_block = (
            "\n[resume]\n"
            f'checkpoint  = "{resume["checkpoint"]}"\n'
            f'resumed_at  = {resume["resumed_at"]}\n'
        )

    content = (
        "# Run info — written by train_team_ppo.py.\n"
        "\n"
        "[run]\n"
        f'name          = "{getattr(args, "run_name", None) or run_dir.parent.name}"\n'
        f'trial         = "{run_dir.name}"\n'
        f'started       = "{started.isoformat(timespec="seconds")}"\n'
        f"{elapsed_line}\n"
        f"{finished_line}\n"
        f"{steps_line}\n"
        f"{resume_block}"
        f"{extra_block}"
    )
    (run_dir / "info.toml").write_text(content)
    cfg_path = getattr(args, "config", None)
    if cfg_path and Path(cfg_path).exists():
        shutil.copy(cfg_path, run_dir / "config_snapshot.toml")


def build_callbacks(
    *,
    run_dir: Path,
    eval_env_fn: Callable[[], Any],
    config: dict[str, Any],
    n_envs: int,
    video_env_fn: Callable[[], Any] | None = None,
    verbose: int = 0,
) -> list:
    """Build the standard SB3 callback set: checkpoint + eval + (optional) video.

    When ``video_env_fn`` is provided, a ``VideoRecorderCallback`` is appended
    using the ``[training.callbacks.video]`` sub-block.  ``video_env_fn`` should
    return an env in ``rgb_array`` mode (single-agent ``QuidditchSimpleEnv`` or
    team-agent ``OpponentControlledEnv``); the callback resets and rolls out one
    deterministic episode at every ``video_every_n_evals``-th eval trigger.

    ``verbose`` is propagated to every callback so a progress-bar run (verbose=0)
    stays silent while ``--verbose`` keeps SB3's per-eval / per-checkpoint /
    per-video chatter.
    """
    eval_freq = max(config["training"]["eval"]["eval_freq_steps"] // n_envs, 1)
    ckpt_freq = max(
        config["training"]["callbacks"]["checkpoint_freq_steps"] // n_envs, 1
    )

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    # Monitor-wrap eval env so SB3's evaluate_policy stops warning about it —
    # eval reward/length numbers stay correct either way (no other wrapper
    # mutates them) but the wrapper is the canonical fix.
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_fn())])

    cbs: list = [
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            verbose=verbose,
        ),
    ]
    # SB3 emits an unconditional UserWarning when train (SubprocVecEnv) and eval
    # (DummyVecEnv) types differ.  Intentional here — single-process eval is
    # cheaper and behaves the same.  Suppress just that warning at construction.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Training and eval env are not of the same type",
            category=UserWarning,
        )
        cbs.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=eval_freq,
                n_eval_episodes=config["training"]["eval"]["n_eval_episodes"],
                deterministic=True,
                verbose=verbose,
            )
        )

    if video_env_fn is not None:
        # Local import to avoid a hard dep on imageio/moviepy when video is off.
        from scripts.callbacks import VideoRecorderCallback
        from core.world import CONTROL_HZ

        video_cfg = config["training"]["callbacks"].get("video", {})
        video_freq = eval_freq * config["training"]["callbacks"]["video_every_n_evals"]
        cbs.append(
            VideoRecorderCallback(
                env_fn=video_env_fn,
                video_dir=str(run_dir / "videos"),
                record_freq=video_freq,
                fps=config["training"]["callbacks"]["video_fps"],
                sim_hz=CONTROL_HZ,
                grid=video_cfg.get("grid", True),
                grid_cams=tuple(video_cfg["cells"]) if "cells" in video_cfg else None,
                cell_width=video_cfg.get("cell_width", 960),
                cell_height=video_cfg.get("cell_height", 540),
                verbose=verbose,
            )
        )

    return cbs
