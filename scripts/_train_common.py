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
) -> list:
    """Build the standard SB3 callback set: checkpoint + eval.

    Video callback is intentionally omitted: it's coupled to the simple-env
    render plumbing.  Wire it up in the entry script if needed for the team env.
    """
    eval_freq = max(config["training"]["eval"]["eval_freq_steps"] // n_envs, 1)
    ckpt_freq = max(
        config["training"]["callbacks"]["checkpoint_freq_steps"] // n_envs, 1
    )

    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([eval_env_fn])

    return [
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=eval_freq,
            n_eval_episodes=config["training"]["eval"]["n_eval_episodes"],
            deterministic=True,
        ),
    ]
