"""Train a PPO agent on QuidditchSimpleEnv using Stable-Baselines3.

Usage:
    conda activate uav
    cd repo
    python train_ppo.py                        # uses RUN_NAME=ppo_hoop
    python train_ppo.py --run-name ppo_hoop_v2

Run layout:
    runs/<run-name>/<YYYYMMDD_HHMMSS>/         ← one trial per training run
        checkpoints/
        videos/
        PPO_1/                                 ← TensorBoard event files (SB3)
        best_model.zip
        final_model.zip

To watch all training runs in TensorBoard:
    tensorboard --logdir runs

To evaluate the best model visually (MuJoCo viewer):
    python eval_ppo.py --model runs/ppo_hoop/20240416_143022/best_model
"""

import os
import sys
import shutil
import argparse
import tomllib
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Allow imports from the project root (envs/, scripts/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

# SB3 warns when train and eval vec envs are different types (SubprocVecEnv vs
# DummyVecEnv).  Using DummyVecEnv for the single-instance eval env is
# intentional — no subprocess overhead needed for n_envs=1.
warnings.filterwarnings(
    "ignore",
    message="Training and eval env are not of the same type",
    category=UserWarning,
)

# macOS conda ships multiple copies of libomp; suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from envs.quidditch.simple_env import QuidditchSimpleEnv
from core.quadrotor import CONTROL_HZ
from callbacks import VideoRecorderCallback, ResumeProgressCallback


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Config — loaded from config/training.toml at startup
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "training.toml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"{_CONFIG_PATH} not found. Run `make configs` to create it from templates/training.toml."
    )

with _CONFIG_PATH.open("rb") as _f:
    cfg = tomllib.load(_f)


def _load_env_kwargs() -> dict:
    """Build env constructor kwargs from the [env] section of config/training.toml."""
    env = cfg.get("env", {})
    out: dict = {}
    if "randomise_start" in env:
        out["randomise_start"] = bool(env["randomise_start"])
    if "episode_seconds" in env:
        out["episode_seconds"] = float(env["episode_seconds"])
    return out


# ---------------------------------------------------------------------------
# Run-info helpers — write a human-readable TOML snapshot per trial
# ---------------------------------------------------------------------------


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _write_run_info(
    path: str,
    *,
    name: str,
    trial: str,
    started: datetime,
    steps_trained: int | None = None,
    elapsed_s: float | None = None,
    best: dict | None = None,
    resume: dict | None = None,
    pretrain: dict | None = None,
) -> None:
    """Write (or overwrite) a human-readable TOML summary of this trial.

    Written once before training (elapsed/best/steps_trained left as
    "in progress") and again after (filled in).  Read by `scripts/lineage.py`
    to walk the pretrain ancestry chain.
    """
    if elapsed_s is not None:
        elapsed_line = (
            f'elapsed       = "{_fmt_elapsed(elapsed_s)}"  # {elapsed_s:.0f} s total'
        )
        finished_line = f'finished      = "{(started + timedelta(seconds=elapsed_s)).isoformat(timespec="seconds")}"'
    else:
        elapsed_line = 'elapsed       = "in progress"'
        finished_line = 'finished      = "in progress"'

    steps_line = (
        f"steps_trained = {steps_trained}"
        if steps_trained is not None
        else 'steps_trained = "in progress"'
    )

    if best:
        best_block = (
            f"\n[best]\n"
            f"mean_reward = {best['mean_reward']:.4f}\n"
            f"std_reward  = {best['std_reward']:.4f}\n"
            f"at_step     = {best['at_timestep']}\n"
        )
    else:
        best_block = "\n[best]\n# filled in after training completes\n"

    resume_block = (
        (
            f"\n[resume]\n"
            f'checkpoint  = "{resume["checkpoint"]}"\n'
            f"resumed_at  = {resume['resumed_at']}\n"
        )
        if resume
        else ""
    )

    if pretrain:
        total_line = (
            f"total_steps  = {pretrain['total_steps']}"
            if pretrain.get("total_steps") is not None
            else 'total_steps  = "in progress"'
        )
        pretrain_block = (
            f"\n[pretrain]\n"
            f"# parent path is recorded as passed to --pretrain (typically relative to repo root).\n"
            f'parent       = "{pretrain["parent"]}"\n'
            f"parent_steps = {pretrain['parent_steps']}\n"
            f"{total_line}\n"
        )
    else:
        pretrain_block = ""

    content = (
        "# Run info — written by train_ppo.py.  Read by scripts/lineage.py.\n"
        "\n"
        "[run]\n"
        f'name          = "{name}"\n'
        f'trial         = "{trial}"\n'
        f'started       = "{started.isoformat(timespec="seconds")}"\n'
        f"{elapsed_line}\n"
        f"{finished_line}\n"
        f"{steps_line}\n"
        f"{resume_block}"
        f"{pretrain_block}"
        f"{best_block}"
    )

    with open(path, "w") as fh:
        fh.write(content)


def _read_parent_chain_total(pretrain_path: str) -> int | None:
    """Read parent's [pretrain].total_steps (or [run].steps_trained as fallback).

    Returns None if no info file is found alongside the pretrain checkpoint.
    `total_steps` already reflects the parent's full ancestry, so child's
    cumulative = this_value + child's own num_timesteps.
    """
    parent_dir = Path(pretrain_path).resolve().parent
    for fname in ("info.toml", "run_info.toml"):
        candidate = parent_dir / fname
        if not candidate.exists():
            continue
        try:
            data = tomllib.loads(candidate.read_text())
        except Exception:
            return None
        total = data.get("pretrain", {}).get("total_steps")
        if isinstance(total, int):
            return total
        steps_trained = data.get("run", {}).get("steps_trained")
        if isinstance(steps_trained, int):
            return steps_trained
        return None
    return None


def _load_best_metrics(trial_dir: str) -> dict | None:
    """Read the best eval result from EvalCallback's evaluations.npz."""
    import numpy as np

    npz_path = os.path.join(trial_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path)
        means = data["results"].mean(axis=1)
        idx = int(means.argmax())
        return {
            "mean_reward": float(means[idx]),
            "std_reward": float(data["results"][idx].std()),
            "at_timestep": int(data["timesteps"][idx]),
        }
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on QuidditchSimpleEnv")
    p.add_argument(
        "--run-name",
        default=cfg["training"]["run_name"],
        help="Config/experiment label (default: from config/training.toml). A timestamp trial "
        "subdirectory is always created automatically.",
    )
    p.add_argument("--timesteps", type=int, default=cfg["training"]["total_timesteps"])
    p.add_argument("--n-envs", type=int, default=cfg["training"]["n_envs"])
    p.add_argument("--lr", type=float, default=cfg["training"]["ppo"]["lr"])
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed. Overrides config seed. Pass -1 to disable (non-deterministic).",
    )
    p.add_argument(
        "--pretrain",
        default=None,
        metavar="PATH",
        help="Path to a .zip model to warm-start from (e.g. models/20260416_190850). "
        "Loads weights + optimizer state; resets step counter.",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a checkpoint .zip to resume from. Keeps the step counter and trains "
        "for the remaining steps (total_timesteps - checkpoint_steps).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print SB3 training logs instead of showing a rich progress bar.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = 1 if args.verbose else 0
    start_time = datetime.now()
    trial = start_time.strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join("runs", args.run_name, trial)
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    video_dir = os.path.join(trial_dir, "videos")
    run_info_path = os.path.join(trial_dir, "info.toml")
    # SB3 writes TB events to trial_dir/PPO_1/ automatically — no separate tb/ subdir needed.
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- seed ----
    # CLI --seed overrides config; -1 (or absent) means non-deterministic.
    if args.seed is not None:
        seed: int | None = None if args.seed < 0 else args.seed
    else:
        raw = cfg["training"].get("seed", -1)
        seed = None if raw < 0 else raw

    # Snapshot the config file used for this trial so 'make repro' can restore it later.
    shutil.copy(_CONFIG_PATH, os.path.join(trial_dir, "config_snapshot.toml"))

    # ---- environments ----
    base_env_kwargs = _load_env_kwargs()
    # SubprocVecEnv spawns one OS process per env so physics steps run in
    # parallel across CPU cores.  Use class + env_kwargs (not a lambda) so
    # the factory is picklable under macOS's "spawn" multiprocessing start method.
    train_env = make_vec_env(
        QuidditchSimpleEnv,
        n_envs=args.n_envs,
        seed=seed,
        env_kwargs={"render_mode": None, **base_env_kwargs},
        vec_env_cls=SubprocVecEnv,
    )
    # Eval env: single instance, DummyVecEnv is sufficient (no subprocess overhead).
    eval_env = make_vec_env(
        QuidditchSimpleEnv,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": None, **base_env_kwargs},
    )

    # ---- callbacks ----
    # SB3 callback frequencies are in per-env steps; divide by n_envs to get
    # the correct call cadence for VecEnv (e.g. 50_000 // 4 = 12_500 calls).
    eval_freq = max(cfg["training"]["eval"]["eval_freq_steps"] // args.n_envs, 1)
    checkpoint_freq = max(
        cfg["training"]["callbacks"]["checkpoint_freq_steps"] // args.n_envs, 1
    )
    # Video cadence is a multiple of eval cadence; default 2 preserves the
    # legacy 100k-step interval when eval_freq_steps=50k.
    video_every_n_evals = int(
        cfg["training"]["callbacks"].get("video_every_n_evals", 2)
    )
    video_freq = eval_freq * video_every_n_evals

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="ppo_hoop",
        verbose=verbose,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=trial_dir,  # saves best_model.zip here
        log_path=trial_dir,
        eval_freq=eval_freq,
        n_eval_episodes=cfg["training"]["eval"]["n_eval_episodes"],
        deterministic=True,
        verbose=verbose,
    )
    # Video sub-section is optional in older configs; defaults match the
    # template ([training.callbacks.video] grid=true with 960x540 cells →
    # 1080p stitched, cams = south/east/top/tpv).
    video_cfg = cfg["training"]["callbacks"].get("video", {})
    video_cb = VideoRecorderCallback(
        env_fn=lambda: QuidditchSimpleEnv(render_mode="rgb_array", **base_env_kwargs),
        video_dir=video_dir,
        record_freq=video_freq,
        fps=cfg["training"]["callbacks"]["video_fps"],
        sim_hz=CONTROL_HZ,
        verbose=verbose,
        grid=video_cfg.get("grid", True),
        grid_cams=tuple(video_cfg["cells"]) if "cells" in video_cfg else None,
        cell_width=video_cfg.get("cell_width", 960),
        cell_height=video_cfg.get("cell_height", 540),
    )

    # ---- model ----
    resumed_at: int | None = None
    pretrain_parent_steps: int | None = None
    pretrain_chain_total: int | None = None
    if args.resume:
        print(f"{_ts()} ▶️  Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            verbose=verbose,
            tensorboard_log=trial_dir,
        )
        resumed_at = model.num_timesteps
        remaining = max(args.timesteps - resumed_at, 0)
        print(
            f"{_ts()} ⏱  Checkpoint at {resumed_at:,} steps; {remaining:,} remaining to {args.timesteps:,}"
        )
    elif args.pretrain:
        print(f"{_ts()} 🔄 Warm-starting from {args.pretrain}")
        model = PPO.load(
            args.pretrain,
            env=train_env,
            verbose=verbose,
            tensorboard_log=trial_dir,
            # Override hyper-params so the loaded model uses the current config
            n_steps=cfg["training"]["ppo"]["n_steps"],
            batch_size=cfg["training"]["ppo"]["batch_size"],
            n_epochs=cfg["training"]["ppo"]["n_epochs"],
            learning_rate=args.lr,
            gamma=cfg["training"]["ppo"]["gamma"],
            gae_lambda=cfg["training"]["ppo"]["gae_lambda"],
            clip_range=cfg["training"]["ppo"]["clip_range"],
            ent_coef=cfg["training"]["ppo"]["ent_coef"],
        )
        # Capture parent step count from the loaded checkpoint BEFORE training resets
        # the counter (reset_num_timesteps=True for pretrain).  Then walk parent's
        # info.toml to get the cumulative-chain total (handles deeper ancestry).
        pretrain_parent_steps = int(model.num_timesteps)
        chain_total = _read_parent_chain_total(args.pretrain)
        pretrain_chain_total = (
            chain_total if chain_total is not None else pretrain_parent_steps
        )
        print(
            f"{_ts()} ⛓  Pretrain parent has {pretrain_parent_steps:,} steps; "
            f"chain total {pretrain_chain_total:,}"
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=verbose,
            tensorboard_log=trial_dir,
            n_steps=cfg["training"]["ppo"]["n_steps"],
            batch_size=cfg["training"]["ppo"]["batch_size"],
            n_epochs=cfg["training"]["ppo"]["n_epochs"],
            learning_rate=args.lr,
            gamma=cfg["training"]["ppo"]["gamma"],
            gae_lambda=cfg["training"]["ppo"]["gae_lambda"],
            clip_range=cfg["training"]["ppo"]["clip_range"],
            ent_coef=cfg["training"]["ppo"]["ent_coef"],
            seed=seed,
        )

    resume_info = (
        {"checkpoint": args.resume, "resumed_at": resumed_at} if args.resume else None
    )
    pretrain_info = (
        {
            "parent": args.pretrain,
            "parent_steps": pretrain_parent_steps,
            "total_steps": None,  # filled in after training (= chain_total + this run's steps)
        }
        if args.pretrain
        else None
    )
    _write_run_info(
        run_info_path,
        name=args.run_name,
        trial=trial,
        started=start_time,
        resume=resume_info,
        pretrain=pretrain_info,
    )

    print(
        f"{_ts()} 🚀 Training PPO for {args.timesteps:,} timesteps  ({args.n_envs} parallel envs)"
    )
    print(f"{_ts()} 📁 Trial       : {trial_dir}")
    print(f"{_ts()} 📊 Tensorboard : {trial_dir}/PPO_1")
    print()

    extra_callbacks = (
        []
        if (args.verbose or args.resume is None)
        else [ResumeProgressCallback(args.timesteps)]
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, video_cb, *extra_callbacks],
        reset_num_timesteps=args.resume is None,
        progress_bar=not args.verbose and args.resume is None,
    )

    final_path = os.path.join(trial_dir, "final_model")
    model.save(final_path)

    elapsed_s = (datetime.now() - start_time).total_seconds()
    final_steps_trained = int(model.num_timesteps)
    if pretrain_info is not None:
        # Chain total = parent's full ancestry + this trial's actual contribution.
        # For pretrain, model.num_timesteps was reset to 0 at training start, so
        # final_steps_trained is exactly this run's contribution.
        pretrain_info["total_steps"] = (pretrain_chain_total or 0) + final_steps_trained
    _write_run_info(
        run_info_path,
        name=args.run_name,
        trial=trial,
        started=start_time,
        steps_trained=final_steps_trained,
        elapsed_s=elapsed_s,
        best=_load_best_metrics(trial_dir),
        resume=resume_info,
        pretrain=pretrain_info,
    )

    print(
        f"\n{_ts()} ✅ Training done in {_fmt_elapsed(elapsed_s)}. Final model saved to {final_path}.zip"
    )


if __name__ == "__main__":
    main()
