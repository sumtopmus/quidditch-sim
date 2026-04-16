"""Train a PPO agent on QuidditchSimpleEnv using Stable-Baselines3.

Usage:
    conda activate uav
    cd quidditch-sim
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

To evaluate the best model visually (PyBullet GUI):
    python eval_ppo.py --model runs/ppo_hoop/20240416_143022/best_model
"""

import os
import sys
import argparse
from datetime import datetime

# macOS conda ships multiple copies of libomp; suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from envs.quidditch_simple_env import QuidditchSimpleEnv
from callbacks import VideoRecorderCallback


# ---------------------------------------------------------------------------
# Defaults (can be overridden with CLI flags)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "total_timesteps": 2_000_000,
    "n_envs": 4,
    # PPO hyperparams — sensible starting point; tune after first run
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    # Evaluation during training
    "eval_freq_steps": 50_000,
    "n_eval_episodes": 10,
    # Checkpoint frequency (independent of eval)
    "checkpoint_freq_steps": 10_000,
    # Video recording frequency (one episode per checkpoint by default)
    "video_freq_steps": 10_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on QuidditchSimpleEnv")
    p.add_argument(
        "--run-name",
        default="ppo_hoop",
        help="Config/experiment label (default: ppo_hoop). A timestamp trial "
             "subdirectory is always created automatically.",
    )
    p.add_argument("--timesteps", type=int, default=DEFAULTS["total_timesteps"])
    p.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print SB3 training logs instead of showing a rich progress bar.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = 1 if args.verbose else 0
    trial = datetime.now().strftime('%Y%m%d_%H%M%S')
    trial_dir = os.path.join("runs", args.run_name, trial)
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    video_dir = os.path.join(trial_dir, "videos")
    # SB3 writes TB events to trial_dir/PPO_1/ automatically — no separate tb/ subdir needed.
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- environments ----
    # render_mode=None for all envs during training (no GUI = much faster)
    train_env = make_vec_env(
        lambda: QuidditchSimpleEnv(render_mode=None),
        n_envs=args.n_envs,
    )
    eval_env = make_vec_env(
        lambda: QuidditchSimpleEnv(render_mode=None),
        n_envs=1,
    )

    # ---- callbacks ----
    # SB3 callback frequencies are in per-env steps; divide by n_envs to get
    # the correct call cadence for VecEnv (e.g. 50_000 // 4 = 12_500 calls).
    eval_freq = max(DEFAULTS["eval_freq_steps"] // args.n_envs, 1)
    checkpoint_freq = max(DEFAULTS["checkpoint_freq_steps"] // args.n_envs, 1)
    video_freq = max(DEFAULTS["video_freq_steps"] // args.n_envs, 1)

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
        n_eval_episodes=DEFAULTS["n_eval_episodes"],
        deterministic=True,
        verbose=verbose,
    )
    video_cb = VideoRecorderCallback(
        env_fn=lambda: QuidditchSimpleEnv(render_mode="rgb_array"),
        video_dir=video_dir,
        record_freq=video_freq,
        fps=20,
        verbose=verbose,
    )

    # ---- model ----
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        tensorboard_log=trial_dir,
        n_steps=DEFAULTS["n_steps"],
        batch_size=DEFAULTS["batch_size"],
        n_epochs=DEFAULTS["n_epochs"],
        learning_rate=args.lr,
        gamma=DEFAULTS["gamma"],
        gae_lambda=DEFAULTS["gae_lambda"],
        clip_range=DEFAULTS["clip_range"],
        ent_coef=DEFAULTS["ent_coef"],
    )

    print(
        f"Training PPO for {args.timesteps:,} timesteps  ({args.n_envs} parallel envs)"
    )
    print(f"Trial : {trial_dir}")
    print(f"TB    : {trial_dir}/PPO_1  (tensorboard --logdir runs)")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, video_cb],
        progress_bar=not args.verbose,
    )

    final_path = os.path.join(trial_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining done. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
