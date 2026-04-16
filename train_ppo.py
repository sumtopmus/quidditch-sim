"""Train a PPO agent on QuidditchHoopEnv using Stable-Baselines3.

Usage:
    conda activate uav
    cd quidditch-sim
    python train_ppo.py

Checkpoints and TensorBoard logs are written to ./runs/ppo_hoop_v1/.
To watch training progress:
    tensorboard --logdir runs/ppo_hoop_v1/tb

To evaluate the best model visually (PyBullet GUI):
    python eval_ppo.py --model runs/ppo_hoop_v1/best_model
"""

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from envs.quidditch_env import QuidditchHoopEnv


# ---------------------------------------------------------------------------
# Defaults (can be overridden with CLI flags)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "run_name": "ppo_hoop_v1",
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
    "checkpoint_freq_steps": 50_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on QuidditchHoopEnv")
    p.add_argument("--run-name", default=DEFAULTS["run_name"])
    p.add_argument("--timesteps", type=int, default=DEFAULTS["total_timesteps"])
    p.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = os.path.join("runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- environments ----
    # render_mode=None for all envs during training (no GUI = much faster)
    train_env = make_vec_env(
        lambda: QuidditchHoopEnv(render_mode=None),
        n_envs=args.n_envs,
    )
    eval_env = make_vec_env(
        lambda: QuidditchHoopEnv(render_mode=None),
        n_envs=1,
    )

    # ---- callbacks ----
    # SB3 callback frequencies are in per-env steps; divide by n_envs to get
    # the correct call cadence for VecEnv (e.g. 50_000 // 4 = 12_500 calls).
    eval_freq = max(DEFAULTS["eval_freq_steps"] // args.n_envs, 1)
    checkpoint_freq = max(DEFAULTS["checkpoint_freq_steps"] // args.n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="ppo_hoop",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,         # saves best_model.zip here
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=DEFAULTS["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # ---- model ----
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_dir,
        n_steps=DEFAULTS["n_steps"],
        batch_size=DEFAULTS["batch_size"],
        n_epochs=DEFAULTS["n_epochs"],
        learning_rate=args.lr,
        gamma=DEFAULTS["gamma"],
        gae_lambda=DEFAULTS["gae_lambda"],
        clip_range=DEFAULTS["clip_range"],
        ent_coef=DEFAULTS["ent_coef"],
    )

    print(f"Training PPO for {args.timesteps:,} timesteps  ({args.n_envs} parallel envs)")
    print(f"Logs  : {tb_dir}")
    print(f"Models: {run_dir}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining done. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
