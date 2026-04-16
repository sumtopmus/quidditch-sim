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
import argparse
import tomllib
from datetime import datetime
from pathlib import Path

# macOS conda ships multiple copies of libomp; suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from envs.quidditch_simple_env import QuidditchSimpleEnv
from callbacks import VideoRecorderCallback


# ---------------------------------------------------------------------------
# Config — loaded from config/training.toml at startup
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config" / "training.toml"

with _CONFIG_PATH.open("rb") as _f:
    cfg = tomllib.load(_f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on QuidditchSimpleEnv")
    p.add_argument(
        "--run-name",
        default="ppo_hoop",
        help="Config/experiment label (default: ppo_hoop). A timestamp trial "
        "subdirectory is always created automatically.",
    )
    p.add_argument("--timesteps", type=int, default=cfg["training"]["total_timesteps"])
    p.add_argument("--n-envs", type=int, default=cfg["training"]["n_envs"])
    p.add_argument("--lr", type=float, default=cfg["ppo"]["lr"])
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print SB3 training logs instead of showing a rich progress bar.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = 1 if args.verbose else 0
    trial = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join("runs", args.run_name, trial)
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    video_dir = os.path.join(trial_dir, "videos")
    # SB3 writes TB events to trial_dir/PPO_1/ automatically — no separate tb/ subdir needed.
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- environments ----
    # SubprocVecEnv spawns one OS process per env so physics steps run in
    # parallel across CPU cores.  Use class + env_kwargs (not a lambda) so
    # the factory is picklable under macOS's "spawn" multiprocessing start method.
    train_env = make_vec_env(
        QuidditchSimpleEnv,
        n_envs=args.n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )
    # Eval env: single instance, DummyVecEnv is sufficient (no subprocess overhead).
    eval_env = make_vec_env(
        QuidditchSimpleEnv,
        n_envs=1,
        env_kwargs={"render_mode": None},
    )

    # ---- callbacks ----
    # SB3 callback frequencies are in per-env steps; divide by n_envs to get
    # the correct call cadence for VecEnv (e.g. 50_000 // 4 = 12_500 calls).
    eval_freq = max(cfg["eval"]["eval_freq_steps"] // args.n_envs, 1)
    checkpoint_freq = max(cfg["callbacks"]["checkpoint_freq_steps"] // args.n_envs, 1)
    video_freq = max(cfg["callbacks"]["video_freq_steps"] // args.n_envs, 1)

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
        n_eval_episodes=cfg["eval"]["n_eval_episodes"],
        deterministic=True,
        verbose=verbose,
    )
    video_cb = VideoRecorderCallback(
        env_fn=lambda: QuidditchSimpleEnv(render_mode="rgb_array"),
        video_dir=video_dir,
        record_freq=video_freq,
        fps=cfg["callbacks"]["video_fps"],
        verbose=verbose,
    )

    # ---- model ----
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        tensorboard_log=trial_dir,
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"]["n_epochs"],
        learning_rate=args.lr,
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
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
