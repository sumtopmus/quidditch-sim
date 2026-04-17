"""Evaluate a trained PPO model on QuidditchSimpleEnv.

Usage:
    conda activate uav
    cd quidditch-sim

    # Visual evaluation (PyBullet GUI) — 10 episodes:
    python eval_ppo.py --model runs/ppo_hoop_v1/best_model

    # Headless stats over 50 episodes:
    python eval_ppo.py --model runs/ppo_hoop_v1/best_model --no-render --episodes 50

    # Evaluate a specific checkpoint:
    python eval_ppo.py --model runs/ppo_hoop_v1/checkpoints/ppo_hoop_70000_steps
"""

import sys
from pathlib import Path

# Allow imports from the project root (envs/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO

from envs.quidditch_env import QuidditchSimpleEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO model on QuidditchSimpleEnv")
    p.add_argument(
        "--model",
        default="runs/ppo_hoop_v1/best_model",
        help="Path to model .zip (without extension)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless (no PyBullet GUI) — useful for batch stats",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions (default: True)",
    )
    p.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy actions",
    )
    return p.parse_args()


def run_episode(env: QuidditchSimpleEnv, model: PPO, deterministic: bool) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    scored = False
    crashed = False
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if info.get("scored"):
            scored = True
        if terminated and not info.get("scored"):
            crashed = True

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "scored": scored,
        "crashed": crashed,
        "steps": steps,
    }


def main() -> None:
    args = parse_args()

    model_path = args.model
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        # Try with .zip extension
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path} (tried with and without .zip)"
            )

    print(f"Model   : {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Render  : {'no (headless)' if args.no_render else 'yes (PyBullet GUI)'}")
    print(f"Policy  : {'deterministic' if args.deterministic else 'stochastic'}")
    print()

    render_mode = None if args.no_render else "human"
    env = QuidditchSimpleEnv(render_mode=render_mode)
    model = PPO.load(model_path, env=env)

    results = []
    for ep in range(1, args.episodes + 1):
        r = run_episode(env, model, args.deterministic)
        results.append(r)
        status = "SCORE" if r["scored"] else ("CRASH" if r["crashed"] else "timeout")
        print(
            f"  ep {ep:3d}/{args.episodes}  "
            f"reward={r['reward']:+7.2f}  "
            f"steps={r['steps']:5d}  "
            f"{status}"
        )

    env.close()

    # ---- aggregate stats ----
    n = len(results)
    score_rate = sum(r["scored"] for r in results) / n * 100
    crash_rate = sum(r["crashed"] for r in results) / n * 100
    timeout_rate = 100.0 - score_rate - crash_rate
    mean_reward = np.mean([r["reward"] for r in results])
    std_reward = np.std([r["reward"] for r in results])

    scored_steps = [r["steps"] for r in results if r["scored"]]
    mean_steps_to_score = np.mean(scored_steps) if scored_steps else float("nan")

    print()
    print("=" * 50)
    print(f"  Score rate   : {score_rate:5.1f}%  ({sum(r['scored'] for r in results)}/{n})")
    print(f"  Crash rate   : {crash_rate:5.1f}%  ({sum(r['crashed'] for r in results)}/{n})")
    print(f"  Timeout rate : {timeout_rate:5.1f}%")
    print(f"  Mean reward  : {mean_reward:+.2f} ± {std_reward:.2f}")
    if scored_steps:
        print(f"  Steps/score  : {mean_steps_to_score:.0f}  "
              f"({mean_steps_to_score * 0.05:.1f} s at 20 Hz)")
    print("=" * 50)


if __name__ == "__main__":
    main()
