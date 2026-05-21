"""Single-agent eval — Hydra entrypoint.

Usage:
    python -m scripts.eval_solo +eval_solo=default \
        eval_solo.model_uri=models/ppo_hoop_rand_start_20260505_174509/best_model \
        eval.gui=true eval.n_episodes=10
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO

from envs.quidditch.simple_env import QuidditchSimpleEnv
from scripts._artifact_io import resolve_parent


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ep = cfg.eval_solo

    # Route through resolve_parent so wandb://run:alias URIs work alongside
    # filesystem paths.
    model_path = str(resolve_parent(ep.model_uri))
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path} (tried with and without .zip)"
            )

    render_mode = "human" if bool(cfg.eval.gui) else None
    env = QuidditchSimpleEnv(render_mode=render_mode)
    model = PPO.load(model_path, env=env)

    n_eps = int(ep.get("n_episodes", cfg.eval.n_episodes))
    deterministic = bool(cfg.eval.deterministic)
    seed = int(cfg.eval.seed)

    rewards: list[float] = []
    scored_steps: list[int] = []
    n_scored = 0
    n_crashed = 0
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        ep_reward = 0.0
        steps = 0
        scored = False
        crashed = False
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += float(r)
            steps += 1
            if info.get("scored"):
                scored = True
            if terminated and not info.get("scored"):
                crashed = True
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        if scored:
            scored_steps.append(steps)
            n_scored += 1
        if crashed:
            n_crashed += 1

    n = len(rewards)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    score_rate = (n_scored / n * 100) if n else 0.0
    crash_rate = (n_crashed / n * 100) if n else 0.0
    timeout_rate = max(0.0, 100.0 - score_rate - crash_rate)

    print()
    print("=" * 50)
    print(f"  Score rate   : {score_rate:5.1f}%  ({n_scored}/{n})")
    print(f"  Crash rate   : {crash_rate:5.1f}%  ({n_crashed}/{n})")
    print(f"  Timeout rate : {timeout_rate:5.1f}%")
    print(f"  Mean reward  : {mean_reward:+.2f} ± {std_reward:.2f}")
    print(f"  n={n}  mean={mean_reward:+.4f}  std={std_reward:.4f}")
    if scored_steps:
        mean_steps_to_score = float(np.mean(scored_steps))
        print(f"  Steps/score  : {mean_steps_to_score:.0f}  "
              f"({mean_steps_to_score * 0.05:.1f} s at 20 Hz)")
    print("=" * 50)

    if bool(cfg.eval.gui) and getattr(env, "_quad", None) is not None:
        env._quad.idle()
    env.close()


if __name__ == "__main__":
    main()
