"""Sanity-check QuidditchSimpleEnv before committing to a full training run.

Headless mode (default) runs three checks:
  1. stable-baselines3 check_env (obs/action shapes, dtype, step contract)
  2. One episode with a zero-action policy (confirm no crashes in env logic)
  3. One episode with a scripted "fly toward hoop" policy (confirm scoring works)

Viewer mode (--viewer) opens a MuJoCo interactive window and runs only the
scripted flight so you can watch the drone.

Run:
    conda activate uav
    cd quidditch-sim

    # Headless — fast, no window (default):
    python scripts/check_env.py

    # Viewer — opens MuJoCo window; only the scripted flight runs:
    python scripts/check_env.py --viewer
"""

import sys
from pathlib import Path

# Allow imports from the project root (envs/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from stable_baselines3.common.env_checker import check_env

from envs.quidditch_simple_env import (
    QuidditchSimpleEnv,
    HOOP_CENTER,
    HOOP_RADIUS,
    ARENA_RADIUS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check QuidditchSimpleEnv")
    p.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo viewer window — orbit/pan/zoom with the mouse",
    )
    return p.parse_args()


def run_sb3_check(render_mode: str | None = None) -> None:
    print("=== [1/3] stable-baselines3 check_env ===")
    env = QuidditchSimpleEnv(render_mode=render_mode, randomise_start=False)
    check_env(env, warn=True)
    env.close()
    print("PASSED\n")


def run_zero_policy_episode(render_mode: str | None = None) -> None:
    print("=== [2/3] Zero-action episode (10 steps) ===")
    env = QuidditchSimpleEnv(render_mode=render_mode, randomise_start=False)
    obs, _ = env.reset()
    print(f"  initial obs shape : {obs.shape}, dtype: {obs.dtype}")
    print(f"  obs sample        : {obs}")

    total_reward = 0.0
    for step in range(10):
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 5 == 0:
            pos = obs[9:12]
            print(f"  step {step:3d}  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                  f"reward={reward:+.4f}  terminated={terminated}")
        if terminated or truncated:
            print(f"  episode ended early at step {step}: {info}")
            break

    print(f"  total reward: {total_reward:.4f}")
    env.close()
    print("PASSED\n")


def run_scripted_score_episode(render_mode: str | None = None) -> None:
    """Fly directly toward the hoop center via position setpoint commands."""
    print("=== [3/3] Scripted fly-toward-hoop episode ===")
    env = QuidditchSimpleEnv(render_mode=render_mode, randomise_start=False)
    obs, _ = env.reset()

    scored = False
    total_reward = 0.0
    for step in range(env._max_steps):
        pos = obs[9:12]

        vec = HOOP_CENTER - pos.astype(np.float64)
        dist = np.linalg.norm(vec)

        if dist < 0.01:
            action = np.zeros(4, dtype=np.float32)
        else:
            action = np.array([
                np.clip(vec[0] / 0.2, -1.0, 1.0),
                np.clip(vec[1] / 0.2, -1.0, 1.0),
                0.0,
                np.clip(vec[2] / 0.1, -1.0, 1.0),
            ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(f"  step {step:4d}  pos=({pos[0]:+5.1f},{pos[1]:+5.1f},{pos[2]:+4.1f})  "
                  f"dist={dist:5.2f}  reward={reward:+.4f}")

        if info.get("scored"):
            pos_at_score = obs[9:12]
            print(f"\n  SCORED at step {step}!  "
                  f"pos=({pos_at_score[0]:+.2f},{pos_at_score[1]:+.2f},{pos_at_score[2]:+.2f})")
            print(f"  total reward: {total_reward:.4f}")
            scored = True
            break

        if terminated or truncated:
            print(f"  Episode ended at step {step} without scoring: {info}")
            break

    if scored and render_mode == "human":
        assert env._quad is not None
        hover_seconds = 60
        hover_steps = int(hover_seconds / env._quad.step_period)
        print(f"  Hovering for {hover_seconds}s ({hover_steps} steps) — orbit/pan/zoom the window.")
        zero_action = np.zeros(4, dtype=np.float32)
        for _ in range(hover_steps):
            env.step(zero_action)

    env.close()

    if scored:
        print("PASSED (drone scored through hoop)\n")
    else:
        print("NOTE: drone did not score within the episode. Check action scaling or hoop geometry.\n")


if __name__ == "__main__":
    args = parse_args()

    if args.viewer:
        print("Viewer mode — MuJoCo window will open.")
        print("  Orbit: left-drag   Pan: right-drag   Zoom: scroll\n")
        run_scripted_score_episode(render_mode="human")
    else:
        run_sb3_check()
        run_zero_policy_episode()
        run_scripted_score_episode()
        print("All checks complete.")
