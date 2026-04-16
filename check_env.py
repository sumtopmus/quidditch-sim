"""Sanity-check QuidditchHoopEnv before committing to a full training run.

Runs three checks:
  1. stable-baselines3 check_env (obs/action shapes, dtype, step contract)
  2. One episode with a zero-action policy (confirm no crashes in env logic)
  3. One episode with a scripted "fly toward hoop" policy (confirm scoring works)

Run:
    conda activate uav
    cd quidditch-sim
    python check_env.py
"""

import sys
import numpy as np

from stable_baselines3.common.env_checker import check_env

from envs.quidditch_env import (
    QuidditchHoopEnv,
    HOOP_CENTER,
    HOOP_RADIUS,
    ARENA_RADIUS,
)


def run_sb3_check() -> None:
    print("=== [1/3] stable-baselines3 check_env ===")
    env = QuidditchHoopEnv(render_mode=None)
    check_env(env, warn=True)
    env.close()
    print("PASSED\n")


def run_zero_policy_episode() -> None:
    print("=== [2/3] Zero-action episode (10 steps) ===")
    env = QuidditchHoopEnv(render_mode=None)
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


def run_scripted_score_episode() -> None:
    """Fly directly toward the hoop center via position setpoint commands.

    This bypasses the normalized-delta action space and issues absolute
    setpoints by choosing actions that push the setpoint toward the hoop
    each step. The drone should score within a couple of hundred steps
    if the scoring geometry is correct.
    """
    print("=== [3/3] Scripted fly-toward-hoop episode ===")
    env = QuidditchHoopEnv(render_mode=None)
    obs, _ = env.reset()

    # We'll command the delta to always push setpoint toward the hoop.
    # Current setpoint starts at [0, 0, 0, 1.0]. We want to reach [40, 0, 0, 5].
    # Max delta per step: [2, 2, 0.5, 1] (for action = [1,1,1,1]).
    # So we just push action toward hoop consistently.

    scored = False
    total_reward = 0.0
    for step in range(env._max_steps):
        pos = obs[9:12]   # current drone position from obs

        # Direction from current setpoint to hoop (use drone pos as proxy)
        vec = HOOP_CENTER - pos.astype(np.float64)
        dist = np.linalg.norm(vec)

        if dist < 0.01:
            action = np.zeros(4, dtype=np.float32)
        else:
            # Normalize and clip to [-1, 1] action space
            # [dx, dy, dyaw, dz] — ignore yaw for now
            action = np.array([
                np.clip(vec[0] / 2.0, -1.0, 1.0),
                np.clip(vec[1] / 2.0, -1.0, 1.0),
                0.0,
                np.clip(vec[2] / 1.0, -1.0, 1.0),
            ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(f"  step {step:4d}  pos=({pos[0]:+5.1f},{pos[1]:+5.1f},{pos[2]:+4.1f})  "
                  f"dist={dist:5.1f}  reward={reward:+.4f}")

        if info.get("scored"):
            pos_at_score = obs[9:12]
            print(f"\n  SCORED at step {step}!  pos=({pos_at_score[0]:+.2f},{pos_at_score[1]:+.2f},{pos_at_score[2]:+.2f})")
            print(f"  total reward: {total_reward:.4f}")
            scored = True
            break

        if terminated or truncated:
            print(f"  Episode ended at step {step} without scoring: {info}")
            break

    env.close()

    if scored:
        print("PASSED (drone scored through hoop)\n")
    else:
        print("NOTE: drone did not score within the episode. Check action scaling or hoop geometry.\n")


if __name__ == "__main__":
    run_sb3_check()
    run_zero_policy_episode()
    run_scripted_score_episode()
    print("All checks complete.")
