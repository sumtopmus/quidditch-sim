"""Deterministic canary for QuidditchTeamEnv: scripted Red (bee-line to hoop)
vs scripted Blue (bee-line to midpoint).  Prints every event; stable
fingerprint locks at first run and holds thereafter unless we change physics.

Canary fingerprint (locked 2026-05-06):
    [step  176]  TAG_ENTRY                                  blue +5.019  red -5.023
    [step  684]  TERMINATED (SCORE)                         blue -10.005 red +9.999
    EPISODE END  steps=684  red_total=+1.931  blue_total=-7.618

Run:
    conda activate uav
    python scripts/check_team_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig


def beeline_action(obs: np.ndarray) -> np.ndarray:
    """Action = clip([obs[12], obs[13], 0.0, obs[14]], -1, 1).

    obs[12:15] is unit_to_goal (hoop_center for Red, midpoint for Blue).
    Maps directly onto delta-setpoint axes (dx, dy, dyaw, dz)."""
    return np.clip(
        np.array([obs[12], obs[13], 0.0, obs[14]], dtype=np.float32), -1.0, 1.0
    )


def main() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    obs, _ = env.reset(seed=42)
    print(f"[step    0]  red={obs['red_0'][9:12].round(3).tolist()}  "
          f"blue={obs['blue_0'][9:12].round(3).tolist()}")

    total = {"red_0": 0.0, "blue_0": 0.0}
    last_in_zone = False

    step = 0
    while env.agents:
        actions = {
            "red_0":  beeline_action(obs["red_0"]),
            "blue_0": beeline_action(obs["blue_0"]),
        }
        obs, rew, term, trunc, info = env.step(actions)
        step += 1
        total["red_0"]  += rew["red_0"]
        total["blue_0"] += rew["blue_0"]

        if info["red_0"].get("tag_entry"):
            print(f"[step {step:4d}]  TAG_ENTRY                                  "
                  f"blue {rew['blue_0']:+.3f}  red {rew['red_0']:+.3f}")
        elif info["red_0"].get("tag_during") and not last_in_zone:
            print(f"[step {step:4d}]  TAG_RE_ENTER (no pulse, in cooldown)        "
                  f"blue {rew['blue_0']:+.3f}  red {rew['red_0']:+.3f}")
        last_in_zone = info["red_0"].get("tag_during", False)

        terminated = any(term.values())
        truncated  = any(trunc.values())
        if terminated:
            reason = (
                "SCORE"      if info["red_0"]["scored"] else
                "DRONE_DRONE" if info["red_0"]["drone_drone_crash"] else
                "RED_FLOOR"  if info["red_0"]["red_floor"] else
                "RED_WALL"   if info["red_0"]["red_wall_crash"] else
                "RED_OOB"    if info["red_0"]["red_oob"] else
                "BLUE_FLOOR" if info["blue_0"]["blue_floor"] else
                "BLUE_WALL"  if info["blue_0"]["blue_wall_crash"] else
                "BLUE_OOB"   if info["blue_0"]["blue_oob"] else
                "?"
            )
            print(f"[step {step:4d}]  TERMINATED ({reason})                      "
                  f"blue {rew['blue_0']:+.3f}  red {rew['red_0']:+.3f}")
            break
        if truncated:
            print(f"[step {step:4d}]  TIMEOUT")
            break

    print(f"EPISODE END  steps={step}  "
          f"red_total={total['red_0']:+.3f}  blue_total={total['blue_0']:+.3f}")
    env.close()


if __name__ == "__main__":
    main()
