"""Scripted 1v1 narratives for visual review of tag/crash/score behavior.

Two scenarios run back-to-back in a single MuJoCo viewer:
  A) Tagged out — Blue tags Red twice, then rams it down.
  B) Through despite the tag — Red gets tagged once but still scores.

Run via:  make demo  -> pick "scenarios"

NOTE: As of this commit, the scripted policies are placeholder zero-actions.
The narrative beats are tuned in subsequent commits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

# Toggle to draw colored event markers in the 3D scene (deferred — the
# helper is a no-op until SHOW_EVENT_MARKERS=True).
SHOW_EVENT_MARKERS = False

# Type alias for a single-side policy: takes that side's obs, returns its action.
Policy = Callable[[np.ndarray], np.ndarray]


def _placeholder_policy(_obs: np.ndarray) -> np.ndarray:
    """Zero-action placeholder until per-scenario policies are tuned."""
    return np.zeros(4, dtype=np.float32)


def _log_event(t: float, label: str, rew: dict[str, float],
               totals: dict[str, float]) -> None:
    """One-line terminal log for an event."""
    print(
        f"[t={t:5.2f}s] {label:<22} "
        f"red {rew['red_0']:+.3f}  blue {rew['blue_0']:+.3f}   "
        f"totals: red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _detect_label(info: dict, rew: dict[str, float]) -> str | None:
    """Return a human-readable event label for this step, or None."""
    if info["red_0"].get("scored"):
        return "SCORE"
    if info["red_0"].get("drone_drone_crash"):
        return "DRONE-DRONE CRASH"
    if info["red_0"].get("red_floor"):
        return "RED FLOOR"
    if info["red_0"].get("red_wall_crash"):
        return "RED WALL CRASH"
    if info["red_0"].get("red_oob"):
        return "RED OOB"
    if info["blue_0"].get("blue_floor"):
        return "BLUE FLOOR"
    if info["blue_0"].get("blue_wall_crash"):
        return "BLUE WALL CRASH"
    if info["blue_0"].get("blue_oob"):
        return "BLUE OOB"
    if info["red_0"].get("tag_entry"):
        return "TAG ENTRY"
    return None


def _draw_event_marker(env, kind: str) -> None:
    """No-op until SHOW_EVENT_MARKERS=True (deferred)."""
    if not SHOW_EVENT_MARKERS:
        return
    # Future: append an mjvGeom into env._world.viewer.user_scn.geoms
    # — colored sphere at the relevant body position. Kept as a stub so
    # flipping the flag elsewhere doesn't need new plumbing.
    return


def _run_scenario(name: str, env: QuidditchTeamEnv,
                   red_policy: Policy, blue_policy: Policy,
                   max_seconds: float) -> None:
    """Run one scripted scenario from current env state until termination
    or timeout; log events to terminal."""
    print(f"\n[Scenario {name}]")
    obs, _ = env.reset(seed=42)
    totals = {"red_0": 0.0, "blue_0": 0.0}
    dt_step = 1.0 / 240.0  # 240 Hz physics; adjust if env reports differently
    max_steps = int(max_seconds / dt_step)

    for step in range(max_steps):
        actions = {
            "red_0":  red_policy(obs["red_0"]),
            "blue_0": blue_policy(obs["blue_0"]),
        }
        obs, rew, term, trunc, info = env.step(actions)
        totals["red_0"]  += rew["red_0"]
        totals["blue_0"] += rew["blue_0"]

        label = _detect_label(info, rew)
        if label is not None:
            _log_event(step * dt_step, label, rew, totals)
            _draw_event_marker(env, label)

        if any(term.values()) or any(trunc.values()):
            break

    print(
        f"[Scenario {name}] FINAL  "
        f"red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _idle_pause(env: QuidditchTeamEnv, seconds: float) -> None:
    """Step the env with zero actions for `seconds`; lets the viewer breathe
    between scenarios so the user can see the final state."""
    dt_step = 1.0 / 240.0
    for _ in range(int(seconds / dt_step)):
        env.step({
            "red_0":  np.zeros(4, dtype=np.float32),
            "blue_0": np.zeros(4, dtype=np.float32),
        })


def main() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        # Scenario A — Tagged out (placeholder policies for now).
        _run_scenario(
            "A: Tagged out",
            env,
            red_policy=_placeholder_policy,
            blue_policy=_placeholder_policy,
            max_seconds=25.0,
        )
        _idle_pause(env, seconds=1.5)

        # Scenario B — Through despite the tag (placeholder policies for now).
        _run_scenario(
            "B: Through despite the tag",
            env,
            red_policy=_placeholder_policy,
            blue_policy=_placeholder_policy,
            max_seconds=10.0,
        )

        # Hold the viewer open for inspection.
        print("\nDemo complete — viewer remains open. Ctrl-C to quit.")
        while True:
            _idle_pause(env, seconds=1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
