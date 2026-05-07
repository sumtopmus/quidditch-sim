"""Scripted 1v1 takedown demo.

Blue tags Red twice (cooldown elapses between tags), then rams it down — the
drone-drone crash terminates the episode. Red runs the canonical scoring policy
(climb, push through hoop) but never reaches the hoop.

Run via:  make demo  -> pick "takedown"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from demo._team_demo_common import Policy, _delta_action, run_single_scenario_demo


def _takedown_blue_factory() -> Policy:
    """Stateful Blue:
      intercept_1 -> in_sphere_1 -> pullback -> intercept_2 -> ram

    Distances tuned against the env's tag-sphere radius (~0.2 m) and
    cooldown geometry. INSIDE = 0.18 m; OUTSIDE = 0.55 m so cooldown fully
    elapses between tags. RAM_GAIN scales the to-Red vector before
    `_delta_action` clipping, ensuring saturation throughout the ram so
    the contact velocity exceeds CRASH_VEL_THR.
    """
    INSIDE_THR  = 0.18
    OUTSIDE_THR = 0.55
    RAM_GAIN    = 5.0

    state = {"phase": "intercept_1"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept_1":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere_1"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere_1":
            if dist > INSIDE_THR + 0.05:
                state["phase"] = "pullback"
            return _delta_action(to_red)

        if state["phase"] == "pullback":
            if dist > OUTSIDE_THR:
                state["phase"] = "intercept_2"
            return _delta_action(-to_red)

        if state["phase"] == "intercept_2":
            if dist < INSIDE_THR:
                state["phase"] = "ram"
            return _delta_action(to_red)

        if state["phase"] == "ram":
            return _delta_action(to_red * RAM_GAIN)

        return np.zeros(4, dtype=np.float32)

    return policy


def main() -> None:
    run_single_scenario_demo(
        name="Takedown",
        blue_factory=_takedown_blue_factory,
        max_seconds=25.0,
    )


if __name__ == "__main__":
    main()
