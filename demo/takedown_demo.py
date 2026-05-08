"""Scripted 1v1 takedown demo.

Blue parks dead-centre on Red's hoop-bound flight path. Red's body deflects
off Blue and the episode terminates without a score — typically Red wall-
crashes after the bounce. Red runs the canonical scoring policy (climb, push
through hoop) but never reaches the hoop.

Run via:  make demo  -> pick "takedown"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from demo._team_demo_common import Policy, run_single_scenario_demo

# Per-tick setpoint deltas matching ACTION_SCALE in team_env.py.
_DXY_PER_TICK = 0.2
_DZ_PER_TICK  = 0.1


def _takedown_blue_factory() -> Policy:
    """Stateful Blue: pre-position on Red's hoop path → ram on tag.

    Blue parks at INTERCEPT_POINT — between Red's rise corridor and the hoop,
    at hoop altitude. Red's canonical scoring run climbs to (0, 0, 2) then
    flies +x to the hoop, passing through Blue. When Red enters Blue's tag
    sphere we switch to the *ram* phase: Blue's setpoint follows Red, so Blue
    chases the receding Red toward the hoop. Because both drones are at
    saturation along the same axis but in opposite directions during the
    intercept window, the closing velocity easily exceeds CRASH_VEL_THR
    (1.5 m/s) and triggers a drone-drone crash before Red can reach the hoop.
    """
    # Park on Red's hoop path at Red's actual cruise altitude (Red's setpoint
    # is z=2.0 but PID overshoot pushes Red to ~z=2.10 in cruise). The static
    # block stops Red short — Red's body deflects off Blue's body and the
    # episode terminates without a score (typically Red wall-crashes after
    # the deflection).
    INTERCEPT_POINT = np.array([1.0, 0.0, 2.10], dtype=np.float64)

    setpoint: np.ndarray | None = None

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal setpoint
        blue_pos = obs[9:12].astype(np.float64)
        if setpoint is None:
            setpoint = blue_pos.copy()

        target = INTERCEPT_POINT.copy()

        delta = target - setpoint
        delta[0] = np.clip(delta[0], -_DXY_PER_TICK, _DXY_PER_TICK)
        delta[1] = np.clip(delta[1], -_DXY_PER_TICK, _DXY_PER_TICK)
        delta[2] = np.clip(delta[2], -_DZ_PER_TICK,  _DZ_PER_TICK)
        setpoint += delta

        return np.array([
            delta[0] / _DXY_PER_TICK,
            delta[1] / _DXY_PER_TICK,
            0.0,
            delta[2] / _DZ_PER_TICK,
        ], dtype=np.float32)

    return policy


def main() -> None:
    run_single_scenario_demo(
        name="Takedown",
        blue_factory=_takedown_blue_factory,
        max_seconds=25.0,
    )


if __name__ == "__main__":
    main()
