"""Scripted 1v1 score-through-tag demo.

Blue parks just to the +y side of Red's hoop-bound path — close enough that
its tag sphere clips Red as Red flies past, far enough that the drone bodies
don't physically collide. As soon as the tag fires Blue peels further +y so
Red has a clear lane to the hoop. Demonstrates that a single tag does not
end the episode.

Run via:  make demo  -> pick "score-through-tag"
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


def _score_through_tag_blue_factory() -> Policy:
    """Stateful Blue: pre-position alongside hoop path → peel off on tag.

    Blue's intercept point is offset +y from Red's path so the tag sphere
    (radius 0.3) clips Red as Red passes but the cf2 collision hulls don't
    touch. After tag fires Blue translates further +y, clearing Red's lane to
    the hoop so Red's canonical scoring policy can finish its run.
    """
    TAG_INSIDE      = 0.25
    INTERCEPT_POINT = np.array([1.0, 0.20, 2.10], dtype=np.float64)
    PEEL_OFF_POINT  = np.array([1.0, 1.50, 2.10], dtype=np.float64)

    setpoint: np.ndarray | None = None
    state = {"phase": "pre_position"}

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal setpoint
        blue_pos = obs[9:12].astype(np.float64)
        to_red   = obs[16:19].astype(np.float64)
        dist     = float(np.linalg.norm(to_red))
        if setpoint is None:
            setpoint = blue_pos.copy()

        if state["phase"] == "pre_position":
            target = INTERCEPT_POINT.copy()
            if dist < TAG_INSIDE:
                state["phase"] = "peel_off"
        if state["phase"] == "peel_off":
            target = PEEL_OFF_POINT.copy()

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
        name="Score through tag",
        blue_factory=_score_through_tag_blue_factory,
        max_seconds=15.0,
    )


if __name__ == "__main__":
    main()
