"""Scripted 1v1 score-through-tag demo.

Blue intercepts Red and lets the tag pulse fire once, then peels off so Red is
free to continue toward the hoop and score. Demonstrates that a single tag
does not end the episode.

Run via:  make demo  -> pick "score-through-tag"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from demo._team_demo_common import Policy, _delta_action, run_single_scenario_demo


def _score_through_tag_blue_factory() -> Policy:
    """One-tag-then-peel-off:
      intercept -> in_sphere -> peel_off

    Blue closes once, lets the tag pulse fire, then retreats along
    `PEEL_OFF_DIR` so Red is free to continue toward the hoop.
    """
    INSIDE_THR   = 0.18
    PEEL_OFF_DIR = np.array([0.0, 1.5, 0.5], dtype=np.float64)  # +y mostly, slight up

    state = {"phase": "intercept"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere":
            # Hold for one step inside the sphere — tag_entry fires —
            # then retreat.
            state["phase"] = "peel_off"
            return _delta_action(to_red)

        if state["phase"] == "peel_off":
            return _delta_action(PEEL_OFF_DIR)

        return np.zeros(4, dtype=np.float32)

    return policy


def main() -> None:
    run_single_scenario_demo(
        name="Score through tag",
        blue_factory=_score_through_tag_blue_factory,
        max_seconds=15.0,
    )


if __name__ == "__main__":
    main()
