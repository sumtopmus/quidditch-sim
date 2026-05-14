"""StepState and RewardStack — the data + container for composable rewards.

Each reward term is an object with a `.compute(state: StepState) -> dict[str, float]`
method, returning per-agent reward deltas.  `RewardStack` runs every term per
step and sums their contributions into a per-agent total.

`StepState` is a pure data dataclass carrying all the per-step inputs terms
may need (positions, derived distances, flags from crash/score/tag detection,
fixed constants like arena_radius).  Envs build a fresh StepState each step
from their own state machines and pass it to `RewardStack.compute_step`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepState:
    """Per-step inputs for reward terms.

    Single-agent envs leave team-only fields at their defaults; team envs
    populate all fields.  Terms read only the fields they need.
    """
    # Agents present this step.  Single-agent: ("drone_0",).  Team: ("red_0", "blue_0").
    agent_ids: tuple[str, ...]

    # World-frame positions (team-only; left as zeros for single-agent).
    red_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    blue_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Derived distances.
    dist_b2r: float = 0.0           # ‖red_pos - blue_pos‖
    dist_b2r_prev: float = 0.0      # previous step's value (for closing velocity)
    step_period: float = 1 / 240.0  # dt
    dist_red_to_hoop: float = 0.0
    dist_blue_to_midpoint: float = 0.0
    dist_blue_to_hoop: float = 0.0
    dist_drone_to_hoop: float = 0.0  # single-agent only

    # Tag state machine flags.
    tag_entry: bool = False
    tag_during: bool = False

    # Score + crash flags.
    scored: bool = False
    red_floor: bool = False
    blue_floor: bool = False
    red_wall_crash: bool = False
    blue_wall_crash: bool = False
    red_oob: bool = False
    blue_oob: bool = False
    drone_drone_crash: bool = False
    drone_crash: bool = False        # single-agent: any crash terminal

    # Constants snapshotted at step time so terms don't need refs to env config.
    arena_radius: float = 3.0
    tag_radius: float = 0.3


@dataclass
class RewardStack:
    """Holds an ordered list of reward terms; accumulates per-agent rewards per step."""
    terms: list[Any]

    def compute_step(self, state: StepState) -> dict[str, float]:
        totals: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for term in self.terms:
            for agent, r in term.compute(state).items():
                totals[agent] += r
        return totals
