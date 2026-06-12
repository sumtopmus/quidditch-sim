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
    dist_red_to_hoop_prev: float = 0.0   # previous step's value (progress shaping)
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

    # ── Intercept-shaping inputs ────────────────────────────────────────────
    # Populated by team_env.step when learner_id is set; both default to 0 so
    # single-agent envs and the no-learner canary path keep working.
    # future_red = red_pos + REWARD_LOOKAHEAD_S · red_vel_world
    # dist_def_to_future_red = ‖defender_pos - future_red‖
    dist_def_to_future_red:      float = 0.0
    dist_def_to_future_red_prev: float = 0.0

    # World-frame hoop centre, snapshotted at step time so reward terms don't
    # need to import constants.  Team envs populate it from HOOP_CENTER;
    # single-agent + no-team callers leave it at the zero default (unused).
    hoop_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))


# Dense SHAPING terms (annealed by dense_scale). The sparse OUTCOME terms
# (ScoreEvent, TakeDown, CrashEvent) are never annealed — they encode the true
# objective and must keep full magnitude through the dense->sparse transition.
_DENSE_TERM_TYPES = frozenset({
    "HoopApproachShaping", "HoopDistancePenalty", "HoopAnchor",
    "ZeroSumDistMirror", "InterceptShaping", "GoalSideCone",
    "ProximityGradedTag", "ClosingVelInTagZone", "TagEntryPulse",
})


@dataclass
class RewardStack:
    """Ordered reward terms; accumulates per-agent rewards per step.

    `dense_scale` (default 1.0, identity) multiplies the dense shaping terms
    only — the Step-5a dense->sparse anneal. CurriculumCallback mutates it at
    runtime via set_dense_scale. Sparse outcome terms are unaffected.
    """
    terms: list[Any]
    dense_scale: float = 1.0

    def set_dense_scale(self, scale: float) -> None:
        self.dense_scale = float(scale)

    def compute_step(self, state: StepState) -> dict[str, float]:
        totals: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for term in self.terms:
            scale = (self.dense_scale
                     if type(term).__name__ in _DENSE_TERM_TYPES else 1.0)
            for agent, r in term.compute(state).items():
                totals[agent] += scale * r
        return totals
