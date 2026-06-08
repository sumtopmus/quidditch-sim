"""Composable reward terms — one dataclass per reward signal.

Every term has a `.compute(state: StepState) -> dict[str, float]` method
returning per-agent reward deltas for the current step.  Terms are pure:
they read from `state` and produce a dict; they hold no internal state.

Naming convention: events (one-shot, fire on a flag) use `Event` suffix;
continuous (per-step shaping) terms use a descriptive noun.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.quidditch.rewards.stack import StepState


@dataclass
class ScoreEvent:
    """+magnitude to scorer when `state.scored` is True; mirror to opponent.

    Team mode: `scorer="red_0"`, `zero_sum_opponent="blue_0"` (blue gets
    −magnitude when red scores).  Single-agent: `scorer="drone_0"`,
    `zero_sum_opponent=None`.
    """
    magnitude: float
    scorer: str
    zero_sum_opponent: str | None = None

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.scored:
            out[self.scorer] += self.magnitude
            if self.zero_sum_opponent is not None:
                out[self.zero_sum_opponent] -= self.magnitude
        return out


@dataclass
class CrashEvent:
    """`magnitude` (typically negative) to each agent whose crash flags fire.

    `agent_to_crash_flags`: maps an agent id to the names of the StepState
    fields whose truthiness triggers the penalty for that agent (any True
    fires once; the penalty does not stack within one step).
    """
    magnitude: float
    agent_to_crash_flags: dict[str, tuple[str, ...]]

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for agent, flag_names in self.agent_to_crash_flags.items():
            if agent not in out:
                continue
            if any(getattr(state, fn) for fn in flag_names):
                out[agent] += self.magnitude
        return out


# Field-name lookup for the "target" string in HoopDistancePenalty.
_TARGET_FIELDS: dict[str, str] = {
    "hoop":       "dist_red_to_hoop",
    "midpoint":   "dist_blue_to_midpoint",
    "drone_hoop": "dist_drone_to_hoop",
}


@dataclass
class HoopDistancePenalty:
    """`-(dist_to_target / arena_radius) * scale` per agent each step.

    `agent_to_target`: maps an agent id to a target name from `_TARGET_FIELDS`.
    Each agent uses its own (agent-specific) distance from StepState.

    Used three ways:
      - team red: `{"red_0": "hoop"}`
      - team blue: `{"blue_0": "midpoint"}` (midpoint shaping)
      - single agent: `{"drone_0": "drone_hoop"}`
    """
    scale: float
    agent_to_target: dict[str, str]

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for agent, target in self.agent_to_target.items():
            if agent not in out:
                continue
            dist = getattr(state, _TARGET_FIELDS[target])
            out[agent] -= (dist / state.arena_radius) * self.scale
        return out


@dataclass
class HoopApproachShaping:
    """Potential-based progress shaping toward the hoop for one agent.

    `+scale * (dist_red_to_hoop_prev - dist_red_to_hoop)` — positive when the
    agent closes on the hoop this step, negative when it recedes.  This is the
    classic potential-based form (Φ = -distance), so it is policy-invariant:
    circling or camping nets ~zero, and the cumulative reward over an episode
    telescopes to `scale * (dist_initial - dist_final)`.  That avoids the
    camping hack a raw proximity bonus would create here (scoring terminates
    the episode, so a per-step "be near the hoop" reward would pay an agent to
    hover next to the hoop forever instead of flying through it).

    Reuses StepState.dist_red_to_hoop / dist_red_to_hoop_prev for any agent id
    (the field names are red-flavored but carry whichever learner's hoop
    distance the env populates).
    """
    scale: float
    agent: str = "red_0"

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if self.agent in out:
            progress = state.dist_red_to_hoop_prev - state.dist_red_to_hoop
            out[self.agent] = self.scale * progress
        return out


@dataclass
class HoopAnchor:
    """`-(dist_blue_to_hoop / arena_radius) * scale` for each configured agent.

    Conventionally Blue-only — keeps the defender near the hoop regardless of
    Red's position.
    """
    scale: float
    agents: tuple[str, ...] = ("blue_0",)

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for agent in self.agents:
            if agent not in out:
                continue
            out[agent] -= (state.dist_blue_to_hoop / state.arena_radius) * self.scale
        return out


@dataclass
class ZeroSumDistMirror:
    """`+(dist_red_to_hoop / arena_radius) * scale` for each configured agent.

    Conventionally Blue-only — defender is rewarded for keeping Red far from
    the hoop.  Same magnitude as Red's `HoopDistancePenalty(scale, …)` so the
    two cancel when summed across agents.
    """
    scale: float
    agents: tuple[str, ...] = ("blue_0",)

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        for agent in self.agents:
            if agent not in out:
                continue
            out[agent] += (state.dist_red_to_hoop / state.arena_radius) * self.scale
        return out


@dataclass
class TagEntryPulse:
    """Zero-sum `+magnitude / -magnitude` on `state.tag_entry`.

    `gainer` gets +magnitude, `loser` gets -magnitude.  Cooldown gating is
    handled upstream by the env's tag state machine — by the time `tag_entry`
    is True, cooldown has already passed.
    """
    magnitude: float
    gainer: str
    loser: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.tag_entry:
            out[self.gainer] += self.magnitude
            out[self.loser]  -= self.magnitude
        return out


@dataclass
class ProximityGradedTag:
    """Zero-sum per-step bonus while `state.tag_during` is True.

    bonus = max_reward * max(0, 1 - dist_b2r / tag_radius)

    Peaks at contact (dist=0 → bonus=max_reward), decays to 0 at the tag-zone
    boundary (dist=tag_radius).  Gives PPO a gradient pointing *toward* contact
    instead of a flat plateau inside the soft-tag sphere.
    """
    max_reward: float
    gainer: str
    loser: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.tag_during:
            bonus = self.max_reward * max(0.0, 1.0 - state.dist_b2r / state.tag_radius)
            out[self.gainer] += bonus
            out[self.loser]  -= bonus
        return out


@dataclass
class ClosingVelInTagZone:
    """Zero-sum bonus while `state.tag_during` for closing on the opponent.

    bonus = scale * max(0, (dist_b2r_prev - dist_b2r) / step_period)

    Rewards driving in faster than the opponent can flee — the precondition
    for crossing CRASH_VEL_THR and triggering a TakeDown event.
    """
    scale: float
    gainer: str
    loser: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.tag_during:
            closing = (state.dist_b2r_prev - state.dist_b2r) / state.step_period
            bonus = self.scale * max(0.0, closing)
            out[self.gainer] += bonus
            out[self.loser]  -= bonus
        return out


@dataclass
class TakeDown:
    """Event on `state.drone_drone_crash` (env-gated by |v_rel·normal| > thr).

    `aggressor` gets +aggressor_reward; `victim` gets +victim_penalty
    (typically a negative value).  Two independent magnitudes so the two sides
    of the event can diverge later without touching env code.
    """
    aggressor_reward: float
    victim_penalty: float
    aggressor: str
    victim: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.drone_drone_crash:
            out[self.aggressor] += self.aggressor_reward
            out[self.victim]    += self.victim_penalty
        return out


@dataclass
class InterceptShaping:
    """Closing-velocity reward on Red's short-horizon predicted position,
    active only when Red is within `activation_dist` of the hoop.

    Inputs (`dist_def_to_future_red`, `dist_def_to_future_red_prev`) are
    populated each step by team_env from the world-frame Red velocity:
        future_red = red_pos + lookahead_s · red_vel_world
        dist_def_to_future_red = ‖defender_pos - future_red‖

    Mirrors ClosingVelInTagZone in structure (zero-floored closing rate
    weighted by `scale`), but gated by dist_red_to_hoop instead of
    tag_during.  Non-zero-sum: only the defender is rewarded — Red is
    not penalised on this signal (Red already has its own
    HoopDistancePenalty pulling it toward the hoop).

    `lookahead_s` is informational here (the env consumed it when
    populating the future-red distances); kept in the dataclass so the
    YAML stays self-documenting next to `scale` and `activation_dist`.
    """
    scale: float
    lookahead_s: float
    activation_dist: float
    defender: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.dist_red_to_hoop >= self.activation_dist:
            return out
        closing = (
            state.dist_def_to_future_red_prev - state.dist_def_to_future_red
        ) / state.step_period
        out[self.defender] += self.scale * max(0.0, closing)
        return out


@dataclass
class GoalSideCone:
    """Per-step pull for defender to be on the goal side of Red→hoop axis.

        axis      = red_pos - hoop_pos
        bh        = blue_pos - hoop_pos
        along     = bh · (axis / ‖axis‖)
        t         = along / ‖axis‖             (0 at hoop, 1 at Red)
        cos_align = along / ‖bh‖               (cosine of angle from axis)

        reward = scale × max(0, cos_align) × t        if 0 ≤ t ≤ 1
        reward = 0                                     otherwise

    Zero at hoop (t=0), peaks at Red (t=1, on-axis), zero past Red
    (t>1), zero behind hoop (t<0).  Decays smoothly off-axis via
    cos_align.  Non-zero-sum: only `defender` is rewarded.  HoopAnchor
    still pulls defender back toward hoop, balancing this outward pull
    at an equilibrium guard position.
    """
    scale: float
    defender: str = "blue_0"

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if self.defender not in out:
            return out
        axis = state.red_pos - state.hoop_pos
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-8:
            return out
        bh = state.blue_pos - state.hoop_pos
        bh_len = float(np.linalg.norm(bh))
        if bh_len < 1e-8:
            return out
        along = float(np.dot(bh, axis / axis_len))
        t = along / axis_len
        if t < 0.0 or t > 1.0:
            return out
        cos_align = along / bh_len
        out[self.defender] += self.scale * max(0.0, cos_align) * t
        return out
