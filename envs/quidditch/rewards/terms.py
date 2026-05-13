"""Composable reward terms — one dataclass per reward signal.

Every term has a `.compute(state: StepState) -> dict[str, float]` method
returning per-agent reward deltas for the current step.  Terms are pure:
they read from `state` and produce a dict; they hold no internal state.

Naming convention: events (one-shot, fire on a flag) use `Event` suffix;
continuous (per-step shaping) terms use a descriptive noun.
"""
from __future__ import annotations

from dataclasses import dataclass

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
