"""Tests for RewardStack, StepState, and the 9 reward terms."""
from __future__ import annotations

import numpy as np

from envs.quidditch.rewards.stack import RewardStack, StepState


def _make_state(**overrides) -> StepState:
    """Build a StepState with sensible defaults for tests."""
    defaults = dict(
        red_pos=np.array([0.0, 0.0, 1.5]),
        blue_pos=np.array([1.0, 0.0, 1.5]),
        dist_b2r=1.0,
        dist_b2r_prev=1.0,
        step_period=1 / 240.0,
        tag_entry=False,
        tag_during=False,
        dist_red_to_hoop=2.0,
        dist_blue_to_midpoint=1.5,
        dist_blue_to_hoop=2.5,
        scored=False,
        red_floor=False, blue_floor=False,
        red_wall_crash=False, blue_wall_crash=False,
        red_oob=False, blue_oob=False,
        drone_drone_crash=False,
        arena_radius=3.0, tag_radius=0.3,
        agent_ids=("red_0", "blue_0"),
    )
    defaults.update(overrides)
    return StepState(**defaults)


def test_empty_stack_returns_zero_for_each_agent():
    stack = RewardStack(terms=[])
    rewards = stack.compute_step(_make_state())
    assert rewards == {"red_0": 0.0, "blue_0": 0.0}


def test_stack_sums_term_contributions():
    class _ConstTerm:
        def __init__(self, value, agents):
            self.value = value; self.agents = agents
        def compute(self, state):
            return {a: self.value for a in self.agents}

    stack = RewardStack(terms=[
        _ConstTerm(1.0, ("red_0",)),
        _ConstTerm(2.0, ("red_0", "blue_0")),
        _ConstTerm(-0.5, ("blue_0",)),
    ])
    rewards = stack.compute_step(_make_state())
    assert rewards == {"red_0": 3.0, "blue_0": 1.5}


def test_single_agent_state():
    """StepState supports the single-agent case (one agent in agent_ids)."""
    state = _make_state(agent_ids=("drone_0",))
    stack = RewardStack(terms=[])
    rewards = stack.compute_step(state)
    assert rewards == {"drone_0": 0.0}


from envs.quidditch.rewards.terms import ScoreEvent


def test_score_event_team_zero_sum_on_scored():
    term = ScoreEvent(magnitude=10.0, scorer="red_0", zero_sum_opponent="blue_0")
    rewards = term.compute(_make_state(scored=True))
    assert rewards == {"red_0": 10.0, "blue_0": -10.0}


def test_score_event_team_zero_when_not_scored():
    term = ScoreEvent(magnitude=10.0, scorer="red_0", zero_sum_opponent="blue_0")
    rewards = term.compute(_make_state(scored=False))
    assert rewards == {"red_0": 0.0, "blue_0": 0.0}


def test_score_event_single_agent_no_mirror():
    term = ScoreEvent(magnitude=10.0, scorer="drone_0", zero_sum_opponent=None)
    rewards = term.compute(_make_state(agent_ids=("drone_0",), scored=True))
    assert rewards == {"drone_0": 10.0}


from envs.quidditch.rewards.terms import CrashEvent


def test_crash_event_red_only_on_red_floor():
    term = CrashEvent(magnitude=-20.0,
                       agent_to_crash_flags={
                           "red_0": ("red_floor", "red_wall_crash", "red_oob"),
                           "blue_0": ("blue_floor", "blue_wall_crash", "blue_oob"),
                       })
    rewards = term.compute(_make_state(red_floor=True))
    assert rewards == {"red_0": -20.0, "blue_0": 0.0}


def test_crash_event_both_when_both_crash():
    term = CrashEvent(magnitude=-20.0,
                       agent_to_crash_flags={
                           "red_0": ("red_floor", "red_wall_crash", "red_oob"),
                           "blue_0": ("blue_floor", "blue_wall_crash", "blue_oob"),
                       })
    rewards = term.compute(_make_state(red_oob=True, blue_wall_crash=True))
    assert rewards == {"red_0": -20.0, "blue_0": -20.0}


def test_crash_event_single_agent():
    term = CrashEvent(magnitude=-20.0,
                       agent_to_crash_flags={"drone_0": ("drone_crash",)})
    rewards = term.compute(_make_state(agent_ids=("drone_0",), drone_crash=True))
    assert rewards == {"drone_0": -20.0}


from envs.quidditch.rewards.terms import HoopDistancePenalty


def test_hoop_distance_penalty_team():
    term = HoopDistancePenalty(scale=0.01,
                                 agent_to_target={"red_0": "hoop", "blue_0": "midpoint"})
    state = _make_state(dist_red_to_hoop=3.0, dist_blue_to_midpoint=1.5,
                        arena_radius=3.0)
    rewards = term.compute(state)
    # red: -(3.0/3.0) * 0.01 = -0.01
    # blue: -(1.5/3.0) * 0.01 = -0.005
    assert rewards == {"red_0": -0.01, "blue_0": -0.005}


def test_hoop_distance_penalty_single_agent():
    term = HoopDistancePenalty(scale=0.01,
                                 agent_to_target={"drone_0": "drone_hoop"})
    state = _make_state(agent_ids=("drone_0",), dist_drone_to_hoop=1.5,
                        arena_radius=3.0)
    rewards = term.compute(state)
    assert rewards == {"drone_0": -0.005}


from envs.quidditch.rewards.terms import HoopAnchor


def test_hoop_anchor_blue_only():
    term = HoopAnchor(scale=0.005, agents=("blue_0",))
    state = _make_state(dist_blue_to_hoop=3.0, arena_radius=3.0)
    rewards = term.compute(state)
    # blue: -(3.0/3.0) * 0.005 = -0.005
    assert rewards == {"red_0": 0.0, "blue_0": -0.005}


from envs.quidditch.rewards.terms import ZeroSumDistMirror


def test_zero_sum_dist_mirror_blue_only():
    term = ZeroSumDistMirror(scale=0.01, agents=("blue_0",))
    state = _make_state(dist_red_to_hoop=2.0, arena_radius=3.0)
    rewards = term.compute(state)
    # blue: +(2.0/3.0) * 0.01 = +0.0066666...
    assert rewards["red_0"] == 0.0
    assert rewards["blue_0"] == (2.0 / 3.0) * 0.01


from envs.quidditch.rewards.terms import TagEntryPulse


def test_tag_entry_pulse_fires_only_on_entry():
    term = TagEntryPulse(magnitude=5.0, gainer="blue_0", loser="red_0")
    assert term.compute(_make_state(tag_entry=True)) == {"red_0": -5.0, "blue_0": 5.0}
    assert term.compute(_make_state(tag_entry=False, tag_during=True)) == \
        {"red_0": 0.0, "blue_0": 0.0}


from envs.quidditch.rewards.terms import ProximityGradedTag


def test_proximity_graded_tag_peaks_at_contact():
    term = ProximityGradedTag(max_reward=0.05, gainer="blue_0", loser="red_0")
    # Inside zone, exact contact: dist=0, peaks at max_reward
    state_peak = _make_state(tag_during=True, dist_b2r=0.0, tag_radius=0.3)
    assert term.compute(state_peak) == {"red_0": -0.05, "blue_0": 0.05}

    # Inside zone, at boundary: dist=tag_radius, decays to 0
    state_edge = _make_state(tag_during=True, dist_b2r=0.3, tag_radius=0.3)
    out = term.compute(state_edge)
    assert out["blue_0"] == 0.0
    assert out["red_0"]  == 0.0


def test_proximity_graded_tag_zero_when_not_during():
    term = ProximityGradedTag(max_reward=0.05, gainer="blue_0", loser="red_0")
    state = _make_state(tag_during=False, dist_b2r=0.0)
    assert term.compute(state) == {"red_0": 0.0, "blue_0": 0.0}


from envs.quidditch.rewards.terms import ClosingVelInTagZone


def test_closing_vel_positive_when_closing():
    term = ClosingVelInTagZone(scale=0.05, gainer="blue_0", loser="red_0")
    state = _make_state(tag_during=True,
                        dist_b2r=0.2, dist_b2r_prev=0.5, step_period=1/240.0)
    # closing = (0.5 - 0.2) / (1/240) = 72.0 m/s
    out = term.compute(state)
    assert out["blue_0"] == 0.05 * 72.0
    assert out["red_0"]  == -0.05 * 72.0


def test_closing_vel_zero_when_separating():
    term = ClosingVelInTagZone(scale=0.05, gainer="blue_0", loser="red_0")
    state = _make_state(tag_during=True,
                        dist_b2r=0.5, dist_b2r_prev=0.2, step_period=1/240.0)
    out = term.compute(state)
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_closing_vel_zero_when_not_during():
    term = ClosingVelInTagZone(scale=0.05, gainer="blue_0", loser="red_0")
    state = _make_state(tag_during=False, dist_b2r=0.0, dist_b2r_prev=0.5)
    assert term.compute(state) == {"red_0": 0.0, "blue_0": 0.0}


from envs.quidditch.rewards.terms import TakeDown


def test_take_down_fires_on_drone_drone_crash():
    term = TakeDown(aggressor_reward=20.0, victim_penalty=-20.0,
                     aggressor="blue_0", victim="red_0")
    out = term.compute(_make_state(drone_drone_crash=True))
    assert out == {"red_0": -20.0, "blue_0": 20.0}


def test_take_down_silent_otherwise():
    term = TakeDown(aggressor_reward=20.0, victim_penalty=-20.0,
                     aggressor="blue_0", victim="red_0")
    assert term.compute(_make_state(drone_drone_crash=False)) == {"red_0": 0.0, "blue_0": 0.0}


def test_team_v2_stack_matches_current_constants():
    """The full team_v2 stack with current constants from rewards.py must produce
    the exact rewards team_env.step() would have produced for a sample state.
    Locks term composition during the Phase 2.12–2.13 env refactor.
    """
    from envs.quidditch.rewards import (
        SCORE_REWARD, CRASH_PENALTY, DIST_REWARD_SCALE, HOOP_ANCHOR_SCALE,
        TAG_ENTRY_REWARD, TAG_DURATION_REWARD_MAX, CLOSING_VEL_REWARD_SCALE,
        TAKE_DOWN_REWARD, TAKE_DOWN_PENALTY,
    )
    from envs.quidditch.rewards.terms import (
        HoopDistancePenalty, HoopAnchor, ZeroSumDistMirror,
        TagEntryPulse, ProximityGradedTag, ClosingVelInTagZone,
        ScoreEvent, CrashEvent, TakeDown,
    )

    stack = RewardStack(terms=[
        TagEntryPulse(magnitude=TAG_ENTRY_REWARD,
                       gainer="blue_0", loser="red_0"),
        ProximityGradedTag(max_reward=TAG_DURATION_REWARD_MAX,
                            gainer="blue_0", loser="red_0"),
        ClosingVelInTagZone(scale=CLOSING_VEL_REWARD_SCALE,
                             gainer="blue_0", loser="red_0"),
        HoopDistancePenalty(scale=DIST_REWARD_SCALE,
                             agent_to_target={"red_0": "hoop", "blue_0": "midpoint"}),
        ZeroSumDistMirror(scale=DIST_REWARD_SCALE, agents=("blue_0",)),
        HoopAnchor(scale=HOOP_ANCHOR_SCALE, agents=("blue_0",)),
        ScoreEvent(magnitude=SCORE_REWARD, scorer="red_0",
                    zero_sum_opponent="blue_0"),
        TakeDown(aggressor_reward=TAKE_DOWN_REWARD,
                  victim_penalty=TAKE_DOWN_PENALTY,
                  aggressor="blue_0", victim="red_0"),
        CrashEvent(magnitude=CRASH_PENALTY,
                    agent_to_crash_flags={
                        "red_0":  ("red_floor",  "red_wall_crash",  "red_oob"),
                        "blue_0": ("blue_floor", "blue_wall_crash", "blue_oob"),
                    }),
    ])

    state = _make_state(
        tag_entry=True, tag_during=True,
        dist_b2r=0.15, dist_b2r_prev=0.30, step_period=1 / 240.0,
        dist_red_to_hoop=2.0, dist_blue_to_midpoint=1.0, dist_blue_to_hoop=2.5,
        arena_radius=3.0, tag_radius=0.3,
        scored=False, drone_drone_crash=False,
    )
    out = stack.compute_step(state)

    # Reproduce the same arithmetic team_env.step() does line-by-line.
    expected_red = 0.0
    expected_blue = 0.0
    # Tag entry pulse
    expected_blue += TAG_ENTRY_REWARD
    expected_red  -= TAG_ENTRY_REWARD
    # Tag-during bonuses
    prox_bonus = TAG_DURATION_REWARD_MAX * max(0.0, 1.0 - 0.15 / 0.3)
    close_bonus = CLOSING_VEL_REWARD_SCALE * max(0.0, (0.30 - 0.15) / (1 / 240.0))
    expected_blue += prox_bonus + close_bonus
    expected_red  -= prox_bonus + close_bonus
    # Distance shaping
    expected_red  -= (2.0 / 3.0) * DIST_REWARD_SCALE
    expected_blue -= (1.0 / 3.0) * DIST_REWARD_SCALE
    expected_blue += (2.0 / 3.0) * DIST_REWARD_SCALE
    # Hoop anchor
    expected_blue -= (2.5 / 3.0) * HOOP_ANCHOR_SCALE

    assert abs(out["red_0"]  - expected_red)  < 1e-12, (out["red_0"], expected_red)
    assert abs(out["blue_0"] - expected_blue) < 1e-12, (out["blue_0"], expected_blue)
