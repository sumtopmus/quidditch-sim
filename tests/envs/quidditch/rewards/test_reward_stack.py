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


def test_team_v2_stack_produces_expected_rewards():
    """`default_team_stack()` (loaded from conf/reward/team_v2.yaml) must produce
    the exact rewards team_env.step() does for a known StepState.  Locks YAML
    magnitudes + term composition against silent drift.
    """
    from envs.quidditch.rewards import default_team_stack

    # Literals match conf/reward/team_v2.yaml — kept in sync by this test.
    TAG_ENTRY_REWARD = 5.0
    TAG_DURATION_REWARD_MAX = 0.05
    CLOSING_VEL_REWARD_SCALE = 0.05
    DIST_REWARD_SCALE = 0.01
    HOOP_ANCHOR_SCALE = 0.005

    stack = default_team_stack()

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


def test_step_state_has_future_red_fields_defaulted_to_zero():
    """New fields for InterceptShaping; default 0.0 so existing callers
    that build StepState without them keep working."""
    from envs.quidditch.rewards.stack import StepState
    state = StepState(agent_ids=("red_0", "blue_0"))
    assert state.dist_def_to_future_red == 0.0
    assert state.dist_def_to_future_red_prev == 0.0


def test_reward_lookahead_constant_exists():
    """Constant lives in envs.quidditch.constants so team_env (which
    populates the StepState fields) and the YAML (which carries the
    InterceptShaping lookahead_s param) read from the same source."""
    from envs.quidditch.constants import REWARD_LOOKAHEAD_S
    assert REWARD_LOOKAHEAD_S == 0.5


from envs.quidditch.rewards.terms import InterceptShaping


def test_intercept_shaping_zero_when_red_far_from_hoop():
    """Activation gate: dist_red_to_hoop >= activation_dist → zero reward."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=2.0,                  # >= 1.5, gate misses
        dist_def_to_future_red=0.3,
        dist_def_to_future_red_prev=0.5,       # blue closing on future-red
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_intercept_shaping_positive_when_red_near_and_blue_closing():
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=1.0,                  # < 1.5, gate fires
        dist_def_to_future_red=0.3,
        dist_def_to_future_red_prev=0.5,
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    # closing = (0.5 - 0.3) / (1/240) = 48 m/s
    # bonus = 0.05 * 48 = 2.4
    assert out["blue_0"] == 0.05 * (0.5 - 0.3) / (1 / 240.0)
    # Term is non-zero-sum: Red is NOT penalised.
    assert out["red_0"] == 0.0


def test_intercept_shaping_zero_when_blue_separating_from_future_red():
    """max(0, closing) floors at 0 when defender is moving away from
    future-red — no negative reward for retreating."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=1.0,
        dist_def_to_future_red=0.7,
        dist_def_to_future_red_prev=0.5,       # blue retreating
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_intercept_shaping_uses_only_defender_field():
    """Even with all other state inputs non-zero, only `defender` is rewarded."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=0.5,
        dist_def_to_future_red=0.0, dist_def_to_future_red_prev=1.0,
        step_period=1 / 240.0,
        # Distract: set other things that might leak into reward.
        tag_during=True, tag_entry=True, scored=True, drone_drone_crash=True,
    )
    out = term.compute(state)
    assert out["red_0"] == 0.0
    assert out["blue_0"] > 0.0


def test_team_v3_intercept_stack_composition():
    """conf/reward/team_v3_intercept.yaml: 10 terms in the expected order,
    Blue removed from HoopDistancePenalty, InterceptShaping inserted
    between HoopAnchor and ScoreEvent."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_v3_intercept")
    expected = [
        "TagEntryPulse", "ProximityGradedTag", "ClosingVelInTagZone",
        "HoopDistancePenalty", "ZeroSumDistMirror", "HoopAnchor",
        "InterceptShaping",
        "ScoreEvent", "TakeDown", "CrashEvent",
    ]
    assert [type(t).__name__ for t in stack.terms] == expected

    # HoopDistancePenalty in v3 is Red-only (no blue→midpoint entry).
    hdp = next(t for t in stack.terms if type(t).__name__ == "HoopDistancePenalty")
    assert dict(hdp.agent_to_target) == {"red_0": "hoop"}

    # InterceptShaping carries the spec'd starting values.
    isp = next(t for t in stack.terms if type(t).__name__ == "InterceptShaping")
    assert isp.scale == 0.05
    assert isp.lookahead_s == 0.5
    assert isp.activation_dist == 1.5
    assert isp.defender == "blue_0"


def test_step_state_hoop_pos_defaults_to_zeros():
    from envs.quidditch.rewards.stack import StepState
    state = StepState(agent_ids=("red_0", "blue_0"))
    assert np.array_equal(state.hoop_pos, np.zeros(3))


from envs.quidditch.rewards.terms import GoalSideCone


def _cone_state(blue_xyz, red_xyz=(-1.0, 0.0, 2.0), hoop_xyz=(2.0, 0.0, 2.0)):
    return _make_state(
        red_pos=np.array(red_xyz, dtype=float),
        blue_pos=np.array(blue_xyz, dtype=float),
        hoop_pos=np.array(hoop_xyz, dtype=float),
    )


def test_goal_side_cone_max_when_blue_at_red_on_axis():
    """Blue at Red's position → t=1, cos_align=1 → reward = scale."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(-1.0, 0.0, 2.0)))
    assert out["red_0"] == 0.0
    assert abs(out["blue_0"] - 0.01) < 1e-12


def test_goal_side_cone_half_when_blue_midway_on_axis():
    """Blue at midpoint(hoop, red) → t=0.5, cos_align=1 → reward = scale/2."""
    term = GoalSideCone(scale=0.02, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(0.5, 0.0, 2.0)))
    assert abs(out["blue_0"] - 0.01) < 1e-12


def test_goal_side_cone_zero_at_hoop():
    """Blue at the hoop → bh = 0 → reward 0 (HoopAnchor still pulls)."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(2.0, 0.0, 2.0)))
    assert out["blue_0"] == 0.0


def test_goal_side_cone_zero_when_blue_past_red():
    """Blue further from hoop than Red along axis → t>1 → reward 0."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(-2.0, 0.0, 2.0)))  # t = 4/3
    assert out["blue_0"] == 0.0


def test_goal_side_cone_zero_when_blue_behind_hoop():
    """Blue on the wrong side of hoop (away from Red) → cos_align<0 → reward 0."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(3.0, 0.0, 2.0)))
    assert out["blue_0"] == 0.0


def test_goal_side_cone_decays_off_axis():
    """Blue at t≈0.5 along axis but offset perpendicular → cos_align<1 → reward < scale/2."""
    term = GoalSideCone(scale=0.02, defender="blue_0")
    # Hoop at (2,0,2), Red at (-1,0,2): axis along -x.  Blue at (0.5, 1, 2) is
    # 1m off-axis (in +y).  along = 1.5, axis_len = 3 → t = 0.5.
    # ‖bh‖ = √(1.5²+1²) = √3.25, cos_align = 1.5/√3.25 ≈ 0.832.
    out = term.compute(_cone_state(blue_xyz=(0.5, 1.0, 2.0)))
    expected = 0.02 * (1.5 / np.sqrt(3.25)) * 0.5
    assert abs(out["blue_0"] - expected) < 1e-12


def test_goal_side_cone_zero_when_red_at_hoop():
    """Degenerate: axis_len ≈ 0 → reward 0 (no axis to project onto)."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(
        blue_xyz=(0.5, 0.0, 2.0), red_xyz=(2.0, 0.0, 2.0)))
    assert out["blue_0"] == 0.0


def test_goal_side_cone_does_not_reward_red():
    """Non-zero-sum: defender-only term.  Red gets 0 even at favorable geom."""
    term = GoalSideCone(scale=0.01, defender="blue_0")
    out = term.compute(_cone_state(blue_xyz=(-1.0, 0.0, 2.0)))
    assert out["red_0"] == 0.0


from envs.quidditch.rewards.terms import HoopApproachShaping


def test_hoop_approach_shaping_positive_when_closing():
    """Potential-based progress: reward = scale * (prev_dist - curr_dist)."""
    term = HoopApproachShaping(scale=2.0, agent="red_0")
    out = term.compute(_make_state(dist_red_to_hoop=1.5, dist_red_to_hoop_prev=2.0))
    assert out["red_0"] == 2.0 * (2.0 - 1.5)   # +1.0 for closing 0.5 m
    assert out["blue_0"] == 0.0


def test_hoop_approach_shaping_negative_when_receding():
    term = HoopApproachShaping(scale=2.0, agent="red_0")
    out = term.compute(_make_state(dist_red_to_hoop=2.5, dist_red_to_hoop_prev=2.0))
    assert out["red_0"] == 2.0 * (2.0 - 2.5)   # -1.0 for receding 0.5 m
    assert out["blue_0"] == 0.0


def test_hoop_approach_shaping_zero_when_stationary():
    """No camping benefit: net-zero when distance is unchanged."""
    term = HoopApproachShaping(scale=2.0, agent="red_0")
    out = term.compute(_make_state(dist_red_to_hoop=2.0, dist_red_to_hoop_prev=2.0))
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_hoop_approach_shaping_only_configured_agent():
    """Single-agent reuse: agent='drone_0' rewards only that agent."""
    term = HoopApproachShaping(scale=1.0, agent="drone_0")
    out = term.compute(_make_state(
        agent_ids=("drone_0",), dist_red_to_hoop=1.0, dist_red_to_hoop_prev=1.5))
    assert out == {"drone_0": 0.5}


def test_step_state_dist_red_to_hoop_prev_defaults_to_zero():
    from envs.quidditch.rewards.stack import StepState
    state = StepState(agent_ids=("red_0", "blue_0"))
    assert state.dist_red_to_hoop_prev == 0.0


def test_team_v4_cone_stack_composition():
    """conf/reward/team_v4_cone.yaml: adds GoalSideCone to v3_intercept;
    11 terms in the expected order."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_v4_cone")
    expected = [
        "TagEntryPulse", "ProximityGradedTag", "ClosingVelInTagZone",
        "HoopDistancePenalty", "ZeroSumDistMirror", "HoopAnchor",
        "GoalSideCone", "InterceptShaping",
        "ScoreEvent", "TakeDown", "CrashEvent",
    ]
    assert [type(t).__name__ for t in stack.terms] == expected
    cone = next(t for t in stack.terms if type(t).__name__ == "GoalSideCone")
    assert cone.scale == 0.01
    assert cone.defender == "blue_0"


def test_red_scoring_stack_composition():
    """conf/reward/red_scoring.yaml: a dominant dense hoop-approach shaper, the
    +10 score event, and a crash/OOB penalty.  Red-only (Blue is a frozen
    hover in the skeleton, so no defender terms)."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("red_scoring")
    assert [type(t).__name__ for t in stack.terms] == [
        "HoopApproachShaping", "ScoreEvent", "CrashEvent",
    ]
    shaping = stack.terms[0]
    assert shaping.scale == 2.0
    assert shaping.agent == "red_0"
    score = stack.terms[1]
    assert score.magnitude == 10.0
    assert score.scorer == "red_0"


def test_team_selfplay_v1_stack_composition():
    """conf/reward/team_selfplay_v1.yaml: Red dense-approach + zero-sum score;
    Blue anchor + INTERCEPT shaping + takedown; both crash-penalised.
    InterceptShaping is now present (Step-3 defender-aware env refactor)."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_selfplay_v1")
    assert [type(t).__name__ for t in stack.terms] == [
        "HoopApproachShaping", "ScoreEvent", "HoopAnchor",
        "InterceptShaping", "TakeDown", "CrashEvent",
    ]
    shaping = stack.terms[0]
    assert shaping.scale == 2.0 and shaping.agent == "red_0"
    score = stack.terms[1]
    assert score.magnitude == 10.0
    assert score.scorer == "red_0"
    assert score.zero_sum_opponent == "blue_0"   # Blue loses 10 when Red scores
    intercept = stack.terms[3]
    assert intercept.defender == "blue_0"
    assert intercept.scale == 0.05
    assert intercept.activation_dist == 1.5
