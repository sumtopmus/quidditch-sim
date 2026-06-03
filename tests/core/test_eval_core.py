"""Behavior contract for core.eval_core.

Pure-function tests (no real env) on the helpers, plus a slow integration
test running 1 episode against zero_red.
"""
from __future__ import annotations

import pytest


def test_classify_terminal_score_takes_precedence() -> None:
    from core.eval_core import _classify_terminal
    # Score wins over every other condition.
    assert _classify_terminal({"scored": True, "red_floor": True}, {}) == "score"
    assert _classify_terminal({}, {"scored": True, "blue_wall_crash": True}) == "score"


def test_classify_terminal_drone_drone_over_individual_failures() -> None:
    from core.eval_core import _classify_terminal
    assert _classify_terminal(
        {"drone_drone_crash": True, "red_floor": True},
        {"blue_oob": True},
    ) == "drone_drone_crash"


def test_classify_terminal_role_specific_buckets() -> None:
    from core.eval_core import _classify_terminal
    assert _classify_terminal({"red_floor": True}, {}) == "red_floor"
    assert _classify_terminal({"red_wall_crash": True}, {}) == "red_wall"
    assert _classify_terminal({"red_oob": True}, {}) == "red_oob"
    assert _classify_terminal({}, {"blue_floor": True}) == "blue_floor"
    assert _classify_terminal({}, {"blue_wall_crash": True}) == "blue_wall"
    assert _classify_terminal({}, {"blue_oob": True}) == "blue_oob"


def test_classify_terminal_defaults_to_timeout() -> None:
    from core.eval_core import _classify_terminal
    assert _classify_terminal({}, {}) == "timeout"


def test_scenario_spec_is_immutable() -> None:
    from core.eval_core import ScenarioSpec
    s = ScenarioSpec(opponent="beeline_red", opponent_model_path=None,
                     randomise_start=False, n_episodes=1)
    with pytest.raises(Exception):
        s.n_episodes = 2  # type: ignore[misc]


@pytest.mark.slow
def test_run_scenario_produces_episode_results() -> None:
    """Slow integration: 1 episode against zero_red w/ a scripted learner."""
    from core.eval_core import ScenarioSpec, run_scenario, TERMINAL_BUCKETS

    spec = ScenarioSpec(
        opponent="zero",
        opponent_model_path=None,
        randomise_start=False,
        n_episodes=1,
        crash_aftermath_seconds=0.0,
        deterministic=True,
        learner_id="blue_0",
        seed=42,
    )
    result = run_scenario(
        learner_uri="scripted:beeline_blue",
        scenario=spec,
        render=False,
    )

    assert result.scenario == spec
    assert len(result.episodes) == 1
    ep = result.episodes[0]
    assert ep.length > 0
    assert ep.terminal_cause in TERMINAL_BUCKETS
    assert 0.0 <= result.win_rate <= 1.0
    assert isinstance(result.terminal_cause_counts, dict)
