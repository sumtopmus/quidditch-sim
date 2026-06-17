"""Behavior contract for core.eval_core.

Pure-function tests (no real env) on the surviving helpers: _classify_terminal
and the immutable result dataclasses. (The SB3 run_scenario episode loop was
retired in migration Step 6; the RLlib battery is covered by
tests/rllib/test_eval_battery.py.)
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
