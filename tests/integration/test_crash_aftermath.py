"""Crash aftermath: drone-drone ram defers termination so the crash stays on screen.

When `TeamConfig.crash_aftermath_seconds > 0`, the drone-drone-crash trigger
step still applies ±TAKE_DOWN rewards but does NOT terminate.  The env then
keeps stepping for the configured duration with Red's motors cut.  During
aftermath rewards are 0 for both, and other terminal conditions (red floor,
red oob, etc.) are suppressed until the timer expires.

Default `crash_aftermath_seconds = 0.0` preserves training-time behavior
(verified by test_take_down.py — kept untouched).
"""
from __future__ import annotations

from math import ceil

import mujoco
import numpy as np
import pytest

from envs.quidditch.rewards import TAKE_DOWN_REWARD, TAKE_DOWN_PENALTY
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state

pytestmark = pytest.mark.slow


def _ram_until_crash(env: QuidditchTeamEnv, max_steps: int = 30) -> tuple[int, dict]:
    """Drive blue at red at 3 m/s, return (step_index, info) on first drone_drone_crash."""
    set_body_state(env._world, "red_0",  pos=(0.0, 0.0, 1.0), vel=( 0.0, 0.0, 0.0))
    set_body_state(env._world, "blue_0", pos=(0.4, 0.0, 1.0), vel=(-3.0, 0.0, 0.0))
    mujoco.mj_forward(env._world.model, env._world.data)
    env._setpoint_red  = np.array([ 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    env._setpoint_blue = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    env._red.set_setpoint(env._setpoint_red)
    env._blue.set_setpoint(env._setpoint_blue)
    env._red_takeoff_grace = 0
    env._blue_takeoff_grace = 0

    zero = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
    for i in range(max_steps):
        _o, rew, term, _tr, info = env.step(zero)
        if info["red_0"].get("drone_drone_crash"):
            return i, {"rew": rew, "term": term, "info": info}
    raise AssertionError(f"no drone_drone_crash within {max_steps} steps")


def test_aftermath_defers_termination_and_cuts_red_motors() -> None:
    """Trigger step fires take-down rewards but does NOT terminate; Red goes limp."""
    aftermath_s = 3.0
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
        crash_aftermath_seconds=aftermath_s,
    ))
    try:
        env.reset(seed=42)
        _, snap = _ram_until_crash(env)

        # Trigger step: ±TAKE_DOWN rewards still fire, but termination is deferred.
        assert snap["rew"]["blue_0"] > TAKE_DOWN_REWARD - 1.0, (
            f"blue should still receive +{TAKE_DOWN_REWARD} on the trigger step, "
            f"got {snap['rew']['blue_0']:.3f}"
        )
        assert snap["rew"]["red_0"] < TAKE_DOWN_PENALTY + 1.0, (
            f"red should still receive {TAKE_DOWN_PENALTY} on the trigger step, "
            f"got {snap['rew']['red_0']:.3f}"
        )
        assert not any(snap["term"].values()), (
            f"aftermath should defer termination, got term={snap['term']}"
        )

        # Red's motor output is now cut.
        assert env._red._motors_disabled is True
        # And the countdown is armed.
        expected_total = max(1, int(ceil(aftermath_s / env._red.step_period)))
        assert env._aftermath_steps_left == expected_total - 1 or \
               env._aftermath_steps_left == expected_total, (
                   f"aftermath counter should be near {expected_total}, "
                   f"got {env._aftermath_steps_left}"
               )
    finally:
        env.close()


def test_aftermath_terminates_after_full_duration() -> None:
    """Episode terminates exactly after crash_aftermath_seconds elapses; rewards are 0."""
    aftermath_s = 0.5  # ~60 control steps at 120 Hz — small for test speed
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
        crash_aftermath_seconds=aftermath_s,
    ))
    try:
        env.reset(seed=42)
        _ram_until_crash(env)

        # Step until aftermath ends.  Cap is generous so a regression
        # (counter that never decrements) blows up loudly.
        step_period = env._red.step_period
        expected_aftermath_steps = int(ceil(aftermath_s / step_period))
        zero = {a: np.zeros(4, dtype=np.float32) for a in env.possible_agents}
        # env.agents survived the trigger step (we deferred termination)
        assert env.agents, "env.agents should be live during aftermath"

        terminated_at = None
        for i in range(expected_aftermath_steps + 5):
            _o, rew, term, _tr, info = env.step(zero)
            assert rew["red_0"] == 0.0 and rew["blue_0"] == 0.0, (
                f"aftermath rewards should be 0, got {rew} at i={i}"
            )
            assert info["red_0"].get("aftermath") is True
            if any(term.values()):
                terminated_at = i
                break

        assert terminated_at is not None, (
            f"aftermath never terminated within {expected_aftermath_steps + 5} steps"
        )
        # +/- 1 for ceil/decrement edge.
        assert abs(terminated_at - (expected_aftermath_steps - 1)) <= 1, (
            f"aftermath terminated at step {terminated_at}, "
            f"expected ~{expected_aftermath_steps - 1}"
        )
    finally:
        env.close()


def test_default_zero_aftermath_preserves_immediate_termination() -> None:
    """With the default (= 0.0), drone-drone-crash still terminates the same step."""
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        env.reset(seed=42)
        _, snap = _ram_until_crash(env)
        assert any(snap["term"].values()), (
            f"default cfg should terminate on drone-drone-crash, got term={snap['term']}"
        )
        assert env._aftermath_steps_left == 0
        assert env._red._motors_disabled is False
    finally:
        env.close()
