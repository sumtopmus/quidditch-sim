"""End-to-end take-down through the full QuidditchTeamEnv step API.

Sister test to tests/unit/test_crash_detector.py — that one verifies the
physics + CrashDetector path (mj_step only) produces drone-drone events at
|v_rel| > CRASH_VEL_THR.  This one verifies the *env*'s reward + termination
wiring: a drone-drone collision above threshold should apply TAKE_DOWN_REWARD
to blue, TAKE_DOWN_PENALTY to red, and terminate the episode in the same step.

Without this, "the +20 take-down is reachable in training" rests on assumption.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from envs.quidditch.constants import CRASH_VEL_THR
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

# Literals match conf/reward/team_v2.yaml's TakeDown term.
TAKE_DOWN_REWARD = 20.0
TAKE_DOWN_PENALTY = -20.0
from tests.conftest import set_body_state

pytestmark = pytest.mark.slow


def test_take_down_through_env_step_applies_reward_and_terminates() -> None:
    """Blue rams Red at 3 m/s through the env's step() API.

    Setup pins both drones at altitude 1 m and aligns their setpoints with
    the injected state so PID isn't fighting the ramming velocity.  Within
    ~30 control steps Blue should physically contact Red at |v_rel| above
    CRASH_VEL_THR, producing a drone_drone_crash event that the env
    translates into ±TAKE_DOWN_{REWARD,PENALTY} and a terminal step.
    """
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        env.reset(seed=42)
        # Skip takeoff grace so an immediate floor-contact wouldn't suppress;
        # not strictly needed at z=1.0 but keeps the test invariant to grace.
        env._red_takeoff_grace = 0
        env._blue_takeoff_grace = 0

        # Pin Red at altitude; aim Blue at it at 3 m/s along -x.
        # 0.4 m starting gap closes in ~130 ms = ~16 control steps at 120 Hz.
        set_body_state(env._world, "red_0",  pos=(0.0, 0.0, 1.0), vel=( 0.0, 0.0, 0.0))
        set_body_state(env._world, "blue_0", pos=(0.4, 0.0, 1.0), vel=(-3.0, 0.0, 0.0))
        mujoco.mj_forward(env._world.model, env._world.data)

        # Align setpoints with injected state so PID supports (instead of
        # fighting) the ramming motion.  Red holds station; Blue's setpoint
        # is beyond Red so PID keeps driving Blue forward through impact.
        env._setpoint_red  = np.array([ 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        env._setpoint_blue = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        env._red.set_setpoint(env._setpoint_red)
        env._blue.set_setpoint(env._setpoint_blue)

        zero_actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        for step_i in range(30):
            _obs, rew, term, _trunc, info = env.step(zero_actions)

            if info["red_0"].get("drone_drone_crash"):
                # Take-down fired.  rewards are layered: blue receives
                # TAKE_DOWN_REWARD (+20) plus any in-zone tag shaping (≤ ~0.15)
                # minus per-step dist shaping (~-0.002); red receives the
                # mirror.  Slack of ±1.0 absorbs the shaping noise.
                assert rew["blue_0"] > TAKE_DOWN_REWARD - 1.0, (
                    f"blue reward should be ≈ +{TAKE_DOWN_REWARD} after take-down, "
                    f"got {rew['blue_0']:.3f}"
                )
                assert rew["red_0"]  < TAKE_DOWN_PENALTY + 1.0, (
                    f"red reward should be ≈ {TAKE_DOWN_PENALTY} after take-down, "
                    f"got {rew['red_0']:.3f}"
                )
                assert any(term.values()), (
                    f"expected termination after take-down, got term={term}"
                )
                # Sanity: it wasn't a SCORE termination spuriously dressed up.
                assert not info["red_0"].get("scored"), (
                    f"take-down step should not also be a score, got info={info['red_0']}"
                )
                return

        raise AssertionError(
            f"no drone_drone_crash event in {step_i + 1} env steps — "
            f"the +{TAKE_DOWN_REWARD} take-down may be unreachable through env.step()"
        )
    finally:
        env.close()
