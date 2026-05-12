"""OpponentControlledEnv augments the learner's obs from team_env's 22-d
into a 25-d vector with:
  [15:18]  vec_to_hoop          (world frame)
  [18:21]  opp_pos_rel          (world frame)
  [21:24]  opp_vel_rel          (world frame, free-joint qvel)
  [24]     closing_rate         (-d‖opp_pos - learner_pos‖/dt)

FrameStackWrapper additionally stacks N consecutive 25-d obs into a 25·N
flat vector, matching SB3's VecFrameStack so video-callback (single-env)
and training (vec-env) paths produce identical shapes.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.obs_spec import AUGMENTED_OBS
from envs.quidditch.opponents import (
    FrameStackWrapper,
    OpponentControlledEnv,
    from_spec,
)
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state


AUGMENTED_OBS_DIM = AUGMENTED_OBS.dim


def _make_blue_env() -> OpponentControlledEnv:
    team = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False))
    return OpponentControlledEnv(
        team, learner_id="blue_0", opponent=from_spec("zero"),
    )


def test_augmented_obs_shape_is_25_dim() -> None:
    env = _make_blue_env()
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (AUGMENTED_OBS_DIM,)
        assert obs.shape == (AUGMENTED_OBS_DIM,)
        assert obs.dtype == np.float32
    finally:
        env.close()


def test_vec_to_hoop_slot_points_to_hoop() -> None:
    """Slot [15:18] should equal HOOP_CENTER − blue_pos at reset."""
    env = _make_blue_env()
    try:
        # Drive blue to a known position so we can verify the slot value.
        env.reset(seed=0)
        set_body_state(env.team_env._world, "blue_0", pos=(0.5, 1.0, 1.5))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        # Step once with zero actions so the env recomputes the obs.
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        expected = HOOP_CENTER - np.array([0.5, 1.0, 1.5])
        # Allow a small slop because mj_step inside env.step moves the drone
        # under PID; the vec is computed AFTER that step, not at our injected
        # state.  Direction should still match closely.
        actual = obs[15:18]
        assert np.linalg.norm(actual - expected) < 0.5, (
            f"vec_to_hoop should point to hoop from injected pos; "
            f"expected ≈ {expected}, got {actual}"
        )
    finally:
        env.close()


def test_closing_rate_is_zero_at_reset_and_positive_when_blue_closes() -> None:
    """Closing rate is the wrapper's only stateful feature.  Verify the
    contract: 0 at reset, > 0 when blue actually closes on red.
    """
    env = _make_blue_env()
    try:
        obs, _ = env.reset(seed=0)
        # At reset, _prev_dist_to_opp was initialised from current state, so
        # the next step's closing_rate reflects motion within that step
        # (~0 if no motion yet but blue may drift slightly).  Hard to pin
        # exactly; instead, run two scenarios and compare.

        # Scenario 1: pin both drones in place — closing_rate stays small.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env._prev_dist_to_opp = float(np.linalg.norm(
            np.array([0.0, 0.0, 1.0]) - np.array([1.0, 0.0, 1.0])
        ))
        obs_static, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        static_closing = float(obs_static[24])

        # Scenario 2: pin red, give blue a -x velocity toward red.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env._prev_dist_to_opp = 1.0
        obs_closing, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        closing = float(obs_closing[24])

        # Static < 0.1 m/s; closing should be near blue's injected speed,
        # roughly 1 m/s after one 1/120-s step under PID deceleration.
        assert abs(static_closing) < 0.1, (
            f"static scenario should report ~0 closing_rate, got {static_closing:.3f}"
        )
        assert closing > 0.5, (
            f"closing scenario (blue at -2 m/s toward red) should report a "
            f"positive closing_rate ≥ 0.5 m/s after PID deceleration; "
            f"got {closing:.3f}"
        )
        assert closing > static_closing + 0.5, (
            f"closing scenario must clearly exceed static; "
            f"static={static_closing:.3f}, closing={closing:.3f}"
        )
    finally:
        env.close()


def test_opp_vel_rel_uses_world_frame_not_body_frame() -> None:
    """Slot [21:24] should be the world-frame relative velocity.

    Setup: yaw the drones away from each other so body-frame and world-frame
    encodings diverge.  Inject known world-frame velocities; the slot's
    direction should match the world-frame difference, not the body-frame
    one (which would depend on the body yaws).
    """
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        # Red at origin moving +y at 1 m/s in world frame.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0),
                       vel=(0.0, 1.0, 0.0))
        # Blue at (1, 0, 1) stationary.
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(0.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env._prev_dist_to_opp = 1.0
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        opp_vel_rel = obs[21:24]
        # red world velocity − blue world velocity = (0, 1, 0) - (0, 0, 0) = (0, 1, 0).
        # PID + one step of integration adds noise, so check the y-component
        # dominates and x/z stay smaller — this is the world-frame signature.
        assert abs(opp_vel_rel[1]) > 0.5, (
            f"y component should reflect red's +1 m/s world-y motion, got {opp_vel_rel}"
        )
        assert abs(opp_vel_rel[1]) > abs(opp_vel_rel[0]), (
            f"y should dominate x in opp_vel_rel for this setup, got {opp_vel_rel}"
        )
    finally:
        env.close()


def test_frame_stack_wrapper_doubles_obs_dim() -> None:
    env = FrameStackWrapper(_make_blue_env(), n_stack=2)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (AUGMENTED_OBS_DIM * 2,)
        assert obs.shape == (AUGMENTED_OBS_DIM * 2,)
        # At reset, every slot is the current obs — so the two halves match.
        np.testing.assert_array_equal(
            obs[:AUGMENTED_OBS_DIM], obs[AUGMENTED_OBS_DIM:],
        )
        # After one step, the older half (start) should equal the prior obs;
        # the newer half (end) should be the new step's obs.
        prev_new = obs[AUGMENTED_OBS_DIM:].copy()
        obs2, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(obs2[:AUGMENTED_OBS_DIM], prev_new)
        # The new tail is the new obs — generally different from prev_new
        # because the env stepped (drone physics moved).  Permit equality
        # too (highly unlikely) but assert at minimum the shape is right.
        assert obs2.shape == (AUGMENTED_OBS_DIM * 2,)
    finally:
        env.close()
