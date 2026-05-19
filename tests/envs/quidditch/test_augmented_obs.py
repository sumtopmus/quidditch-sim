"""DUEL_V2_WORLD obs construction — formerly done by OCE._augment_learner_obs,
now done by team_env._pack_agent_obs directly.  OCE is a pure pass-through.

FrameStackWrapper additionally stacks N consecutive 25-d obs into a 25·N
flat vector, matching SB3's VecFrameStack so video-callback (single-env)
and training (vec-env) paths produce identical shapes.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.obs_spec import load_obs_yaml

DUEL_V2_WORLD = load_obs_yaml("duel_v2_world")
from envs.quidditch.opponents import (
    FrameStackWrapper,
    OpponentControlledEnv,
    from_spec,
)
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state


DUEL_V2_WORLD_DIM = DUEL_V2_WORLD.dim


def _make_blue_env() -> OpponentControlledEnv:
    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V2_WORLD,
    )
    return OpponentControlledEnv(
        team, learner_id="blue_0", opponent=from_spec("zero"),
    )


def test_v2_obs_shape_is_25_dim() -> None:
    env = _make_blue_env()
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V2_WORLD_DIM,)
        assert obs.shape == (DUEL_V2_WORLD_DIM,)
        assert obs.dtype == np.float32
    finally:
        env.close()


def test_v2_vec_to_hoop_slot_points_to_hoop() -> None:
    """Slot [15:18] equals HOOP_CENTER − blue_pos after the first step."""
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env.team_env._world, "blue_0", pos=(0.5, 1.0, 1.5))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        expected = HOOP_CENTER - np.array([0.5, 1.0, 1.5])
        actual = obs[15:18]
        assert np.linalg.norm(actual - expected) < 0.5, (
            f"vec_to_hoop should point to hoop from injected pos; "
            f"expected ≈ {expected}, got {actual}"
        )
    finally:
        env.close()


def test_v2_closing_rate_zero_at_static_positive_when_closing() -> None:
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        # Scenario 1: pin both → closing ≈ 0.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs_static, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        static_closing = float(obs_static[24])

        # Scenario 2: blue moving toward red at -2 m/s.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs_closing, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        closing = float(obs_closing[24])

        assert abs(static_closing) < 0.1
        assert closing > 0.5
        assert closing > static_closing + 0.5
    finally:
        env.close()


def test_v2_opp_vel_rel_uses_world_frame() -> None:
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0),
                       vel=(0.0, 1.0, 0.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(0.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        opp_vel_rel = obs[21:24]
        assert abs(opp_vel_rel[1]) > 0.5, (
            f"y component should reflect red's +1 m/s world-y motion, got {opp_vel_rel}"
        )
        assert abs(opp_vel_rel[1]) > abs(opp_vel_rel[0])
    finally:
        env.close()


def test_frame_stack_wrapper_doubles_obs_dim() -> None:
    env = FrameStackWrapper(_make_blue_env(), n_stack=2)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V2_WORLD_DIM * 2,)
        assert obs.shape == (DUEL_V2_WORLD_DIM * 2,)
        np.testing.assert_array_equal(
            obs[:DUEL_V2_WORLD_DIM], obs[DUEL_V2_WORLD_DIM:],
        )
        prev_new = obs[DUEL_V2_WORLD_DIM:].copy()
        obs2, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(obs2[:DUEL_V2_WORLD_DIM], prev_new)
        assert obs2.shape == (DUEL_V2_WORLD_DIM * 2,)
    finally:
        env.close()
