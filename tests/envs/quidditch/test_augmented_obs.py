"""DUEL_V2_WORLD obs construction — done by team_env._pack_agent_obs directly.

(Formerly exercised through the SB3 OpponentControlledEnv pass-through +
FrameStackWrapper; both were retired in migration Step 6, so these tests drive
QuidditchTeamEnv directly and read the learner's per-agent obs.)
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.obs_spec import load_obs_yaml
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state

DUEL_V2_WORLD = load_obs_yaml("duel_v2_world")
DUEL_V2_WORLD_DIM = DUEL_V2_WORLD.dim

# Both agents hold their setpoint each step; we only inspect the learner's obs.
_HOLD = {
    "blue_0": np.zeros(4, dtype=np.float32),
    "red_0":  np.zeros(4, dtype=np.float32),
}


def _make_blue_env() -> QuidditchTeamEnv:
    return QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V2_WORLD,
    )


def test_v2_obs_shape_is_25_dim() -> None:
    env = _make_blue_env()
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space("blue_0").shape == (DUEL_V2_WORLD_DIM,)
        assert obs["blue_0"].shape == (DUEL_V2_WORLD_DIM,)
        assert obs["blue_0"].dtype == np.float32
    finally:
        env.close()


def test_v2_vec_to_hoop_slot_points_to_hoop() -> None:
    """Slot [15:18] equals HOOP_CENTER − blue_pos after the first step."""
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env._world, "blue_0", pos=(0.5, 1.0, 1.5))
        mujoco.mj_forward(env._world.model, env._world.data)
        obs, _, _, _, _ = env.step(_HOLD)

        expected = HOOP_CENTER - np.array([0.5, 1.0, 1.5])
        actual = obs["blue_0"][15:18]
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
        set_body_state(env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env._world, "blue_0", pos=(1.0, 0.0, 1.0))
        mujoco.mj_forward(env._world.model, env._world.data)
        env._prev_dist_to_opp = 1.0
        obs_static, _, _, _, _ = env.step(_HOLD)
        static_closing = float(obs_static["blue_0"][24])

        # Scenario 2: blue moving toward red at -2 m/s.
        set_body_state(env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(env._world.model, env._world.data)
        env._prev_dist_to_opp = 1.0
        obs_closing, _, _, _, _ = env.step(_HOLD)
        closing = float(obs_closing["blue_0"][24])

        assert abs(static_closing) < 0.1
        assert closing > 0.5
        assert closing > static_closing + 0.5
    finally:
        env.close()


def test_v2_opp_vel_rel_uses_world_frame() -> None:
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env._world, "red_0",  pos=(0.0, 0.0, 1.0),
                       vel=(0.0, 1.0, 0.0))
        set_body_state(env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(0.0, 0.0, 0.0))
        mujoco.mj_forward(env._world.model, env._world.data)
        env._prev_dist_to_opp = 1.0
        obs, _, _, _, _ = env.step(_HOLD)

        opp_vel_rel = obs["blue_0"][21:24]
        assert abs(opp_vel_rel[1]) > 0.5, (
            f"y component should reflect red's +1 m/s world-y motion, got {opp_vel_rel}"
        )
        assert abs(opp_vel_rel[1]) > abs(opp_vel_rel[0])
    finally:
        env.close()
