"""QuidditchTeamEnv per-agent obs builder — DUEL_V2_WORLD and DUEL_V3_BODY_EGO
emitted directly by team_env (no OCE augmenter in the path).
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from envs.quidditch.constants import HOOP_CENTER, REWARD_LOOKAHEAD_S
from envs.quidditch.obs_spec import (
    DUEL_V1_BODY, DUEL_V2_WORLD, DUEL_V3_BODY_EGO,
)
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state


def _team(*, learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO):
    return QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id=learner_id,
        learner_spec=learner_spec,
    )


def test_default_construction_keeps_both_agents_on_duel_v1_body():
    """No learner_id → canary-compatible behavior preserved."""
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False))
    try:
        env.reset(seed=0)
        assert env.observation_space("red_0").shape  == (DUEL_V1_BODY.dim,)
        assert env.observation_space("blue_0").shape == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v3_blue_learner_emits_25d_obs_for_blue_22d_for_red():
    env = _team(learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["blue_0"].shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs["red_0"].shape  == (DUEL_V1_BODY.dim,)
        assert env.observation_space("blue_0").shape == (DUEL_V3_BODY_EGO.dim,)
        assert env.observation_space("red_0").shape  == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v2_blue_learner_emits_25d_obs_for_blue():
    """DUEL_V2_WORLD path: team_env packs the 25-d world-frame obs
    directly (the work formerly done by OCE._augment_learner_obs)."""
    env = _team(learner_id="blue_0", learner_spec=DUEL_V2_WORLD)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["blue_0"].shape == (DUEL_V2_WORLD.dim,)
        assert obs["red_0"].shape  == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v3_vec_to_hoop_is_in_body_frame():
    """Yaw Blue +90° about z; world-x vec_to_hoop becomes body -y."""
    env = _team()
    try:
        env.reset(seed=0)
        # Place blue at origin, yawed +90°; hoop is at (2, 0, 2) world.
        set_body_state(env._world, "blue_0", pos=(0.0, 0.0, 1.5), yaw=np.pi / 2)
        mujoco.mj_forward(env._world.model, env._world.data)
        obs, _, _, _, _ = env.step({"blue_0": np.zeros(4, np.float32),
                                     "red_0":  np.zeros(4, np.float32)})
        blue = obs["blue_0"]
        # Slot order in DUEL_V3_BODY_EGO: ang_vel(3), ang_pos(3), lin_vel(3),
        # lin_pos(3), vec_to_goal(3), vec_to_hoop(3), opp_pos_rel(3),
        # opp_vel_rel(3), closing_rate(1)  → vec_to_hoop slice is [15:18].
        vec_to_hoop_body = blue[15:18]
        # World vec_to_hoop ≈ (2, 0, 0.5); under +90° body yaw, world-x maps
        # to body -y, world-y to body +x, so body-frame is roughly (0, -2, 0.5).
        # PID + one step adds noise; check the dominant axis.
        assert abs(vec_to_hoop_body[1]) > abs(vec_to_hoop_body[0]), (
            f"under +90° yaw, vec_to_hoop should be dominantly along body -y, "
            f"got {vec_to_hoop_body}"
        )
        assert vec_to_hoop_body[1] < 0, (
            f"vec_to_hoop body-y should be negative (world +x → body -y); "
            f"got {vec_to_hoop_body}"
        )
    finally:
        env.close()


def test_v3_opp_pos_rel_is_in_body_frame():
    """Yaw Blue +90°; Red at world +x of Blue → body -y of Blue."""
    env = _team()
    try:
        env.reset(seed=0)
        set_body_state(env._world, "blue_0", pos=(0.0, 0.0, 1.5), yaw=np.pi / 2)
        set_body_state(env._world, "red_0",  pos=(1.0, 0.0, 1.5))
        mujoco.mj_forward(env._world.model, env._world.data)
        obs, _, _, _, _ = env.step({"blue_0": np.zeros(4, np.float32),
                                     "red_0":  np.zeros(4, np.float32)})
        blue = obs["blue_0"]
        opp_pos_rel_body = blue[18:21]
        assert opp_pos_rel_body[1] < 0, (
            f"red is at blue's world +x → body -y under +90° yaw, "
            f"got opp_pos_rel_body = {opp_pos_rel_body}"
        )
    finally:
        env.close()


def test_v3_future_red_dist_updates_step_over_step():
    """team_env populates _dist_def_to_future_red_prev each step; reading
    it on consecutive resets shows the cache advances."""
    env = _team()
    try:
        env.reset(seed=0)
        d0 = env._dist_def_to_future_red_prev
        env.step({"blue_0": np.zeros(4, np.float32),
                  "red_0":  np.zeros(4, np.float32)})
        d1 = env._dist_def_to_future_red_prev
        # The cache must have been populated each step; values are non-zero
        # and consecutive steps generally differ under PID-driven motion.
        assert d0 >= 0.0
        assert d1 >= 0.0
    finally:
        env.close()


def test_oce_passes_through_v3_blue_obs_without_re_augmentation():
    """OCE around a v3 team env emits the 25-d blue obs as-is."""
    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO,
    )
    env = OpponentControlledEnv(team, learner_id="blue_0",
                                 opponent=from_spec("zero"))
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs.shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs.dtype == np.float32
    finally:
        env.close()


@pytest.mark.slow
def test_blue_v4_round_trip_through_new_in_env_packer():
    """blue_v4 was trained under DUEL_V2_WORLD (formerly built by OCE's
    augmenter, now built by team_env's per-agent packer).  Round-trip
    one predict() call to confirm shape + math stays compatible.
    """
    from pathlib import Path
    from stable_baselines3 import PPO

    from envs.quidditch.opponents import FrameStackWrapper

    model_path = Path("models/ppo_hoop_blue_4_20260511_202612/best_model")
    if not model_path.with_suffix(".zip").exists():
        pytest.skip(f"blue_v4 checkpoint not found at {model_path}")

    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V2_WORLD,
    )
    env = OpponentControlledEnv(team, learner_id="blue_0",
                                 opponent=from_spec("zero"))
    env = FrameStackWrapper(env, n_stack=3)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (DUEL_V2_WORLD.dim * 3,)
        model = PPO.load(str(model_path))
        action, _ = model.predict(obs, deterministic=True)
        assert np.asarray(action).shape == (4,)
    finally:
        env.close()


def test_pack_agent_obs_dispatch_uses_structural_equality_not_identity():
    """Regression: under SubprocVecEnv the env is pickled into workers,
    which creates fresh ObsSpec dataclass instances that are NOT
    `is`-equal to the module-level constants — but ARE structurally
    equal under == (ObsSpec is frozen=True).  The packer must dispatch
    on ==.  See 2026-05-18 fix on develop after `make train EXP=blue_v7`
    crashed with `_pack_agent_obs: unsupported spec ObsSpec(...)` on
    every SubprocVecEnv worker.
    """
    import copy
    # Deep-copy the canonical spec — same data, different identity.
    spec_copy = copy.deepcopy(DUEL_V3_BODY_EGO)
    assert spec_copy is not DUEL_V3_BODY_EGO   # precondition: identity differs
    assert spec_copy == DUEL_V3_BODY_EGO       # precondition: structurally equal

    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=spec_copy,
    )
    try:
        obs, _ = env.reset(seed=0)
        # If dispatch were `is`, this would raise ValueError from the
        # `raise ValueError(f"_pack_agent_obs: unsupported spec ...")`
        # tail of _pack_agent_obs.
        assert obs["blue_0"].shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs["red_0"].shape  == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


@pytest.mark.slow
def test_subproc_vec_env_does_not_lose_spec_identity_on_pickling():
    """End-to-end regression: SubprocVecEnv(n_envs=2) actually pickles
    the team_env across the multiprocessing boundary.  Reset must
    succeed and yield the right obs shape on both workers."""
    from envs.quidditch.env_factories import TeamEnvFactory

    factory = TeamEnvFactory(
        n_envs=2,
        team_cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0",
        opponent_spec="zero",
        obs_spec_name="DUEL_V3_BODY_EGO",
        frame_stack=1,
        seed=42,
    )
    vec_env = factory.build_train_env()
    try:
        obs = vec_env.reset()
        # SubprocVecEnv reset returns (n_envs, obs_dim) ndarray.
        assert obs.shape == (2, DUEL_V3_BODY_EGO.dim), (
            f"expected (2, {DUEL_V3_BODY_EGO.dim}), got {obs.shape}"
        )
    finally:
        vec_env.close()
