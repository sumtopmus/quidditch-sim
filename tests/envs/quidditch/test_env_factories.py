"""Tests for SimpleEnvFactory and TeamEnvFactory."""
from __future__ import annotations


_SIMPLE_BLOCKS = ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                  "UNIT_TO_GOAL", "SIGNED_DIST_NORM"]
_DUEL_V2_BLOCKS = ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                   "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
                   "OPP_VEL_REL_WORLD", "CLOSING_RATE"]
_DUEL_V3_BLOCKS = ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                   "VEC_TO_GOAL_BODY", "VEC_TO_HOOP_BODY", "OPP_POS_REL_BODY",
                   "OPP_VEL_REL_BODY_EGO", "CLOSING_RATE"]


def test_simple_env_factory_builds_16d_train_env():
    from envs.quidditch.env_factories import SimpleEnvFactory
    factory = SimpleEnvFactory(
        n_envs=2, randomise_start=False, episode_seconds=30.0,
        obs_blocks=_SIMPLE_BLOCKS, obs_name="SIMPLE_ENV_OBS", seed=42,
    )
    train_env = factory.build_train_env()
    try:
        assert train_env.observation_space.shape == (16,)
        assert train_env.num_envs == 2
    finally:
        train_env.close()


def test_simple_env_factory_builds_eval_env_single_subprocess():
    from envs.quidditch.env_factories import SimpleEnvFactory
    factory = SimpleEnvFactory(
        n_envs=2, randomise_start=False, episode_seconds=30.0,
        obs_blocks=_SIMPLE_BLOCKS, obs_name="SIMPLE_ENV_OBS", seed=42,
    )
    eval_env = factory.build_eval_env()
    try:
        assert eval_env.num_envs == 1
        assert eval_env.observation_space.shape == (16,)
    finally:
        eval_env.close()


def test_team_env_factory_builds_75d_train_env_with_frame_stack():
    from envs.quidditch.env_factories import TeamEnvFactory
    from envs.quidditch.team_env import TeamConfig
    factory = TeamEnvFactory(
        n_envs=2,
        team_cfg=TeamConfig(),
        learner_id="blue_0",
        opponent_spec="beeline_red",
        obs_blocks=_DUEL_V2_BLOCKS,
        obs_name="DUEL_V2_WORLD",
        frame_stack=3,
        seed=42,
    )
    train_env = factory.build_train_env()
    try:
        assert train_env.observation_space.shape == (75,)
        assert train_env.num_envs == 2
    finally:
        train_env.close()


def test_team_env_factory_team_obs_unstacked():
    """frame_stack=1 means no VecFrameStack wrap; obs shape == 25-d
    DUEL_V2_WORLD (OpponentControlledEnv always augments)."""
    from envs.quidditch.env_factories import TeamEnvFactory
    from envs.quidditch.team_env import TeamConfig
    factory = TeamEnvFactory(
        n_envs=1,
        team_cfg=TeamConfig(),
        learner_id="red_0",
        opponent_spec="beeline_blue",
        obs_blocks=_DUEL_V2_BLOCKS,
        obs_name="DUEL_V2_WORLD",
        frame_stack=1,
        seed=42,
    )
    env = factory.build_train_env()
    try:
        assert env.observation_space.shape == (25,)
    finally:
        env.close()


def test_team_factory_threads_learner_id_and_spec_into_team_env():
    """TeamEnvFactory must resolve cfg.obs.blocks → ObsSpec and pass it as
    learner_spec to QuidditchTeamEnv, so the learner sees the right shape."""
    from envs.quidditch.env_factories import TeamEnvFactory
    from envs.quidditch.obs_spec import DUEL_V3_BODY_EGO
    from envs.quidditch.team_env import TeamConfig

    factory = TeamEnvFactory(
        n_envs=1,
        team_cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0",
        opponent_spec="zero",
        obs_blocks=_DUEL_V3_BLOCKS,
        obs_name="DUEL_V3_BODY_EGO",
        frame_stack=1,
        seed=42,
    )
    vec_env = factory.build_train_env()
    try:
        # SB3 vec_envs expose .observation_space.shape on the wrapped single env.
        assert vec_env.observation_space.shape == (DUEL_V3_BODY_EGO.dim,)
    finally:
        vec_env.close()
