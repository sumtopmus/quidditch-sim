"""Tests for SimpleEnvFactory and TeamEnvFactory."""
from __future__ import annotations


def test_simple_env_factory_builds_16d_train_env():
    from envs.quidditch.env_factories import SimpleEnvFactory
    factory = SimpleEnvFactory(
        n_envs=2, randomise_start=False, episode_seconds=30.0,
        obs_spec_name="SIMPLE_ENV_OBS", seed=42,
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
        obs_spec_name="SIMPLE_ENV_OBS", seed=42,
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
        obs_spec_name="DUEL_V2_WORLD",
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
        obs_spec_name="DUEL_V2_WORLD",
        frame_stack=1,
        seed=42,
    )
    env = factory.build_train_env()
    try:
        assert env.observation_space.shape == (25,)
    finally:
        env.close()
