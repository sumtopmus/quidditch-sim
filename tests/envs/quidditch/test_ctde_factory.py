from gymnasium import spaces
from envs.quidditch.env_factories import TeamEnvFactory
from envs.quidditch.team_env import TeamConfig


def _factory(**kw):
    return TeamEnvFactory(
        n_envs=1, team_cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", opponent_spec="beeline_red",
        obs_blocks=[], obs_name="CTDE_V1", frame_stack=3,
        ctde_mode=True, obs_stem="ctde_v1", **kw)


def test_factory_builds_dict_train_env():
    vec = _factory().build_train_env()
    assert isinstance(vec.observation_space, spaces.Dict)
    assert vec.observation_space["actor"].shape == (69,)   # 23 * 3
    assert vec.observation_space["critic"].shape == (28,)  # unstacked
