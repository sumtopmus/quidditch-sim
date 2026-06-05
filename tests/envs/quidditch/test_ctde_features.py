import numpy as np
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_spec_from_block_names


def _env():
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False),
                           learner_id="blue_0",
                           learner_spec=build_spec_from_block_names(["LIN_POS"]))
    env.reset(seed=0)
    return env


def test_feature_dict_has_world_vel_and_time_remaining():
    env = _env()
    feats = env._build_agent_features("blue_0")
    assert feats["lin_vel_world"].shape == (3,)
    assert feats["time_remaining"].shape == (1,)
    assert 0.0 <= float(feats["time_remaining"][0]) <= 1.0
