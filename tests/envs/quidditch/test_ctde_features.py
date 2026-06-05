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


def test_ground_truth_critic_features():
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    assert crit["red_pos_abs"].shape == (3,)
    assert crit["blue_pos_abs"].shape == (3,)
    assert crit["tag_state_onehot"].shape == (2,)
    assert crit["terminal_margins"].shape == (4,)
    # margins are normalized to [0,1].
    assert np.all(crit["terminal_margins"] >= 0.0)
    assert np.all(crit["terminal_margins"] <= 1.0)
    # opp_next_action passes straight through.
    a = np.array([0.1, -0.2, 0.3, 0.4], np.float32)
    assert np.allclose(env._build_critic_features("blue_0", a)["opp_next_action"], a)
