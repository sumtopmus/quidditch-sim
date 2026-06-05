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


def test_oracle_features_finite_and_shaped():
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    for k in ("self_future_disp", "opp_future_rel", "score_pred", "takedown_pred"):
        assert crit[k].shape == (3,)
        assert np.all(np.isfinite(crit[k])), k
    # score_pred[2] is a cosine alignment in [-1, 1].
    assert -1.0001 <= float(crit["score_pred"][2]) <= 1.0001
    # takedown_pred[2] is a 0/1 flag.
    assert float(crit["takedown_pred"][2]) in (0.0, 1.0)


def test_score_pred_degenerate_velocity_is_sentinel():
    # At reset Red hovers (near-zero world velocity) -> time-to-plane sentinel = 1.0
    env = _env()
    crit = env._build_critic_features("blue_0", np.zeros(4, np.float32))
    assert float(crit["score_pred"][0]) == 1.0       # capped time / Tcap
