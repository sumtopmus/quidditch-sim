import numpy as np
from gymnasium import spaces
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_ctde_specs_from_yaml, build_spec_from_block_names


def _ctde_env():
    actor, critic = build_ctde_specs_from_yaml("ctde_v1")
    return QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=actor,
        ctde_mode=True, critic_spec=critic,
    )


def test_learner_obs_is_dict_opponent_flat():
    env = _ctde_env()
    assert isinstance(env.observation_space("blue_0"), spaces.Dict)
    assert isinstance(env.observation_space("red_0"), spaces.Box)
    obs, _ = env.reset(seed=0)
    assert set(obs["blue_0"].keys()) == {"actor", "critic"}
    assert obs["blue_0"]["actor"].shape == (23,)
    assert obs["blue_0"]["critic"].shape == (28,)
    assert obs["red_0"].shape == (22,)        # DUEL_V1_BODY flat


def test_flat_mode_unchanged():
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False),
                           learner_id="blue_0",
                           learner_spec=build_spec_from_block_names(
                               ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS"]))
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["blue_0"], np.ndarray)   # still flat


from envs.quidditch.opponents import OpponentControlledEnv, BeelineRed


def test_oce_ctde_embedded_action_equals_applied():
    env = _ctde_env()
    oce = OpponentControlledEnv(env, learner_id="blue_0", opponent=BeelineRed())
    obs, _ = oce.reset(seed=0)
    assert set(obs.keys()) == {"actor", "critic"}
    # opp_next_action is the first 4 dims of the (unnormalized passthrough) critic.
    embedded = obs["critic"][:4].copy()
    # Stepping applies env._pending_opp_action; it must equal what was embedded.
    np.testing.assert_allclose(env._pending_opp_action, embedded, atol=1e-6)
    oce.step(np.zeros(4, np.float32))
