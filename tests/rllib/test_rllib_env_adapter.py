"""Contract tests for the RLlib MultiAgentEnv adapter."""
from __future__ import annotations

import numpy as np

from envs.quidditch.rllib_env import make_team_env


def _env():
    return make_team_env({"learner_id": "red_0", "obs_blocks": [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
    ]})


def test_spaces_are_per_agent_and_keyed():
    env = _env()
    assert set(env.possible_agents) == {"red_0", "blue_0"}
    assert set(env.observation_spaces.keys()) == {"red_0", "blue_0"}
    assert set(env.action_spaces.keys()) == {"red_0", "blue_0"}
    assert env.action_spaces["red_0"].shape == (4,)


def test_reset_returns_obs_and_info_dicts():
    env = _env()
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"red_0", "blue_0"}
    assert obs["red_0"].dtype == np.float32
    assert obs["red_0"].shape == env.observation_spaces["red_0"].shape


def test_step_adds_all_done_keys():
    env = _env()
    env.reset(seed=0)
    actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    assert "__all__" in term
    assert "__all__" in trunc
    assert set(rew.keys()) == {"red_0", "blue_0"}
