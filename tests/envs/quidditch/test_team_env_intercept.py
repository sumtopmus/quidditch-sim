"""Defender-aware intercept-shaping inputs: dist_def_to_future_red is computed
from blue_0 (the defender) regardless of learner_id, so InterceptShaping works
in two-policy self-play (RLlib migration Step 3 prerequisite)."""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.constants import REWARD_LOOKAHEAD_S
from envs.quidditch.rewards.stack import RewardStack
from envs.quidditch.team_env import QuidditchTeamEnv

_ZERO_ACTION = {"red_0": np.zeros(4, np.float32), "blue_0": np.zeros(4, np.float32)}


def _blue_to_future_red(env: QuidditchTeamEnv) -> float:
    """Recompute the defender(blue)->future-Red distance straight from env state."""
    red_vel = env._world.data.qvel[env._red_dofadr : env._red_dofadr + 3].copy()
    future_red = env._red_pos() + REWARD_LOOKAHEAD_S * red_vel
    return float(np.linalg.norm(env._blue_pos() - future_red))


@pytest.mark.parametrize("learner_id", [None, "red_0", "blue_0"])
def test_intercept_inputs_are_blue_based_regardless_of_learner_id(learner_id):
    env = QuidditchTeamEnv(reward_stack=RewardStack(terms=[]), learner_id=learner_id)
    env.reset(seed=0)

    # Reset seeds both caches to the blue->future-Red distance (a real, nonzero
    # distance because blue and red start apart).
    expected = _blue_to_future_red(env)
    assert expected > 0.0
    assert env._dist_def_to_future_red == pytest.approx(expected)
    assert env._dist_def_to_future_red_prev == pytest.approx(expected)

    prev = env._dist_def_to_future_red
    env.step(_ZERO_ACTION)

    # After a step: prev rolls to the old value, curr is recomputed blue-based.
    assert env._dist_def_to_future_red == pytest.approx(_blue_to_future_red(env))
    assert env._dist_def_to_future_red_prev == pytest.approx(prev)
