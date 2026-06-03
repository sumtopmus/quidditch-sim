"""Verify simple_env universal feature dict refactor produces byte-identical obs."""
from __future__ import annotations

import numpy as np

from envs.quidditch.simple_env import QuidditchSimpleEnv


def test_simple_env_obs_deterministic_on_reset():
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    obs1, _ = env.reset(seed=42)
    env.close()

    env2 = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    obs2, _ = env2.reset(seed=42)
    env2.close()

    np.testing.assert_array_equal(obs1, obs2)
    assert obs1.shape == (16,)
    assert obs1.dtype == np.float32
