"""Fixed Red start position — a deterministic-attack-run curriculum lever.

When randomise_red_start is False, TeamConfig may pin Red's spawn to an explicit
(x, y, z) instead of the origin, so a from-scratch learner sees a constant
starting state (removing start-disc variance from the gradient).
"""
from __future__ import annotations

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig


def _reset_red_pos(cfg: TeamConfig) -> np.ndarray:
    env = QuidditchTeamEnv(cfg=cfg)
    try:
        env.reset(seed=0)
        return env._red_pos().copy()
    finally:
        env.close()


def test_fixed_start_uses_configured_red_position():
    pos = _reset_red_pos(
        TeamConfig(randomise_red_start=False, red_start_pos=(-1.0, 0.0, 0.5))
    )
    assert np.allclose(pos, [-1.0, 0.0, 0.5], atol=1e-6)


def test_fixed_start_defaults_to_origin_when_pos_unset():
    """Back-compat: randomise=False with no explicit pos → origin (legacy)."""
    pos = _reset_red_pos(TeamConfig(randomise_red_start=False))
    assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-6)


def test_fixed_start_is_deterministic_across_seeds():
    cfg = TeamConfig(randomise_red_start=False, red_start_pos=(-1.0, 0.0, 0.5))
    a = _reset_red_pos(cfg)
    b = _reset_red_pos(cfg)
    assert np.allclose(a, b)
