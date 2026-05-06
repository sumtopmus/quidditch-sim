"""Opponents that drive a single agent in QuidditchTeamEnv.

An Opponent is structurally a (reset, act) callable pair.  Implementations
include scripted policies (Beeline*, Intercepter*), a no-op (Zero), and
SB3-checkpoint loaders (FrozenPolicy, Mixture).  The training-side
OpponentControlledEnv wrapper queries an opponent each step to drive the
non-learner agent.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Opponent(Protocol):
    """Stateless contract from the env's view; instance state is fine."""
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class ZeroOpponent:
    """Returns zeros — agent holds initial setpoint (hovers in place)."""
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


def _beeline_act(obs: np.ndarray) -> np.ndarray:
    """Steer along obs[12:15] (unit_to_goal)."""
    return np.clip(
        np.array([obs[12], obs[13], 0.0, obs[14]], dtype=np.float32),
        -1.0, 1.0,
    )


class BeelineRed:
    """obs[12:15] is unit_to_hoop_center for the attacker role."""
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray:
        return _beeline_act(obs)


class BeelineBlue:
    """obs[12:15] is unit_to_midpoint for the defender role.

    The midpoint is computed env-side (uses cfg.midpoint_alpha), so this
    scripted policy needs no knowledge of α or the hoop position — same
    code as BeelineRed.
    """
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray:
        return _beeline_act(obs)


class IntercepterBlue:
    """Lookahead defender: target = red_pos + lookahead · red_velocity.

    Reads obs[16:19] (opp_pos_rel) and obs[19:22] (opp_vel_rel) and steers
    toward an extrapolated future position of Red.  Falls back to Beeline
    behavior if the lookahead distance exceeds `lookahead_max`.
    """
    def __init__(self, lookahead: float = 0.5, lookahead_max: float = 1.0) -> None:
        self.lookahead = float(lookahead)
        self.lookahead_max = float(lookahead_max)

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        opp_pos_rel = obs[16:19]
        opp_vel_rel = obs[19:22]
        future = opp_pos_rel + self.lookahead * opp_vel_rel
        d = float(np.linalg.norm(future))
        if d > self.lookahead_max or d < 1e-6:
            return _beeline_act(obs)
        unit = future / d
        return np.clip(
            np.array([unit[0], unit[1], 0.0, unit[2]], dtype=np.float32),
            -1.0, 1.0,
        )
