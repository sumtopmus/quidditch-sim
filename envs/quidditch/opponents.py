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


class FrozenPolicyOpponent:
    """Wraps an SB3 PPO checkpoint.  Loaded once at construction.

    Args:
        model_path: path to the .zip (with or without the .zip suffix is fine
            — SB3.load handles both).
        deterministic: if True, calls predict(obs, deterministic=True);
            otherwise sampled.  Default False.
    """
    def __init__(self, model_path: str | None = None,
                 deterministic: bool = False) -> None:
        from stable_baselines3 import PPO
        if model_path is None:
            raise ValueError("FrozenPolicyOpponent: model_path is required")
        self.model = PPO.load(model_path)
        self.deterministic = bool(deterministic)

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(4)


class MixtureOpponent:
    """Picks one opponent per episode (weighted random) — supports league play."""
    def __init__(self, mixture: list[tuple[float, Opponent]]) -> None:
        if not mixture:
            raise ValueError("MixtureOpponent: empty mixture")
        weights = np.asarray([w for w, _ in mixture], dtype=np.float64)
        if (weights <= 0).any() or weights.sum() <= 0:
            raise ValueError("MixtureOpponent: weights must be positive")
        self._weights = weights / weights.sum()
        self._opps = [o for _, o in mixture]
        self._rng = np.random.default_rng()
        self._current: Opponent = self._opps[0]

    def reset(self) -> None:
        idx = int(self._rng.choice(len(self._opps), p=self._weights))
        self._current = self._opps[idx]
        self._current.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._current.act(obs)


_REGISTRY: dict[str, type[Opponent]] = {
    "zero":             ZeroOpponent,
    "beeline_red":      BeelineRed,
    "beeline_blue":     BeelineBlue,
    "intercepter_blue": IntercepterBlue,
}


def from_spec(spec: str) -> Opponent:
    """Parse a CLI-friendly opponent spec string.

    Forms:
        "beeline_blue"
        "intercepter_blue:lookahead=0.5"
        "zero"
        "frozen:path/to/best_model.zip"
        "mixture:0.5*beeline_blue,0.5*frozen:path/to/blue_v1.zip"
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("from_spec: empty spec")

    if spec.startswith("mixture:"):
        body = spec[len("mixture:"):]
        parts = [p.strip() for p in body.split(",") if p.strip()]
        components: list[tuple[float, Opponent]] = []
        for p in parts:
            if "*" not in p:
                raise ValueError(f"from_spec: mixture component missing '*': {p!r}")
            w_str, sub_spec = p.split("*", 1)
            components.append((float(w_str), from_spec(sub_spec)))
        return MixtureOpponent(components)

    if spec.startswith("frozen:"):
        return FrozenPolicyOpponent(model_path=spec[len("frozen:"):])

    if ":" in spec:
        name, kv_str = spec.split(":", 1)
        kwargs: dict[str, float] = {}
        for kv in kv_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=", 1)
            kwargs[k.strip()] = float(v)
    else:
        name, kwargs = spec, {}

    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"from_spec: unknown opponent {name!r}")
    return cls(**kwargs)  # type: ignore[arg-type]
