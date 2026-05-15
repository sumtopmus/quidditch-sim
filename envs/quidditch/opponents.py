"""Opponents that drive a single agent in QuidditchTeamEnv.

An Opponent is structurally a (reset, act) callable pair.  Implementations
include scripted policies (Beeline*, Intercepter*), a no-op (Zero), and
SB3-checkpoint loaders (FrozenPolicy, Mixture).  The training-side
OpponentControlledEnv wrapper queries an opponent each step to drive the
non-learner agent.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO

from envs.quidditch import obs_spec
from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.obs_spec import AUGMENTED_OBS
from envs.quidditch.team_env import QuidditchTeamEnv


# ── Learner-obs augmentation ─────────────────────────────────────────────────
# OpponentControlledEnv hands the learner a 25-d obs derived from team_env's
# raw 22-d per-agent obs.  The transformation is wrapper-local: team_env and
# the frozen-opponent path see the original 22-d shape.  See AUGMENTED_OBS in
# envs.quidditch.obs_spec for the canonical slot layout.


class FrameStackWrapper(gym.Wrapper):
    """Single-env frame stacking, compatible with SB3's VecFrameStack layout.

    Stacks the last ``n_stack`` 1-D observations along axis 0 — oldest first,
    newest last — matching ``VecFrameStack`` so a model trained under a
    frame-stacked vec env can be evaluated on a frame-stacked single env
    (e.g. the periodic video callback) without obs-shape mismatch.
    """

    def __init__(self, env: gym.Env, n_stack: int) -> None:
        super().__init__(env)
        if n_stack < 1:
            raise ValueError(f"n_stack must be ≥ 1, got {n_stack}")
        single_shape = env.observation_space.shape
        if len(single_shape) != 1:
            raise ValueError(
                f"FrameStackWrapper expects 1-D obs, got shape {single_shape}"
            )
        self.n_stack = n_stack
        self._single = single_shape[0]
        new_shape = (self._single * n_stack,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape,
            dtype=env.observation_space.dtype,
        )
        self._frames = np.zeros(new_shape, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialise every stack slot to the current obs so the policy never
        # sees an all-zero "history" that wouldn't appear in the vec env.
        for i in range(self.n_stack):
            self._frames[i * self._single : (i + 1) * self._single] = obs
        return self._frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Shift older frames left (drop oldest); append newest at the tail.
        self._frames[: -self._single] = self._frames[self._single :]
        self._frames[-self._single :] = obs
        return self._frames.copy(), reward, terminated, truncated, info

    # gym.Wrapper.__getattr__ skips underscore-prefixed names, so passthroughs
    # the video callback needs (e.g. env._world.render_cells) won't resolve
    # through the wrapper without an explicit hop.
    @property
    def _world(self):
        return self.env._world


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


def from_spec(spec: str, *, deterministic: bool = False) -> Opponent:
    """Parse a CLI-friendly opponent spec string.

    Forms:
        "beeline_blue"
        "intercepter_blue:lookahead=0.5"
        "zero"
        "frozen:path/to/best_model.zip"
        "mixture:0.5*beeline_blue,0.5*frozen:path/to/blue_v1.zip"

    Args:
        deterministic: forwarded to every FrozenPolicyOpponent constructed
            (including frozen leaves nested inside a mixture). Scripted
            opponents ignore the flag — they're already deterministic.
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
            components.append((float(w_str), from_spec(sub_spec, deterministic=deterministic)))
        return MixtureOpponent(components)

    if spec.startswith("frozen:"):
        from scripts._artifact_io import resolve_parent
        raw_path = spec[len("frozen:"):]
        return FrozenPolicyOpponent(
            model_path=str(resolve_parent(raw_path)),
            deterministic=deterministic,
        )

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


class OpponentControlledEnv(gym.Env):
    """Reduces a QuidditchTeamEnv to single-agent Gym (for SB3) by driving
    the non-learner agent from a frozen Opponent each step.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        team_env: QuidditchTeamEnv,
        *,
        learner_id: str,
        opponent: Opponent,
    ) -> None:
        super().__init__()
        if learner_id not in team_env.possible_agents:
            raise ValueError(
                f"OpponentControlledEnv: learner_id={learner_id!r} not in "
                f"team_env.possible_agents={team_env.possible_agents}"
            )
        self.team_env = team_env
        self.learner_id = learner_id
        self.opponent_id = next(a for a in team_env.possible_agents if a != learner_id)
        self.opponent = opponent

        # Learner's policy sees the 25-d augmented obs; the opponent (via
        # self._last_opp_obs) still receives team_env's raw 22-d obs because
        # that's what frozen-opponent checkpoints were trained on.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(AUGMENTED_OBS.dim,), dtype=np.float32,
        )
        self.action_space      = team_env.action_space(learner_id)
        self.render_mode = team_env.render_mode

        self._last_opp_obs: np.ndarray = np.zeros(
            team_env.observation_space(self.opponent_id).shape, dtype=np.float32,
        )
        # Free-joint dofadr lookups (cached on first reset, when world exists).
        self._learner_dofadr: int = -1
        self._opp_dofadr:     int = -1
        # State for closing_rate computation.
        self._prev_dist_to_opp: float = 0.0
        # Last team_env infos from reset/step — exposed for eval callers that
        # need both agents' info dicts (the wrapper only forwards the learner's).
        self.last_team_infos: dict = {}

    def _cache_dofadrs(self) -> None:
        model = self.team_env._world.model
        for prefix, attr in (
            (self.learner_id,  "_learner_dofadr"),
            (self.opponent_id, "_opp_dofadr"),
        ):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix)
            jnt = int(model.body_jntadr[bid])
            setattr(self, attr, int(model.jnt_dofadr[jnt]))

    def _augment_learner_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Map team_env's 22-d learner obs to the 25-d AUGMENTED_OBS layout.

        The raw 22-d obs is TEAM_ENV_OBS (body-mixed opp_vel_rel + signed-distance
        scalar).  Here we replace slot 15 with vec_to_hoop (world), slots 19:22
        with world-frame opp_vel_rel, and append closing_rate.  See the [obs]
        section of the design spec for the contract.
        """
        data = self.team_env._world.data
        # Free-joint qvel[0:3] is world-frame linear velocity for both bodies.
        learner_vel_world = data.qvel[
            self._learner_dofadr : self._learner_dofadr + 3
        ].astype(np.float64)
        opp_vel_world     = data.qvel[
            self._opp_dofadr : self._opp_dofadr + 3
        ].astype(np.float64)
        opp_vel_rel       = (opp_vel_world - learner_vel_world).astype(np.float32)

        # Pull positions from raw_obs so we don't redo sensor reads:
        #   raw_obs[9:12]  = learner_pos          (world)
        #   raw_obs[16:19] = opp_pos - learner_pos (world)
        learner_pos = raw_obs[9:12]
        opp_pos_rel = raw_obs[16:19]
        vec_to_hoop = (HOOP_CENTER - learner_pos).astype(np.float32)

        dist_to_opp = float(np.linalg.norm(opp_pos_rel))
        closing_rate = (
            (self._prev_dist_to_opp - dist_to_opp) / self.team_env._red.step_period
        )
        self._prev_dist_to_opp = dist_to_opp

        return obs_spec.pack(AUGMENTED_OBS, {
            "ang_vel":      raw_obs[0:3],
            "ang_pos":      raw_obs[3:6],
            "lin_vel":      raw_obs[6:9],
            "lin_pos":      raw_obs[9:12],
            "unit_to_goal": raw_obs[12:15],
            "vec_to_hoop":  vec_to_hoop,
            "opp_pos_rel":  opp_pos_rel,
            "opp_vel_rel":  opp_vel_rel,
            "closing_rate": [closing_rate],
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self.team_env.reset(seed=seed, options=options)
        self.opponent.reset()
        if self._learner_dofadr < 0:
            self._cache_dofadrs()
        self._last_opp_obs = obs[self.opponent_id]
        # Initialise prev-distance so the first step's closing_rate is 0.
        opp_pos_rel = obs[self.learner_id][16:19]
        self._prev_dist_to_opp = float(np.linalg.norm(opp_pos_rel))
        self.last_team_infos = infos
        return self._augment_learner_obs(obs[self.learner_id]), infos[self.learner_id]

    def step(self, action):
        opp_action = self.opponent.act(self._last_opp_obs)
        actions = {self.learner_id: action, self.opponent_id: opp_action}
        obs, rew, term, trunc, infos = self.team_env.step(actions)
        self._last_opp_obs = obs[self.opponent_id]
        self.last_team_infos = infos
        return (
            self._augment_learner_obs(obs[self.learner_id]),
            float(rew[self.learner_id]),
            bool(term[self.learner_id]),
            bool(trunc[self.learner_id]),
            infos[self.learner_id],
        )

    def render(self):
        return self.team_env.render()

    def close(self):
        self.team_env.close()

    @property
    def _world(self):
        """Passthrough to the wrapped team env's World so the video callback
        (and any other reach-through consumer) can call ``render_cells`` /
        ``render_frame`` without knowing about the wrapper."""
        return self.team_env._world
