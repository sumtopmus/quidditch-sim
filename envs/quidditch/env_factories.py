"""Hydra-instantiable env factories.

Each factory owns env-construction logic previously inlined in
scripts/train_ppo.py and scripts/train_team_ppo.py.  Factories are
constructed once at the top of the training script (Phase 3) or
instantiated by Hydra from a conf/env/*.yaml file (Phase 4), then asked
for the train and eval vec envs they produce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
)


@dataclass
class SimpleEnvFactory:
    """Builds vec envs around QuidditchSimpleEnv (single-agent, 16-d obs)."""
    n_envs: int
    randomise_start: bool
    episode_seconds: float
    obs_spec_name: str = "SIMPLE_ENV_OBS"
    seed: int = 42
    # Optional reward stack injected by scripts/train.py after instantiation.
    # The env itself still constructs its own RewardStack in __init__ from
    # Python constants — Phase 5 wires this in.  Held here so Hydra can hand
    # a composed stack to the factory without yet touching the env class.
    reward_stack: Any = None

    def _make_thunk(self):
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        rs = self.randomise_start
        eps = self.episode_seconds
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode=None, randomise_start=rs, episode_seconds=eps,
            )
        return _thunk

    def build_train_env(self) -> VecEnv:
        thunk = self._make_thunk()
        env_fns = [thunk for _ in range(self.n_envs)]
        return SubprocVecEnv(env_fns) if self.n_envs > 1 else DummyVecEnv(env_fns)

    def build_eval_env(self) -> VecEnv:
        return DummyVecEnv([self._make_thunk()])

    def build_video_env_fn(self):
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        rs = self.randomise_start
        eps = self.episode_seconds
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode="rgb_array", randomise_start=rs, episode_seconds=eps,
            )
        return _thunk


@dataclass
class TeamEnvFactory:
    """Builds vec envs around QuidditchTeamEnv + OpponentControlledEnv.

    Wraps in VecFrameStack when frame_stack > 1.  Eval and video envs match
    the same stack depth so SB3's EvalCallback doesn't reject the obs shape.
    """
    n_envs: int
    team_cfg: Any
    learner_id: str
    opponent_spec: str
    obs_spec_name: str = "DUEL_V2_WORLD"
    frame_stack: int = 3
    seed: int = 42
    # See SimpleEnvFactory.reward_stack — same semantics.
    reward_stack: Any = None

    def _make_thunk(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        def _thunk():
            team = QuidditchTeamEnv(cfg=cfg)
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
        return _thunk

    def build_train_env(self) -> VecEnv:
        thunk = self._make_thunk()
        env_fns = [thunk for _ in range(self.n_envs)]
        vec = SubprocVecEnv(env_fns) if self.n_envs > 1 else DummyVecEnv(env_fns)
        if self.frame_stack > 1:
            vec = VecFrameStack(vec, n_stack=self.frame_stack)
        return vec

    def build_eval_env(self) -> VecEnv:
        vec = DummyVecEnv([self._make_thunk()])
        if self.frame_stack > 1:
            vec = VecFrameStack(vec, n_stack=self.frame_stack)
        return vec

    def build_video_env_fn(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        frame_stack = self.frame_stack
        def _thunk():
            team = QuidditchTeamEnv(cfg=cfg, render_mode="rgb_array")
            opp = from_spec(opp_spec, deterministic=True)
            env = OpponentControlledEnv(team, learner_id=learner, opponent=opp)
            if frame_stack > 1:
                from envs.quidditch.opponents import FrameStackWrapper
                return FrameStackWrapper(env, n_stack=frame_stack)
            return env
        return _thunk
