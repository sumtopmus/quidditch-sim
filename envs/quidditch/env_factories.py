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
    obs_blocks: list[str]
    obs_name: str
    seed: int = 42
    # Reward stack instantiated from cfg.reward in scripts/train.py and set on
    # the factory before env construction.  If None, the env falls back to
    # default_simple_stack() (loaded from conf/reward/single_agent.yaml).
    reward_stack: Any = None

    def _make_thunk(self):
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        from envs.quidditch.obs_spec import build_spec_from_block_names
        rs = self.randomise_start
        eps = self.episode_seconds
        reward_stack = self.reward_stack
        spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode=None, randomise_start=rs, episode_seconds=eps,
                reward_stack=reward_stack, spec=spec,
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
        from envs.quidditch.obs_spec import build_spec_from_block_names
        rs = self.randomise_start
        eps = self.episode_seconds
        reward_stack = self.reward_stack
        spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode="rgb_array", randomise_start=rs, episode_seconds=eps,
                reward_stack=reward_stack, spec=spec,
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
    obs_blocks: list[str]
    obs_name: str
    frame_stack: int = 3
    seed: int = 42
    # See SimpleEnvFactory.reward_stack — same semantics.  Falls back to
    # default_team_stack() (loaded from conf/reward/team_v2.yaml).
    reward_stack: Any = None
    # CTDE (opt-in): when True, the learner emits Dict({actor, critic}) and the
    # specs come from conf/obs/<obs_stem>.yaml via build_ctde_specs_from_yaml.
    ctde_mode: bool = False
    obs_stem: str = ""        # conf/obs/<stem>.yaml for build_ctde_specs_from_yaml

    def _make_thunk(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        from envs.quidditch.obs_spec import (
            build_spec_from_block_names, build_ctde_specs_from_yaml,
        )
        cfg, learner, opp_spec = self.team_cfg, self.learner_id, self.opponent_spec
        reward_stack = self.reward_stack
        ctde, stem = self.ctde_mode, self.obs_stem
        if ctde:
            actor_spec, critic_spec = build_ctde_specs_from_yaml(stem)
        else:
            actor_spec, critic_spec = build_spec_from_block_names(self.obs_blocks), None

        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, reward_stack=reward_stack,
                learner_id=learner, learner_spec=actor_spec,
                ctde_mode=ctde, critic_spec=critic_spec,
            )
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
        return _thunk

    def _wrap_stack(self, vec):
        if self.frame_stack <= 1:
            return vec
        if self.ctde_mode:
            from envs.quidditch.dict_frame_stack import SelectiveDictFrameStack
            return SelectiveDictFrameStack(vec, n_stack=self.frame_stack, keys=("actor",))
        return VecFrameStack(vec, n_stack=self.frame_stack)

    def build_train_env(self) -> VecEnv:
        thunk = self._make_thunk()
        env_fns = [thunk for _ in range(self.n_envs)]
        vec = SubprocVecEnv(env_fns) if self.n_envs > 1 else DummyVecEnv(env_fns)
        return self._wrap_stack(vec)

    def build_eval_env(self) -> VecEnv:
        return self._wrap_stack(DummyVecEnv([self._make_thunk()]))

    def build_video_env_fn(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import (
            OpponentControlledEnv, from_spec, FrameStackWrapper,
        )
        from envs.quidditch.obs_spec import (
            build_spec_from_block_names, build_ctde_specs_from_yaml,
        )
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        frame_stack = self.frame_stack
        reward_stack = self.reward_stack
        ctde, stem = self.ctde_mode, self.obs_stem
        if ctde:
            actor_spec, critic_spec = build_ctde_specs_from_yaml(stem)
        else:
            actor_spec, critic_spec = build_spec_from_block_names(self.obs_blocks), None

        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, render_mode="rgb_array",
                reward_stack=reward_stack,
                learner_id=learner, learner_spec=actor_spec,
                ctde_mode=ctde, critic_spec=critic_spec,
            )
            opp = from_spec(opp_spec, deterministic=True)
            env = OpponentControlledEnv(team, learner_id=learner, opponent=opp)
            if frame_stack > 1:
                return FrameStackWrapper(env, n_stack=frame_stack,
                                         stack_keys=("actor",) if ctde else None)
            return env
        return _thunk
