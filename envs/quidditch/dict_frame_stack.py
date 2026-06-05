"""VecFrameStack variant that stacks only selected keys of a Dict obs.

SB3's VecFrameStack stacks every key of a Dict space with the same depth; the
CTDE critic view already encodes temporal/future content, so we stack only the
actor key and pass the critic key through.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


class SelectiveDictFrameStack(VecEnvWrapper):
    def __init__(self, venv, n_stack: int, keys: tuple[str, ...] = ("actor",)) -> None:
        assert isinstance(venv.observation_space, spaces.Dict)
        self._keys = tuple(keys)
        new_spaces: dict = {}
        self._stackers: dict[str, StackedObservations] = {}
        for key, sub in venv.observation_space.spaces.items():
            if key in self._keys:
                st = StackedObservations(venv.num_envs, n_stack, sub)
                self._stackers[key] = st
                new_spaces[key] = st.stacked_observation_space
            else:
                new_spaces[key] = sub
        super().__init__(venv, observation_space=spaces.Dict(new_spaces))

    def reset(self):
        obs = self.venv.reset()
        out = dict(obs)
        for key, st in self._stackers.items():
            out[key] = st.reset(np.asarray(obs[key]))
        return out

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        out = dict(obs)
        for key, st in self._stackers.items():
            stacked, infos = st.update(np.asarray(obs[key]), dones, infos)
            out[key] = stacked
        return out, rewards, dones, infos
