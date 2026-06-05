import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.quidditch.dict_frame_stack import SelectiveDictFrameStack


class _DictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "actor":  spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "critic": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return {"actor": np.full(2, self._i, np.float32),
                "critic": np.full(3, self._i, np.float32)}, {}

    def step(self, a):
        self._i += 1
        return ({"actor": np.full(2, self._i, np.float32),
                 "critic": np.full(3, self._i, np.float32)}, 0.0,
                self._i >= 5, False, {})


def test_stacks_actor_only():
    vec = SelectiveDictFrameStack(DummyVecEnv([_DictEnv]), n_stack=3, keys=("actor",))
    assert vec.observation_space["actor"].shape == (6,)    # 2 * 3
    assert vec.observation_space["critic"].shape == (3,)   # unstacked
    obs = vec.reset()
    assert obs["actor"].shape == (1, 6)
    assert obs["critic"].shape == (1, 3)
