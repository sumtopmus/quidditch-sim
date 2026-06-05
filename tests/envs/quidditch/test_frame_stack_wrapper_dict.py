import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.quidditch.opponents import FrameStackWrapper


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
        return {"actor": np.full(2, 1.0, np.float32),
                "critic": np.full(3, 9.0, np.float32)}, {}

    def step(self, a):
        self._i += 1
        return ({"actor": np.full(2, float(self._i + 1), np.float32),
                 "critic": np.full(3, 9.0, np.float32)}, 0.0,
                self._i >= 5, False, {})


def test_frame_stack_wrapper_stacks_actor_only():
    env = FrameStackWrapper(_DictEnv(), n_stack=3, stack_keys=("actor",))
    assert env.observation_space["actor"].shape == (6,)    # 2 * 3
    assert env.observation_space["critic"].shape == (3,)   # unstacked
    obs, _ = env.reset()
    assert obs["actor"].shape == (6,)
    assert obs["critic"].shape == (3,)
    # reset fills all 3 slots with the initial actor obs (value 1.0)
    np.testing.assert_allclose(obs["actor"], np.full(6, 1.0, np.float32))
    obs, *_ = env.step(env.action_space.sample())
    # oldest first, newest at the tail: [1,1, 1,1, 2,2]
    np.testing.assert_allclose(obs["actor"], [1, 1, 1, 1, 2, 2])
    np.testing.assert_allclose(obs["critic"], [9, 9, 9])
