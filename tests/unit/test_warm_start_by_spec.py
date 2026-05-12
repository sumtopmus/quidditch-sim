"""warm_start_ppo_by_spec — named-block input-layer surgery."""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import (
    SIMPLE_ENV_OBS, TEAM_ENV_OBS, AUGMENTED_OBS, ObsSpec, ObsBlock,
)
from core.policies.warm_start import warm_start_ppo_by_spec


class _DummyEnv(gym.Env):
    """Minimal gym env with a given obs dim, used to construct fresh PPOs."""
    def __init__(self, obs_dim: int) -> None:
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                 shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,),
                                            dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _make_dummy_vec_env(obs_dim: int):
    return DummyVecEnv([lambda: _DummyEnv(obs_dim)])


def _make_and_save_dummy_ppo(tmp_path, obs_dim: int):
    env = _make_dummy_vec_env(obs_dim)
    ppo = PPO("MlpPolicy", env, n_steps=8, batch_size=4, verbose=0)
    path = tmp_path / "dummy.zip"
    ppo.save(path)
    return path


def _input_layer_weight(ppo: PPO) -> torch.Tensor:
    # MlpExtractor.policy_net[0].weight is the first Linear of the policy net.
    return ppo.policy.mlp_extractor.policy_net[0].weight


def test_simple_to_team_copies_first_16_columns(tmp_path):
    parent_ckpt = _make_and_save_dummy_ppo(tmp_path, obs_dim=SIMPLE_ENV_OBS.dim)
    parent = PPO.load(parent_ckpt)
    old_w = _input_layer_weight(parent).detach().clone()

    new_env = _make_dummy_vec_env(TEAM_ENV_OBS.dim)
    fresh = warm_start_ppo_by_spec(
        old_checkpoint=parent_ckpt,
        new_env=new_env,
        parent_spec=SIMPLE_ENV_OBS, parent_n_stack=1,
        current_spec=TEAM_ENV_OBS,  current_n_stack=1,
    )
    new_w = _input_layer_weight(fresh)
    # SIMPLE_ENV_OBS is a prefix of TEAM_ENV_OBS, so columns 0:16 must be copied
    # byte-for-byte.
    assert torch.allclose(new_w[:, :16], old_w)
    # The added columns 16:22 must be small (σ=0.01 init).
    added = new_w[:, 16:22].detach().cpu().numpy()
    assert np.std(added) < 0.05   # generous bound on σ=0.01 init


def test_team_to_augmented_handles_offset_shift(tmp_path):
    parent_ckpt = _make_and_save_dummy_ppo(tmp_path, obs_dim=TEAM_ENV_OBS.dim)
    parent = PPO.load(parent_ckpt)
    old_w = _input_layer_weight(parent).detach().clone()

    new_env = _make_dummy_vec_env(AUGMENTED_OBS.dim)
    fresh = warm_start_ppo_by_spec(
        old_checkpoint=parent_ckpt,
        new_env=new_env,
        parent_spec=TEAM_ENV_OBS, parent_n_stack=1,
        current_spec=AUGMENTED_OBS, current_n_stack=1,
    )
    new_w = _input_layer_weight(fresh)
    # Blocks shared by (name, dim, frame): ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS,
    # UNIT_TO_GOAL  (slots 0:15 in both), OPP_POS_REL (parent 16:19, current 18:21).
    # signed_dist_norm (parent 15:16) → removed.
    # vec_to_hoop (current 15:18)     → added (small-init).
    # opp_vel_rel: parent body_mixed vs current world → frame diff, small-init.
    # closing_rate (current 24:25)    → added (small-init).
    assert torch.allclose(new_w[:, 0:15], old_w[:, 0:15])
    assert torch.allclose(new_w[:, 18:21], old_w[:, 16:19])  # opp_pos_rel shifted


def test_n_stack_repeat_when_parent_1_current_3(tmp_path):
    parent_ckpt = _make_and_save_dummy_ppo(tmp_path, obs_dim=SIMPLE_ENV_OBS.dim)
    parent = PPO.load(parent_ckpt)
    old_w = _input_layer_weight(parent).detach().clone()

    new_env = _make_dummy_vec_env(SIMPLE_ENV_OBS.dim * 3)
    fresh = warm_start_ppo_by_spec(
        old_checkpoint=parent_ckpt,
        new_env=new_env,
        parent_spec=SIMPLE_ENV_OBS, parent_n_stack=1,
        current_spec=SIMPLE_ENV_OBS, current_n_stack=3,
    )
    new_w = _input_layer_weight(fresh)
    # Each of the 3 slices should be a copy of the parent's input layer.
    for i in range(3):
        sl = slice(i * SIMPLE_ENV_OBS.dim, (i + 1) * SIMPLE_ENV_OBS.dim)
        assert torch.allclose(new_w[:, sl], old_w)


def test_frame_change_does_not_copy_columns(tmp_path):
    # Parent uses body_mixed opp_vel_rel; current uses world.  Same name, same
    # dim, different frame → must be small-init, not copied.
    parent_ckpt = _make_and_save_dummy_ppo(tmp_path, obs_dim=TEAM_ENV_OBS.dim)
    parent = PPO.load(parent_ckpt)
    old_w = _input_layer_weight(parent).detach().clone()

    # Construct a "team-with-world-opp-vel" spec for the test.
    from envs.quidditch.obs_spec import OPP_VEL_REL_WORLD, ANG_VEL, ANG_POS, \
        LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM, OPP_POS_REL
    current = ObsSpec((ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL,
                       SIGNED_DIST_NORM, OPP_POS_REL, OPP_VEL_REL_WORLD))

    new_env = _make_dummy_vec_env(current.dim)
    fresh = warm_start_ppo_by_spec(
        old_checkpoint=parent_ckpt, new_env=new_env,
        parent_spec=TEAM_ENV_OBS, parent_n_stack=1,
        current_spec=current,     current_n_stack=1,
    )
    new_w = _input_layer_weight(fresh)
    # Slots 0:19 match exactly.
    assert torch.allclose(new_w[:, 0:19], old_w[:, 0:19])
    # Slots 19:22 (opp_vel_rel) must NOT equal the parent's — it's a frame change.
    assert not torch.allclose(new_w[:, 19:22], old_w[:, 19:22])
