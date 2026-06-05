import numpy as np
import torch
from gymnasium import spaces

from core.policies.asymmetric import _KeyedExtractor, _AsymmetricMlpExtractor


def _dict_space(actor_dim=6, critic_dim=4):
    return spaces.Dict({
        "actor":  spaces.Box(-np.inf, np.inf, (actor_dim,), np.float32),
        "critic": spaces.Box(-np.inf, np.inf, (critic_dim,), np.float32),
    })


def test_keyed_extractor_selects_and_concats_keys():
    sp = _dict_space(6, 4)
    pi_ext = _KeyedExtractor(sp, ("actor",))
    vf_ext = _KeyedExtractor(sp, ("actor", "critic"))
    assert pi_ext.features_dim == 6
    assert vf_ext.features_dim == 10
    obs = {
        "actor":  torch.zeros(2, 6),
        "critic": torch.ones(2, 4),
    }
    assert pi_ext(obs).shape == (2, 6)
    out = vf_ext(obs)
    assert out.shape == (2, 10)
    # actor slice zeros, critic slice ones (concat order = keys order).
    assert torch.allclose(out[:, :6], torch.zeros(2, 6))
    assert torch.allclose(out[:, 6:], torch.ones(2, 4))


def test_asymmetric_mlp_extractor_has_separate_input_dims():
    ext = _AsymmetricMlpExtractor(pi_dim=6, vf_dim=10, net_arch=[8, 8],
                                  activation_fn=torch.nn.Tanh, device="cpu")
    assert ext.latent_dim_pi == 8
    assert ext.latent_dim_vf == 8
    lat_pi = ext.forward_actor(torch.zeros(3, 6))
    lat_vf = ext.forward_critic(torch.zeros(3, 10))
    assert lat_pi.shape == (3, 8)
    assert lat_vf.shape == (3, 8)


from core.policies.asymmetric import AsymmetricActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class _MockDictEnv(gym.Env):
    """Tiny Dict-obs env: actor=6-d, critic=4-d, action=2-d Box."""
    def __init__(self):
        super().__init__()
        self.observation_space = _dict_space(6, 4)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0

    def _obs(self):
        return {
            "actor":  self.observation_space["actor"].sample() * 0.0 + 0.1,
            "critic": self.observation_space["critic"].sample() * 0.0 + 0.2,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, self._t >= 16, False, {}


def _make_policy():
    env = DummyVecEnv([_MockDictEnv])
    lr = lambda _: 3e-4
    return AsymmetricActorCriticPolicy(
        env.observation_space, env.action_space, lr,
        net_arch=[8, 8],
    )


def test_policy_constructs_with_split_dims():
    pol = _make_policy()
    assert pol.pi_features_extractor.features_dim == 6
    assert pol.vf_features_extractor.features_dim == 10


def test_critic_obs_has_no_gradient_path_to_action():
    pol = _make_policy()
    obs = {
        "actor":  torch.full((1, 6), 0.1, requires_grad=False),
        "critic": torch.full((1, 4), 0.2, requires_grad=True),
    }
    dist = pol.get_distribution(obs)
    logp = dist.log_prob(torch.zeros(1, 2))
    logp.sum().backward()
    # The action distribution must not depend on the critic key.
    assert obs["critic"].grad is None or torch.allclose(
        obs["critic"].grad, torch.zeros_like(obs["critic"].grad))


def test_critic_obs_does_affect_value():
    pol = _make_policy()
    base = {"actor": torch.full((1, 6), 0.1), "critic": torch.full((1, 4), 0.2)}
    other = {"actor": torch.full((1, 6), 0.1), "critic": torch.full((1, 4), 5.0)}
    v0 = pol.predict_values(base)
    v1 = pol.predict_values(other)
    assert not torch.allclose(v0, v1)


from stable_baselines3 import PPO


def test_ppo_learns_and_round_trips(tmp_path):
    env = DummyVecEnv([_MockDictEnv])
    model = PPO(AsymmetricActorCriticPolicy, env,
                policy_kwargs=dict(net_arch=[8, 8]),
                n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=64)
    p = tmp_path / "m.zip"
    model.save(str(p))
    loaded = PPO.load(str(p), env=env)            # reconstructs the custom policy
    obs = env.reset()
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert a1.shape == a2.shape
    np.testing.assert_allclose(a1, a2, atol=1e-5)  # constructor params survived save/load
