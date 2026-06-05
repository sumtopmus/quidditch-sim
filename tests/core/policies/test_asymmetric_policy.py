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
