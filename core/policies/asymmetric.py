"""AsymmetricActorCriticPolicy — privileged-critic (CTDE) PPO policy for SB3.

The actor (policy) network reads only obs["actor"]; the value network reads
the concatenation of the chosen critic keys (obs["actor"] + obs["critic"]).
Because the critic key never enters the policy branch, the action distribution
has zero gradient w.r.t. obs["critic"] — the privileged info lowers value
variance without biasing the policy gradient (Pinto et al. 2017).

See docs/superpowers/specs/2026-06-04-oracle-critic-obs-design.md §6.2.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _KeyedExtractor(BaseFeaturesExtractor):
    """Flatten + concat a fixed subset of a Dict obs's keys, in key order."""

    def __init__(self, observation_space: spaces.Dict, keys: tuple[str, ...]) -> None:
        dim = sum(int(np.prod(observation_space[k].shape)) for k in keys)
        super().__init__(observation_space, features_dim=dim)
        self._keys = tuple(keys)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([torch.flatten(obs[k], start_dim=1) for k in self._keys], dim=1)


def _make_trunk(in_dim: int, layers: list[int], activation_fn) -> tuple[nn.Sequential, int]:
    mods: list[nn.Module] = []
    last = in_dim
    for h in layers:
        mods.append(nn.Linear(last, h))
        mods.append(activation_fn())
        last = h
    return nn.Sequential(*mods), last


class _AsymmetricMlpExtractor(nn.Module):
    """Like SB3's MlpExtractor but with independent input dims for pi and vf."""

    def __init__(self, *, pi_dim: int, vf_dim: int, net_arch, activation_fn, device) -> None:
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]
        if isinstance(net_arch, dict):
            pi_layers = list(net_arch.get("pi", []))
            vf_layers = list(net_arch.get("vf", []))
        else:
            pi_layers = vf_layers = list(net_arch)
        self.policy_net, self.latent_dim_pi = _make_trunk(pi_dim, pi_layers, activation_fn)
        self.value_net,  self.latent_dim_vf = _make_trunk(vf_dim, vf_layers, activation_fn)
        self.policy_net.to(device)
        self.value_net.to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """PPO policy whose value net sees more of the Dict obs than the actor.

    actor_key:   the single Dict key the policy network reads.
    critic_keys: the Dict keys the value network reads (concatenated, in order).
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args,
                 actor_key: str = "actor",
                 critic_keys: tuple[str, ...] = ("actor", "critic"),
                 **kwargs):
        self._actor_key = actor_key
        self._critic_keys = tuple(critic_keys)
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Replace the base class's two identical CombinedExtractors with keyed
        # ones, then build an MLP extractor with per-head input dims.
        self.pi_features_extractor = _KeyedExtractor(
            self.observation_space, (self._actor_key,)).to(self.device)
        self.vf_features_extractor = _KeyedExtractor(
            self.observation_space, self._critic_keys).to(self.device)
        self.mlp_extractor = _AsymmetricMlpExtractor(
            pi_dim=self.pi_features_extractor.features_dim,
            vf_dim=self.vf_features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(actor_key=self._actor_key, critic_keys=self._critic_keys)
        return data
