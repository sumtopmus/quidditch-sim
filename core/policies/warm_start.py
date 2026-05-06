"""warm_start_ppo — input-layer-augmented warm-start from a single-agent PPO checkpoint.

The old single-agent policy was trained on a 16-d obs.  The team env's per-agent
obs is 22-d, but slots 0:15 are byte-for-byte the same encoding (see the team-play
design spec).  We can therefore warm-start by:

  - building a fresh PPO with the new (22-d) obs space
  - copying the old policy's state_dict into the new policy
  - augmenting the *input* layer's weight matrix from (hidden, 16) to (hidden, 22)
    by writing the old weights into columns 0:16 and small-init Gaussian into
    columns 16:22

The new dim init scale defaults to 0.01 (small), so the warm-started policy starts
behaviorally close to the old single-agent policy and diverges only as SGD figures
out how to use the opponent-relative inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO


def warm_start_ppo(
    *,
    old_checkpoint: str | Path,
    new_env: Any,
    new_input_dim: int = 22,
    old_input_dim: int = 16,
    new_dim_init_scale: float = 0.01,
    **ppo_kwargs: Any,
) -> PPO:
    """Build a fresh PPO and copy weights from `old_checkpoint`, augmenting
    the input layer from `old_input_dim` to `new_input_dim`.

    Args:
        old_checkpoint: path to old PPO best_model.zip (16-d obs space).
        new_env: VecEnv with 22-d obs space.
        new_input_dim, old_input_dim: input dimensions.
        new_dim_init_scale: stddev for the N(0, σ²) init of the new input columns.
        **ppo_kwargs: forwarded to PPO("MlpPolicy", new_env, ...).
    """
    old = PPO.load(old_checkpoint)
    new = PPO("MlpPolicy", new_env, **ppo_kwargs)

    old_sd = old.policy.state_dict()
    new_sd = new.policy.state_dict()

    rng = np.random.default_rng()
    for key, old_w in old_sd.items():
        if key not in new_sd:
            continue
        new_w = new_sd[key]
        if old_w.shape == new_w.shape:
            new_sd[key] = old_w
        elif (
            key.endswith(".weight") and old_w.dim() == 2
            and old_w.shape[0] == new_w.shape[0]
            and old_w.shape[1] == old_input_dim
            and new_w.shape[1] == new_input_dim
        ):
            patched = new_w.detach().clone()
            patched[:, :old_input_dim] = old_w
            extra = torch.from_numpy(
                rng.normal(0.0, new_dim_init_scale,
                           size=(new_w.shape[0], new_input_dim - old_input_dim)),
            ).to(dtype=patched.dtype, device=patched.device)
            patched[:, old_input_dim:] = extra
            new_sd[key] = patched

    new.policy.load_state_dict(new_sd)
    return new
