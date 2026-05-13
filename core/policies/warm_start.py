"""warm_start_ppo_by_spec — named-block input-layer surgery from a parent PPO checkpoint.

Routes weight-copy by `(name, dim, frame)` matching between the parent run's
ObsSpec and the current env's spec.  Columns where every triple matches are
copied byte-for-byte; the rest are σ-Gaussian-inited so the warm-started
policy starts behaviorally close to the parent and diverges only as SGD
figures out how to use the new inputs.

The new dim init scale defaults to 0.01 (small).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.quidditch.obs_spec import ObsSpec


def warm_start_ppo_by_spec(
    *,
    old_checkpoint: str | Path,
    new_env: Any,
    parent_spec: ObsSpec,
    parent_n_stack: int,
    current_spec: ObsSpec,
    current_n_stack: int,
    new_dim_init_scale: float = 0.01,
    **ppo_kwargs: Any,
) -> PPO:
    """Generalized input-layer surgery driven by named-block specs.

    Copies columns where (name, dim, frame) match between parent and current
    specs; small-inits the rest.  Repeats parent slices across current_n_stack
    when parent_n_stack=1 and current is >1; otherwise pairs slices in order.

    Non-input layers: copied when shapes match, skipped otherwise.
    """
    old = PPO.load(old_checkpoint)
    new = PPO("MlpPolicy", new_env, **ppo_kwargs)

    old_sd = old.policy.state_dict()
    new_sd = new.policy.state_dict()

    parent_single = parent_spec.dim
    current_single = current_spec.dim
    new_input_dim = current_single * current_n_stack
    old_input_dim = parent_single * parent_n_stack

    # Build column-mapping per single-frame slice: list of (parent_off, current_off, dim).
    matches: list[tuple[int, int, int]] = []
    parent_offsets = {b: sl for b, sl in parent_spec.offsets()}
    current_offsets = {b: sl for b, sl in current_spec.offsets()}
    for cb, c_slice in current_offsets.items():
        for pb, p_slice in parent_offsets.items():
            if pb.name == cb.name and pb.dim == cb.dim and pb.frame == cb.frame:
                matches.append((p_slice.start, c_slice.start, cb.dim))
                break

    rng = np.random.default_rng()
    for key, old_w in old_sd.items():
        if key not in new_sd:
            continue
        new_w = new_sd[key]
        # Input-layer surgery: 2-D weight matrix whose column count is the input dim.
        # Check this BEFORE the shape-match fast path so that a same-shape but
        # spec-changed (e.g. frame-only) input layer still gets routed through
        # the column-by-column copy.
        is_input = (
            key.endswith(".weight") and old_w.dim() == 2
            and old_w.shape[0] == new_w.shape[0]
            and old_w.shape[1] == old_input_dim
            and new_w.shape[1] == new_input_dim
        )
        if not is_input:
            if old_w.shape == new_w.shape:
                new_sd[key] = old_w
            continue
        patched = new_w.detach().clone()
        # Initialize all columns from a small-init Gaussian; matched columns
        # overwrite below.
        full_random = torch.from_numpy(
            rng.normal(0.0, new_dim_init_scale, size=tuple(patched.shape))
        ).to(dtype=patched.dtype, device=patched.device)
        patched.copy_(full_random)

        for c_slice_idx in range(current_n_stack):
            c_base = c_slice_idx * current_single
            # Pick the parent slice to copy from:
            #   parent_n_stack == 1: repeat parent's only slice into every current slice.
            #   parent_n_stack >= current_n_stack: align by index from the most-recent end.
            if parent_n_stack == 1:
                p_slice_idx = 0
            else:
                # Most-recent slices are at the end (index parent_n_stack-1).
                # When parent has more, take the most-recent current_n_stack slices.
                offset_from_end = (current_n_stack - 1) - c_slice_idx
                p_slice_idx = parent_n_stack - 1 - offset_from_end
                if p_slice_idx < 0:
                    # parent has fewer than the current_n_stack ⇒ leave small-init.
                    continue
            p_base = p_slice_idx * parent_single
            for p_off, c_off, dim in matches:
                patched[:, c_base + c_off : c_base + c_off + dim] = (
                    old_w[:, p_base + p_off : p_base + p_off + dim]
                )
        new_sd[key] = patched

    new.policy.load_state_dict(new_sd)
    return new
