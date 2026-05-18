"""SB3-Logger → wandb.log bridge.

`WandbOutputFormat` is a `KVWriter` that, when appended to
`model.logger.output_formats`, ships every key/value pair SB3's internal
logger dumps (rollout/*, train/*, time/*, eval/*) to `wandb.log` keyed by
the training step.

This is the metric-forwarding piece the 2026-05-14 W&B migration left
unbuilt: it retired `tensorboard_log=...` (which gave wandb's TB sync a
free ride on SB3's metrics) without adding an explicit alternative.  The
official `wandb.integration.sb3.WandbCallback` only handles hyperparam
capture + optional model checkpoints — it does NOT forward metrics.

The `KVWriter` interface is `write(key_values, key_excluded, step)`;
`key_excluded` is `dict[str, tuple[str, ...]]` where excluded backends are
named (e.g. `"tensorboard"`, `"stdout"`).  We honor `"wandb"` as our
opt-out key.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class WandbOutputFormat(KVWriter):
    """Forward SB3 logger key/value dumps to wandb.log keyed by training step.

    Only forwards scalar values (int, float, numpy scalar, 0-d ndarray).
    SB3's `Video`, `Figure`, `Image`, `HParam` types are skipped — videos
    are handled by our own VideoRecorderCallback; the others aren't
    emitted by PPO or EvalCallback today.  Adding richer types is a
    one-isinstance-branch follow-up if a future term needs it.

    Honors `"wandb"` in `key_excluded[key]` as an opt-out so a future
    caller can mark a key as TB/stdout-only.
    """

    def __init__(self) -> None:
        self._is_closed = False

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        if self._is_closed:
            return
        import wandb

        payload: dict[str, float] = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key) or ()
            if "wandb" in excluded:
                continue
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            payload[key] = scalar
        if payload:
            wandb.log(payload, step=int(step))

    def close(self) -> None:
        self._is_closed = True


def _to_scalar(value: Any) -> float | None:
    """Return a Python float if `value` is a scalar SB3 metric, else None.

    Accepts: bool, int, float, numpy scalar, 0-d numpy array, length-1
    1-d numpy array.  Rejects: strings, multi-element arrays, Video /
    Figure / Image / HParam SB3 wrapper types (no isinstance import
    needed — they fall through the type checks here).
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        if value.ndim == 1 and value.shape[0] == 1:
            return float(value.item())
        return None
    if isinstance(value, np.generic):
        return float(value.item())
    return None


class WandbLoggerCallback(BaseCallback):
    """Attaches a `WandbOutputFormat` to the model's logger at training start.

    SB3 creates `model.logger` lazily inside `_setup_learn` (the first thing
    `model.learn()` does), so the writer can't be appended at construction
    time — `model._logger` doesn't exist yet.  A callback's `_init_callback`
    hook fires AFTER `_setup_learn`, which is the right window.

    Idempotent: if a `WandbOutputFormat` is already attached (e.g. via
    `init=resume` re-attaching to an existing logger), don't double-append.
    """

    def _init_callback(self) -> None:
        formats = self.model.logger.output_formats
        if not any(isinstance(f, WandbOutputFormat) for f in formats):
            formats.append(WandbOutputFormat())

    def _on_step(self) -> bool:
        return True
