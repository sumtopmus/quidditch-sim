"""Unit tests for WandbOutputFormat (SB3 logger → wandb.log bridge)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from scripts._wandb_logger import WandbOutputFormat, _to_scalar


# ── _to_scalar (the value coercion helper) ──────────────────────────────────


def test_to_scalar_accepts_python_scalars():
    assert _to_scalar(1) == 1.0
    assert _to_scalar(1.5) == 1.5
    assert _to_scalar(True) == 1.0
    assert _to_scalar(False) == 0.0


def test_to_scalar_accepts_numpy_scalars_and_zero_d_arrays():
    assert _to_scalar(np.float32(2.5)) == 2.5
    assert _to_scalar(np.int64(7)) == 7.0
    assert _to_scalar(np.array(3.14)) == 3.14
    # Length-1 1-D is also OK (SB3 sometimes wraps in a (1,) array).
    assert _to_scalar(np.array([4.2])) == 4.2


def test_to_scalar_rejects_non_scalars():
    assert _to_scalar("string") is None
    assert _to_scalar(np.array([1.0, 2.0])) is None
    assert _to_scalar(np.zeros((3, 3))) is None
    assert _to_scalar(None) is None
    # Object without a __float__ — wrapper types like SB3's Video, Figure
    # land here.
    class _Opaque: pass
    assert _to_scalar(_Opaque()) is None


# ── WandbOutputFormat.write ─────────────────────────────────────────────────


def test_write_forwards_scalar_key_values_to_wandb_log():
    """Typical SB3 PPO dump: nested keys (rollout/train/time/eval) → wandb."""
    writer = WandbOutputFormat()
    key_values = {
        "rollout/ep_rew_mean":  7.3,
        "rollout/ep_len_mean":  500.0,
        "train/loss":           0.42,
        "train/entropy_loss":   -1.1,
        "train/n_updates":      np.int64(120),
        "time/total_timesteps": 12345,
        "eval/mean_reward":     np.float32(5.5),
    }
    key_excluded = {k: () for k in key_values}

    with patch("wandb.log") as mock_log:
        writer.write(key_values, key_excluded, step=12345)

    assert mock_log.call_count == 1
    payload, kwargs = mock_log.call_args.args, mock_log.call_args.kwargs
    assert kwargs["step"] == 12345
    sent = payload[0]
    assert sent["rollout/ep_rew_mean"]  == 7.3
    assert sent["rollout/ep_len_mean"]  == 500.0
    assert sent["train/loss"]           == 0.42
    assert sent["train/entropy_loss"]   == -1.1
    assert sent["train/n_updates"]      == 120.0
    assert sent["time/total_timesteps"] == 12345.0
    assert sent["eval/mean_reward"]     == 5.5


def test_write_skips_non_scalar_values_silently():
    """Video / Figure / Image / multi-dim ndarrays are non-scalar and
    must be skipped (not raise).  Only scalar keys reach wandb.log."""
    writer = WandbOutputFormat()
    key_values = {
        "train/loss":          0.1,
        "eval/video":          np.zeros((4, 3, 60, 80), dtype=np.uint8),  # video tensor
        "train/grad_hist":     np.array([1.0, 2.0, 3.0]),                  # multi-elem
        "train/note":          "string-valued",                            # str
    }
    key_excluded = {k: () for k in key_values}

    with patch("wandb.log") as mock_log:
        writer.write(key_values, key_excluded, step=1)

    payload = mock_log.call_args.args[0]
    assert set(payload.keys()) == {"train/loss"}
    assert payload["train/loss"] == 0.1


def test_write_honors_wandb_in_key_excluded():
    """A future caller can mark a key as wandb-opt-out via key_excluded."""
    writer = WandbOutputFormat()
    key_values = {"train/loss": 0.1, "train/secret": 99.0}
    key_excluded = {
        "train/loss":   (),
        "train/secret": ("wandb",),
    }

    with patch("wandb.log") as mock_log:
        writer.write(key_values, key_excluded, step=2)

    payload = mock_log.call_args.args[0]
    assert set(payload.keys()) == {"train/loss"}


def test_write_skips_call_when_no_scalars_to_log():
    """All-non-scalar dump should not call wandb.log at all (avoids
    empty-payload logs that would bump the wandb step counter)."""
    writer = WandbOutputFormat()
    key_values = {"eval/video": np.zeros((4, 3, 60, 80), dtype=np.uint8)}
    key_excluded = {"eval/video": ()}

    with patch("wandb.log") as mock_log:
        writer.write(key_values, key_excluded, step=3)

    assert mock_log.call_count == 0


def test_write_noop_after_close():
    """KVWriter.close marks the writer dead; subsequent writes are no-ops."""
    writer = WandbOutputFormat()
    writer.close()
    with patch("wandb.log") as mock_log:
        writer.write({"x": 1.0}, {"x": ()}, step=4)
    assert mock_log.call_count == 0


def test_writer_implements_kvwriter_protocol():
    """Sanity: instances of WandbOutputFormat must satisfy isinstance(KVWriter)
    so SB3's logger accepts them in output_formats."""
    from stable_baselines3.common.logger import KVWriter
    assert isinstance(WandbOutputFormat(), KVWriter)


# ── WandbLoggerCallback (the attaching mechanism) ───────────────────────────


def test_callback_attaches_writer_at_init():
    """_init_callback must append WandbOutputFormat to model.logger.output_formats."""
    from unittest.mock import MagicMock
    from scripts._wandb_logger import WandbLoggerCallback

    cb = WandbLoggerCallback()
    cb.model = MagicMock()
    cb.model.logger.output_formats = []  # SB3 creates this in _setup_learn
    cb._init_callback()
    assert len(cb.model.logger.output_formats) == 1
    assert isinstance(cb.model.logger.output_formats[0], WandbOutputFormat)


def test_callback_is_idempotent_against_double_init():
    """If a WandbOutputFormat is already attached (e.g. resume re-entering
    learn()), don't double-append."""
    from unittest.mock import MagicMock
    from scripts._wandb_logger import WandbLoggerCallback

    cb = WandbLoggerCallback()
    cb.model = MagicMock()
    cb.model.logger.output_formats = [WandbOutputFormat()]
    cb._init_callback()
    assert len(cb.model.logger.output_formats) == 1


def test_callback_on_step_is_noop_returning_true():
    """The callback's only job is the init-time attach; _on_step must
    not block training or fire on each step."""
    from unittest.mock import MagicMock
    from scripts._wandb_logger import WandbLoggerCallback
    cb = WandbLoggerCallback()
    cb.model = MagicMock()
    assert cb._on_step() is True
