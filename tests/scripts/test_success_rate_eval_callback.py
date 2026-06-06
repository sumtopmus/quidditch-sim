"""SuccessRateEvalCallback selects best_model by eval/success_rate (honest
prevention) instead of mean reward — so a high-reward staller can never be
saved as best_model (HANDOFF Issue #13).  Selection key is the tuple
(success_rate, mean_reward): success dominates, reward breaks ties.  When the
eval env provides no `is_success` (e.g. the single-agent simple env), the
success component is a constant 0.0 so selection degrades gracefully to pure
reward selection — parity with stock EvalCallback.

These tests exercise the decision logic (`_maybe_save_best`) in isolation,
constructing the callback via __new__ so no eval env is needed.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np

from scripts.callbacks import SuccessRateEvalCallback


def _make_cb(save_path: str, *, verbose: int = 0) -> SuccessRateEvalCallback:
    cb = SuccessRateEvalCallback.__new__(SuccessRateEvalCallback)
    cb._best_key = (-np.inf, -np.inf)
    cb._is_success_buffer = []
    cb.last_mean_reward = -np.inf
    cb.best_model_save_path = save_path
    cb.verbose = verbose
    cb.model = MagicMock()
    return cb


def _eval(cb, *, success_rate: float | None, reward: float) -> bool:
    """Simulate one finished evaluation, then run the save decision."""
    if success_rate is None:
        cb._is_success_buffer = []
    else:
        # buffer of 0/1 outcomes whose mean is success_rate
        cb._is_success_buffer = [1.0] * round(success_rate * 10) + [0.0] * (
            10 - round(success_rate * 10)
        )
    cb.last_mean_reward = reward
    return cb._maybe_save_best(cb.best_model_save_path)


def test_first_eval_saves_and_records_success_rate(tmp_path) -> None:
    cb = _make_cb(str(tmp_path))
    saved = _eval(cb, success_rate=0.4, reward=250.0)
    assert saved is True
    assert cb.best_success_rate == 0.4
    cb.model.save.assert_called_once_with(os.path.join(str(tmp_path), "best_model"))


def test_higher_success_lower_reward_replaces_staller(tmp_path) -> None:
    """The core anti-stall guarantee: a 70%-prevention defender with LOWER
    reward must beat a 40%-prevention staller with higher reward."""
    cb = _make_cb(str(tmp_path))
    _eval(cb, success_rate=0.4, reward=250.0)  # staller, high reward
    cb.model.save.reset_mock()

    saved = _eval(cb, success_rate=0.7, reward=50.0)  # real defender, low reward

    assert saved is True
    assert cb.best_success_rate == 0.7
    cb.model.save.assert_called_once_with(os.path.join(str(tmp_path), "best_model"))


def test_higher_reward_cannot_override_lower_success(tmp_path) -> None:
    cb = _make_cb(str(tmp_path))
    _eval(cb, success_rate=0.7, reward=50.0)
    cb.model.save.reset_mock()

    saved = _eval(cb, success_rate=0.5, reward=1000.0)

    assert saved is False
    assert cb.best_success_rate == 0.7
    cb.model.save.assert_not_called()


def test_reward_breaks_ties_at_equal_success(tmp_path) -> None:
    cb = _make_cb(str(tmp_path))
    _eval(cb, success_rate=0.7, reward=50.0)
    cb.model.save.reset_mock()

    saved = _eval(cb, success_rate=0.7, reward=80.0)

    assert saved is True
    cb.model.save.assert_called_once()


def test_equal_success_lower_reward_does_not_save(tmp_path) -> None:
    cb = _make_cb(str(tmp_path))
    _eval(cb, success_rate=0.7, reward=80.0)
    cb.model.save.reset_mock()

    saved = _eval(cb, success_rate=0.7, reward=80.0)  # no improvement

    assert saved is False
    cb.model.save.assert_not_called()


def test_no_success_info_falls_back_to_reward_selection(tmp_path) -> None:
    """Single-agent simple env provides no is_success -> behaves like a plain
    reward-selecting EvalCallback."""
    cb = _make_cb(str(tmp_path))
    assert _eval(cb, success_rate=None, reward=10.0) is True
    cb.model.save.reset_mock()
    assert _eval(cb, success_rate=None, reward=5.0) is False  # worse reward
    cb.model.save.assert_not_called()
    assert _eval(cb, success_rate=None, reward=20.0) is True  # better reward


def test_no_save_when_path_is_none(tmp_path) -> None:
    cb = _make_cb(str(tmp_path))
    cb.best_model_save_path = None
    saved = cb._maybe_save_best(None)  # buffer empty, reward -inf -> not > -inf
    # first real eval still updates the key but must not crash on a None path
    cb._is_success_buffer = [1.0, 1.0]
    cb.last_mean_reward = 5.0
    saved = cb._maybe_save_best(None)
    assert saved is True
    assert cb.best_success_rate == 1.0
    cb.model.save.assert_not_called()
