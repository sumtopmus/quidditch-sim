"""Regression: env must emit ``info['is_success']`` so SB3's EvalCallback logs
``eval/success_rate``.

SB3 ``EvalCallback`` only records ``eval/success_rate`` when episode ``info``
dicts carry the key ``is_success`` (``_log_success_callback`` reads
``info.get("is_success")``). The env exposes its success signal as ``scored``;
without an ``is_success`` mirror the success buffer stays empty and the metric is
never logged — silently breaking any campaign whose objective/kill-rules key on
``eval/success_rate`` (caught by the smoke campaign, 2026-06).
"""
from __future__ import annotations

import numpy as np

from envs.quidditch.simple_env import QuidditchSimpleEnv


def test_step_info_carries_is_success_mirroring_scored() -> None:
    """Every step's info dict exposes ``is_success`` == ``bool(scored)``."""
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        env.reset(seed=0)
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            assert "is_success" in info, "info must carry is_success for SB3 EvalCallback"
            assert info["is_success"] == bool(info["scored"])
            if terminated or truncated:
                break
    finally:
        env.close()
