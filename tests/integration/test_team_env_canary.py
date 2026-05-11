"""Beeline-vs-beeline deterministic canary for QuidditchTeamEnv.

Ported from scripts/check_team_env.py.

Locked fingerprint (2026-05-11, after proximity + closing-velocity shaping):
    step 176: TAG_ENTRY            blue +5.080  red -5.084
    step 684: TERMINATED (SCORE)   blue -10.005 red +9.999
    EPISODE END: red_total +1.542, blue_total -7.229
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = pytest.mark.slow


def _beeline_action(obs: np.ndarray) -> np.ndarray:
    """obs[12:15] is unit_to_goal — maps to (dx, dy, dyaw=0, dz)."""
    return np.clip(
        np.array([obs[12], obs[13], 0.0, obs[14]], dtype=np.float32),
        -1.0, 1.0,
    )


def test_beeline_red_vs_beeline_blue_canary() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        obs, _ = env.reset(seed=42)
        total = {"red_0": 0.0, "blue_0": 0.0}
        step = 0

        seen_tag_entry_step: int | None = None
        seen_tag_entry_rew: dict[str, float] = {}

        seen_terminate_step: int | None = None
        seen_terminate_rew: dict[str, float] = {}
        seen_terminate_info: dict | None = None

        while env.agents:
            actions = {
                "red_0":  _beeline_action(obs["red_0"]),
                "blue_0": _beeline_action(obs["blue_0"]),
            }
            obs, rew, term, _trunc, info = env.step(actions)
            step += 1
            total["red_0"]  += rew["red_0"]
            total["blue_0"] += rew["blue_0"]

            if seen_tag_entry_step is None and info["red_0"].get("tag_entry"):
                seen_tag_entry_step = step
                seen_tag_entry_rew = dict(rew)

            if any(term.values()):
                seen_terminate_step = step
                seen_terminate_rew = dict(rew)
                seen_terminate_info = info
                break

        assert seen_tag_entry_step == 176, (
            f"expected first tag_entry at step 176, got {seen_tag_entry_step}"
        )
        assert seen_tag_entry_rew["blue_0"] == pytest.approx(+5.080, abs=1e-3)
        assert seen_tag_entry_rew["red_0"]  == pytest.approx(-5.084, abs=1e-3)

        assert seen_terminate_step == 684, (
            f"expected SCORE termination at step 684, got {seen_terminate_step}"
        )
        assert seen_terminate_info is not None
        assert seen_terminate_info["red_0"]["scored"], (
            f"expected SCORE termination, got info={seen_terminate_info}"
        )
        assert seen_terminate_rew["blue_0"] == pytest.approx(-10.005, abs=1e-3)
        assert seen_terminate_rew["red_0"]  == pytest.approx(+9.999,  abs=1e-3)

        assert total["red_0"]  == pytest.approx(+1.542, abs=1e-3)
        assert total["blue_0"] == pytest.approx(-7.229, abs=1e-3)
    finally:
        env.close()
