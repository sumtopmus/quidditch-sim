"""Behavior contracts for the scripted 1v1 demo policies.

Headless (no `render_mode`) — these tests assert that the scripted policies
shipped in `demo.takedown_demo` and `demo.score_through_tag_demo` actually hit
their narrative beats against the canonical scoring Red, deterministically
from `seed=42`.
"""
from __future__ import annotations

import pytest

from core.quadrotor import CONTROL_HZ
from demo._team_demo_common import scoring_red_factory
from demo.score_through_tag_demo import _score_through_tag_blue_factory
from demo.takedown_demo import _takedown_blue_factory
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = pytest.mark.slow


def _run_until_done(*, blue_factory, max_seconds: float):
    """Run env to termination/timeout headless; return list of per-step infos."""
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        obs, _ = env.reset(seed=42)
        red_policy  = scoring_red_factory()
        blue_policy = blue_factory()
        max_steps = int(CONTROL_HZ * max_seconds)
        infos = []
        for _ in range(max_steps):
            actions = {
                "red_0":  red_policy(obs["red_0"]),
                "blue_0": blue_policy(obs["blue_0"]),
            }
            obs, _rew, term, trunc, info = env.step(actions)
            infos.append(info)
            if any(term.values()) or any(trunc.values()):
                break
        return infos
    finally:
        env.close()


# ── takedown ─────────────────────────────────────────────────────────────


def test_takedown_red_does_not_score() -> None:
    """The takedown narrative is "Red is prevented from scoring." We don't
    require ``drone_drone_crash`` specifically — Blue's body parked on Red's
    hoop path produces glancing contacts whose ``v_rel · normal`` rarely
    exceeds CRASH_VEL_THR, so the actual termination is typically Red
    bouncing off Blue into the arena wall (``red_wall_crash``). Visually that
    *is* the takedown."""
    infos = _run_until_done(
        blue_factory=_takedown_blue_factory, max_seconds=25.0,
    )
    final = infos[-1]
    assert not final["red_0"].get("scored"), (
        f"expected Red to be prevented from scoring, got final={final}"
    )


def test_takedown_records_at_least_one_tag_entry() -> None:
    infos = _run_until_done(
        blue_factory=_takedown_blue_factory, max_seconds=25.0,
    )
    n = sum(1 for i in infos if i["red_0"].get("tag_entry"))
    assert n >= 1, f"expected ≥1 tag_entry before takedown, got {n}"


def test_takedown_blue_does_not_self_destruct() -> None:
    infos = _run_until_done(
        blue_factory=_takedown_blue_factory, max_seconds=25.0,
    )
    final = infos[-1]
    blue = final["blue_0"]
    assert not blue.get("blue_wall_crash"), "blue wall-crashed (self-destruct)"
    assert not blue.get("blue_floor"),      "blue floor-crashed (self-destruct)"
    assert not blue.get("blue_oob"),        "blue went out-of-bounds (self-destruct)"


# ── score-through-tag ────────────────────────────────────────────────────


def test_score_through_tag_red_scores() -> None:
    infos = _run_until_done(
        blue_factory=_score_through_tag_blue_factory, max_seconds=15.0,
    )
    assert any(i["red_0"].get("scored") for i in infos), (
        "expected Red to score within 15 s"
    )


def test_score_through_tag_records_at_least_one_tag_entry() -> None:
    infos = _run_until_done(
        blue_factory=_score_through_tag_blue_factory, max_seconds=15.0,
    )
    n = sum(1 for i in infos if i["red_0"].get("tag_entry"))
    assert n >= 1, f"expected ≥1 tag_entry before score, got {n}"
