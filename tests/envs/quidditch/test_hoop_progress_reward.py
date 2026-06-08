"""End-to-end check that team_env tracks Red's previous hoop distance so
HoopApproachShaping (potential-based progress) sees real per-step progress.
"""
from __future__ import annotations

import numpy as np

from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.rewards.stack import RewardStack
from envs.quidditch.rewards.terms import HoopApproachShaping
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig


def test_step_info_exposes_red_hoop_distance():
    """infos['red_0']['dist_red_to_hoop'] lets a metrics callback track how
    close Red gets without re-deriving distance from processed obs."""
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, red_start_pos=(0.5, 0.0, 2.0)))
    try:
        env.reset(seed=0)
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        expected = float(np.linalg.norm(env._red_pos() - HOOP_CENTER))
        assert infos["red_0"]["dist_red_to_hoop"] == expected
    finally:
        env.close()


def test_step_info_exposes_blue_distance_to_red():
    """infos['blue_0']['dist_b2r'] lets the metrics callback track Blue's
    closest approach to Red (its takedown opportunity)."""
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, red_start_pos=(0.5, 0.0, 2.0)))
    try:
        env.reset(seed=0)
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        expected = float(np.linalg.norm(env._red_pos() - env._blue_pos()))
        assert infos["blue_0"]["dist_b2r"] == expected
    finally:
        env.close()


def test_progress_reward_matches_red_motion_one_step():
    """reward == scale * (dist_before - dist_after), proving the env feeds the
    previous-step hoop distance (not the default 0.0) into the term."""
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, red_start_pos=(0.5, 0.0, 2.0)),
        reward_stack=RewardStack(terms=[HoopApproachShaping(scale=1.0, agent="red_0")]),
    )
    try:
        env.reset(seed=0)
        d_before = float(np.linalg.norm(env._red_pos() - HOOP_CENTER))
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, rew, _, _, _ = env.step(actions)
        d_after = float(np.linalg.norm(env._red_pos() - HOOP_CENTER))
        assert abs(rew["red_0"] - (d_before - d_after)) < 1e-9
    finally:
        env.close()


def test_progress_reward_first_step_is_small_not_huge():
    """Guards the prev=0.0 bug: a mis-initialised prev makes the first-step
    reward ≈ -dist_to_hoop (~1.5), not a small per-step delta."""
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, red_start_pos=(0.5, 0.0, 2.0)),
        reward_stack=RewardStack(terms=[HoopApproachShaping(scale=1.0, agent="red_0")]),
    )
    try:
        env.reset(seed=0)
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, rew, _, _, _ = env.step(actions)
        assert abs(rew["red_0"]) < 0.5   # one step of motion, not the full 1.5 m
    finally:
        env.close()
