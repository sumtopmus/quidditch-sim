"""Eval result schema + terminal-cause classification.

The SB3 episode-loop engine (`run_scenario`) was retired with the SB3 path in
migration Step 6; the live RLlib eval battery (`rllib/eval_battery.py`) reuses
the result dataclasses and `_classify_terminal` / `TERMINAL_BUCKETS` defined
here.

Surviving surface:
  - TERMINAL_BUCKETS — the canonical set of episode terminal causes.
  - ScenarioSpec / EpisodeResult / ScenarioResult — eval result dataclasses.
  - _classify_terminal(red_info, blue_info) — map a terminal step's two info
    dicts to a single bucket.
"""
from __future__ import annotations

from dataclasses import dataclass

TERMINAL_BUCKETS = (
    "drone_drone_crash",
    "red_floor", "blue_floor",
    "red_wall",  "blue_wall",
    "red_oob",   "blue_oob",
    "timeout",
    "score",
)


@dataclass(frozen=True)
class ScenarioSpec:
    opponent: str                  # "beeline_red" / "intercepter_red:lookahead=0.5" / "frozen" / ...
    opponent_model_path: str | None  # required when opponent == "frozen"
    randomise_start: bool
    n_episodes: int
    crash_aftermath_seconds: float = 0.0
    deterministic: bool = True
    learner_id: str = "blue_0"
    seed: int = 0


@dataclass(frozen=True)
class EpisodeResult:
    length: int
    reward_learner: float
    reward_opponent: float
    terminal_cause: str            # one of TERMINAL_BUCKETS
    take_down_fired: bool
    score_at_episode_end: int | None  # learner side scored (1), opponent side (-1), neither (None)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioSpec
    episodes: list[EpisodeResult]
    win_rate: float
    mean_reward_learner: float
    mean_reward_opponent: float
    take_down_rate: float
    terminal_cause_counts: dict[str, int]
    mean_episode_length: float


def _classify_terminal(red_info: dict, blue_info: dict) -> str:
    """Map a terminal step's two info dicts to a single TERMINAL_BUCKETS bucket."""
    if red_info.get("scored") or blue_info.get("scored"):
        return "score"
    if red_info.get("drone_drone_crash") or blue_info.get("drone_drone_crash"):
        return "drone_drone_crash"
    if red_info.get("red_floor"):       return "red_floor"
    if red_info.get("red_wall_crash"):  return "red_wall"
    if red_info.get("red_oob"):         return "red_oob"
    if blue_info.get("blue_floor"):     return "blue_floor"
    if blue_info.get("blue_wall_crash"):return "blue_wall"
    if blue_info.get("blue_oob"):       return "blue_oob"
    return "timeout"
