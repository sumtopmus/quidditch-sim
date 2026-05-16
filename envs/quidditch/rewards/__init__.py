"""Reward stack factory helpers — `conf/reward/*.yaml` is the canonical source.

Before 2026-05-15: this module exported per-magnitude Python constants
(`SCORE_REWARD`, `CRASH_PENALTY`, …) AND `conf/reward/*.yaml` declared the
same magnitudes in `_target_` instantiation form, with each env building its
own stack from the constants and the YAML side orphaned (read by `train.py`,
discarded).  Two sources of truth needing manual lockstep.

Now: the YAML files are the single source of truth for reward magnitudes
and term composition.  Envs accept an optional `reward_stack` kwarg and fall
back to one of these helpers, which load the canonical YAML and
Hydra-instantiate it.  Physical / geometric thresholds that *gate* reward
events (`TAG_RADIUS`, `CRASH_VEL_THR`, …) still live in
`envs.quidditch.constants` — they aren't reward magnitudes themselves.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from envs.quidditch.rewards.stack import RewardStack

__all__ = [
    "DEFAULT_MIDPOINT_ALPHA",
    "default_simple_stack",
    "default_team_stack",
    "load_reward_stack",
]

# Defender shaping/tactics: target midpoint = α·red_pos + (1−α)·hoop_center.
# Not a reward magnitude — it parameterises a *target* whose distance gets
# fed to HoopDistancePenalty.  Lives here (rather than in `constants.py`)
# because it's reward-shaping-adjacent; not in the YAML because it gets read
# by `TeamConfig`, not by a reward term.
DEFAULT_MIDPOINT_ALPHA: float = 0.5


def _conf_reward_dir() -> Path:
    # envs/quidditch/rewards/__init__.py → repo/conf/reward/
    return Path(__file__).resolve().parents[3] / "conf" / "reward"


@lru_cache(maxsize=8)
def load_reward_stack(yaml_name: str) -> RewardStack:
    """Load + Hydra-instantiate one `conf/reward/<yaml_name>.yaml`.

    Cached: each YAML is read once per process.  The returned `RewardStack`
    is mutable (its `terms` list could be appended to), so callers wanting
    isolation should build a fresh stack instead of mutating this one.
    Today no caller mutates.
    """
    p = _conf_reward_dir() / f"{yaml_name}.yaml"
    cfg = OmegaConf.load(p)
    return instantiate(cfg, _convert_="all")


def default_simple_stack() -> RewardStack:
    """Single-agent canonical stack (matches `conf/reward/single_agent.yaml`).

    Pins the scoring canary at `step 434 / reward 7.3837`.
    """
    return load_reward_stack("single_agent")


def default_team_stack() -> RewardStack:
    """Team canonical stack (matches `conf/reward/team_v2.yaml`).

    Post-2026-05-12: includes proximity-graded tag + closing-velocity bonus
    + hoop anchor + zero-sum hoop-distance mirror.  Pins the team canary.

    Note: agent IDs in this YAML are hardcoded as `red_0` / `blue_0`.  If
    `TeamConfig.red_prefix` / `blue_prefix` ever differ, callers must build
    a custom stack with matching IDs and pass it via `reward_stack=`.
    """
    return load_reward_stack("team_v2")
