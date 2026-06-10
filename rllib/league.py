"""Step-3 snapshot-population self-play league.

Membership is derived from module-id naming (red_pop_v{N} / blue_pop_v{N}); the
set of modules in the MultiRLModule IS the population, so RLlib's module
checkpointing makes restore faithful with no separate state file.

This module is split into pure helpers (unit-tested) + a league policy_mapping_fn
factory + the LeagueCallback that drives snapshotting on on_train_result.
"""
from __future__ import annotations

import re
import zlib
from typing import Iterable, Optional

RED_POP_RE = re.compile(r"^red_pop_v(\d+)$")
BLUE_POP_RE = re.compile(r"^blue_pop_v(\d+)$")


def population_members(module_ids: Iterable[str], regex: re.Pattern) -> list[str]:
    """Population module ids matching `regex`, sorted ascending by version."""
    matched = [(int(regex.match(m).group(1)), m) for m in module_ids if regex.match(m)]
    return [m for _, m in sorted(matched)]


def next_version(module_ids: Iterable[str], regex: re.Pattern) -> int:
    """Next snapshot version for a side: max existing version + 1, else 1."""
    versions = [int(regex.match(m).group(1)) for m in module_ids if regex.match(m)]
    return (max(versions) + 1) if versions else 1


def read_metric(result: dict, name: str) -> Optional[float]:
    """Recursively find `name` anywhere in the (nested) train result dict.

    ScoreMetricsCallback logs red_score_rate / blue_prevention_rate via the
    MetricsLogger, which nests them under the env-runner results subtree; a
    recursive search avoids hard-coding a version-specific key path.
    """
    if name in result and isinstance(result[name], (int, float)):
        return float(result[name])
    for v in result.values():
        if isinstance(v, dict):
            found = read_metric(v, name)
            if found is not None:
                return found
    return None


def should_snapshot(
    *, metric: float, threshold: float, iters_since_last: int,
    cooldown: int, pop_size: int, cap: int,
) -> bool:
    """Snapshot iff metric clears threshold, cooldown elapsed, and room remains."""
    return metric >= threshold and iters_since_last >= cooldown and pop_size < cap
