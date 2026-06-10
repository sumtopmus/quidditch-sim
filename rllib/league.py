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


def _episode_rng_roll(episode) -> tuple[float, float]:
    """Two deterministic [0,1) draws keyed on the episode id, so both agents in
    one episode see the same matchup (zlib.crc32 is stable across processes,
    unlike hash() under PYTHONHASHSEED)."""
    seed = zlib.crc32(str(getattr(episode, "id_", episode)).encode())
    roll_mode = (seed % 1_000_003) / 1_000_003
    roll_side = ((seed // 1_000_003) % 1_000_003) / 1_000_003
    return roll_mode, roll_side


def make_league_mapping_fn(module_ids: Iterable[str], league_cfg: dict):
    """Build the per-episode matchup fn (uniform sampling, Step 3).

    With prob `live_fraction`: main_red vs main_blue (both learn). Otherwise
    split 50/50: 'red exploits' (main_red vs uniform blue_pop member) or
    'blue exploits' (uniform red_pop member vs main_blue). An empty opposite
    population makes that exploit mode fall back to live. Closes over plain
    lists/floats only, so it pickles across the Ray boundary.
    """
    red_pop = population_members(module_ids, RED_POP_RE)
    blue_pop = population_members(module_ids, BLUE_POP_RE)
    live_fraction = float(league_cfg.get("live_fraction", 0.5))

    def league_mapping_fn(agent_id, episode, **kw):
        roll_mode, roll_side = _episode_rng_roll(episode)
        mode = "live"
        if roll_mode >= live_fraction:
            if roll_side < 0.5:
                mode = "red_exploits" if blue_pop else "live"
            else:
                mode = "blue_exploits" if red_pop else "live"
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        if mode == "red_exploits":
            if agent_id == "red_0":
                return "main_red"
            return blue_pop[int(roll_side * 2 * len(blue_pop)) % len(blue_pop)]
        # blue_exploits
        if agent_id == "red_0":
            return red_pop[int(roll_side * 2 * len(red_pop)) % len(red_pop)]
        return "main_blue"

    return league_mapping_fn
