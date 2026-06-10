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

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

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


def _refresh_mapping_fn(algorithm, mapping_fn) -> None:
    """Reinstall the league mapping fn on every env runner. Needed on restore,
    where the modules are present but the config's serialized mapping fn still
    reflects empty populations. During normal snapshotting, add_module's
    new_agent_to_module_mapping_fn handles the refresh instead.
    """
    def _set(runner, fn=mapping_fn):
        # The env runner reads self.config.policy_mapping_fn when creating each
        # episode; overwrite it in place. If the config is frozen in this Ray
        # version, fall back to a thawed copy.
        try:
            runner.config.policy_mapping_fn = fn
        except Exception:
            cfg = runner.config.copy(copy_frozen=False)
            cfg.policy_mapping_fn = fn
            runner.config = cfg

    algorithm.env_runner_group.foreach_env_runner(_set, local_env_runner=True)


_SIDES = (
    ("red", "red_score_rate", "main_red", RED_POP_RE),
    ("blue", "blue_prevention_rate", "main_blue", BLUE_POP_RE),
)


class LeagueCallback(RLlibCallback):
    """Drives snapshot-population growth. Reads the in-training metrics off the
    train result and freezes a snapshot of a main into its population when its
    win-rate clears the threshold (with a cooldown + population cap). Membership
    is derived from module ids, so restore is faithful with no extra state."""

    def __init__(self):
        super().__init__()
        self._cfg: dict = {}
        self._last_snapshot_iter = {"red": 0, "blue": 0}

    def _league_cfg(self, algorithm) -> dict:
        return dict(algorithm.config.env_config.get("league", {}))

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._last_snapshot_iter = {"red": it, "blue": it}
        module_ids = set(algorithm.get_module().keys())
        _refresh_mapping_fn(algorithm, make_league_mapping_fn(module_ids, self._cfg))

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        for side, metric_name, main_id, regex in _SIDES:
            metric = read_metric(result, metric_name)
            if metric is None:
                continue
            module_ids = set(algorithm.get_module().keys())
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg["snapshot_threshold"]),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(population_members(module_ids, regex)),
                cap=int(self._cfg["population_cap"]),
            ):
                self._snapshot(algorithm, main_id, regex, module_ids)
                self._last_snapshot_iter[side] = it

    def _snapshot(self, algorithm, main_id, regex, module_ids) -> None:
        new_id = (
            f"red_pop_v{next_version(module_ids, regex)}" if regex is RED_POP_RE
            else f"blue_pop_v{next_version(module_ids, regex)}"
        )
        new_mapping_fn = make_league_mapping_fn(module_ids | {new_id}, self._cfg)
        # Add a fresh (default-arch) module, kept out of the gradient update, and
        # refresh the mapping fn across the env-runner group in the same call.
        algorithm.add_module(
            module_id=new_id,
            module_spec=RLModuleSpec(),
            new_should_module_be_updated=["main_red", "main_blue"],
            new_agent_to_module_mapping_fn=new_mapping_fn,
        )
        # Copy the live main's weights into the frozen snapshot on the learner(s)...
        algorithm.learner_group.foreach_learner(
            lambda lrnr, m=main_id, n=new_id: lrnr.module[n].set_state(
                lrnr.module[m].get_state()
            )
        )
        # ...then push the snapshot's weights out to all env runners.
        algorithm.env_runner_group.sync_weights(
            policies=[new_id],
            from_worker_or_learner_group=algorithm.learner_group,
            inference_only=True,
        )
