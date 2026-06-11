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


WINRATE_DEFAULT = 0.5   # assumed winrate vs an opponent with no data yet


def pfsp_weights(
    winrates: dict[str, float], exponent: float, floor: float
) -> dict[str, float]:
    """Normalized PFSP sampling probabilities over frozen opponents.

    P(o) ∝ (1 − winrate_vs_o)^exponent — concentrate on opponents the main
    struggles against — mixed with a uniform floor so dominated opponents keep
    nonzero mass (anti-forgetting). All-dominated (Σ raw = 0) → uniform.
    """
    if not winrates:
        return {}
    raw = {m: (1.0 - min(max(w, 0.0), 1.0)) ** exponent for m, w in winrates.items()}
    total = sum(raw.values())
    n = len(raw)
    if total <= 0.0:
        return {m: 1.0 / n for m in raw}
    return {m: (1.0 - floor) * v / total + floor / n for m, v in raw.items()}


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


def _episode_roll(episode, salt: str) -> float:
    """Deterministic [0,1) draw keyed on (episode id, salt), so both agents in
    one episode see the same matchup and salts give independent draws.
    zlib.crc32 is stable across processes, unlike hash() under PYTHONHASHSEED.
    """
    eid = str(getattr(episode, "id_", episode))
    return zlib.crc32(f"{eid}:{salt}".encode()) / 2**32


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
        mode = "live"
        if _episode_roll(episode, "mode") >= live_fraction:
            if _episode_roll(episode, "side") < 0.5:
                mode = "red_exploits" if blue_pop else "live"
            else:
                mode = "blue_exploits" if red_pop else "live"
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        member = _episode_roll(episode, "member")
        if mode == "red_exploits":
            if agent_id == "red_0":
                return "main_red"
            return blue_pop[int(member * len(blue_pop)) % len(blue_pop)]
        # blue_exploits
        if agent_id == "red_0":
            return red_pop[int(member * len(red_pop)) % len(red_pop)]
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


def _algo_module_ids(algorithm) -> set[str]:
    """Ids of the modules in the local MultiRLModule — the league population
    store. (Algorithm.get_module(id) returns a single sub-module, defaulting to
    'default_policy' which doesn't exist in a multi-agent setup; the
    MultiRLModule whose keys ARE the population lives on the local env runner,
    with the learner group as a fallback when there is no local env runner.)
    """
    runner = getattr(algorithm, "env_runner", None)
    module = getattr(runner, "module", None) if runner is not None else None
    if module is None:
        return set(
            algorithm.learner_group.foreach_learner(
                lambda lrnr: list(lrnr.module.keys())
            )[0]
        )
    return set(module.keys())


def _snapshot_spec(algorithm, main_id):
    """RLModuleSpec mirroring the live main, for add_module at runtime.

    A bare RLModuleSpec() carries no module_class (the MultiRLModuleSpec fills
    that in at config-build time, not for a runtime add_module), so it raises
    'RLModule class is not set.' when the learner tries to build it. Cloning the
    main's spec via from_module is the canonical RLlib self-play pattern.
    """
    return RLModuleSpec.from_module(algorithm.get_module(main_id))


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
        module_ids = _algo_module_ids(algorithm)
        _refresh_mapping_fn(algorithm, make_league_mapping_fn(module_ids, self._cfg))

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        for side, metric_name, main_id, regex in _SIDES:
            metric = read_metric(result, metric_name)
            if metric is None:
                continue
            module_ids = _algo_module_ids(algorithm)
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
        # Add a fresh (main-arch) module, kept out of the gradient update, and
        # refresh the mapping fn across the env-runner group in the same call.
        algorithm.add_module(
            module_id=new_id,
            module_spec=_snapshot_spec(algorithm, main_id),
            new_should_module_be_updated=["main_red", "main_blue"],
            new_agent_to_module_mapping_fn=new_mapping_fn,
        )
        # Copy the live main's weights into the frozen snapshot on the learner —
        # the authoritative copy, excluded from the gradient update — via the
        # canonical RLlib self-play set_state path.
        main_state = algorithm.get_module(main_id).get_state()
        algorithm.set_state(
            {"learner_group": {"learner": {"rl_module": {new_id: main_state}}}}
        )
        # set_state's learner->env-runner sync is inference-only and lands on the
        # *next* iteration, so the freshly-added env-runner module would serve the
        # frozen opponent with random weights for one iteration. Copy main->snap
        # directly on every env runner (incl. the local one) so the opponent has
        # the main's weights from its very next episode.
        def _seed_snapshot(runner, m=main_id, n=new_id):
            runner.module[n].set_state(runner.module[m].get_state())

        algorithm.env_runner_group.foreach_env_runner(
            _seed_snapshot, local_env_runner=True
        )
