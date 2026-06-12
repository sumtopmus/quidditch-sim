"""Step-3 snapshot-population self-play league.

Membership is derived from module-id naming (red_pop_v{N} / blue_pop_v{N}); the
set of modules in the MultiRLModule IS the population, so RLlib's module
checkpointing makes restore faithful with no separate state file.

This module is split into pure helpers (unit-tested) + a league policy_mapping_fn
factory + the LeagueCallback that drives snapshotting on on_train_result.
"""
from __future__ import annotations

import math
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
    # Non-finite winrates (NaN from a drained metric window) count as no-data:
    # a single NaN would otherwise poison every probability, and NaN cum-probs
    # make weighted_pick always return the last member (league collapse).
    raw = {
        m: (1.0 - min(max(w if math.isfinite(w) else WINRATE_DEFAULT, 0.0), 1.0))
        ** exponent
        for m, w in winrates.items()
    }
    total = sum(raw.values())
    n = len(raw)
    if total <= 0.0:
        return {m: 1.0 / n for m in raw}
    return {m: (1.0 - floor) * v / total + floor / n for m, v in raw.items()}


def weighted_pick(items: list[str], probs: dict[str, float], roll: float) -> str:
    """Pick the item whose cumulative-probability bucket contains `roll`."""
    acc = 0.0
    for item in items:
        acc += probs[item]
        if roll < acc:
            return item
    return items[-1]   # roll == 1.0 or float round-off past the total


# Rolling window for the per-matchup winrate means. More responsive than the
# MetricsLogger default lifetime EMA (coeff 0.01), which would take hundreds of
# episodes per matchup to move off its initial value. A module constant (not a
# league_cfg knob) because on_episode_end runs on env runners, where the
# callback never sees on_algorithm_init / the league cfg.
WINRATE_WINDOW = 100


def winrate_key(opponent_id: str) -> str:
    """Metric key carrying the main's winrate vs one frozen opponent. The main
    is implied by the population the opponent belongs to (main_red plays
    blue_pop members, main_blue plays red_pop members)."""
    return f"league_wr_vs_{opponent_id}"


def matchup_outcome(
    red_module: Optional[str], blue_module: Optional[str], scored: bool
) -> Optional[tuple[str, float]]:
    """(winrate metric key, main's win 1.0/0.0) for a main-vs-frozen episode.

    Red main wins when it scores; Blue main wins when it prevents the score
    (the same semantics as red_score_rate / blue_prevention_rate). Live
    main-vs-main episodes carry no PFSP signal -> None.
    """
    if red_module == "main_red" and blue_module and BLUE_POP_RE.match(blue_module):
        return winrate_key(blue_module), 1.0 if scored else 0.0
    if blue_module == "main_blue" and red_module and RED_POP_RE.match(red_module):
        return winrate_key(red_module), 0.0 if scored else 1.0
    return None


def read_metric(result: dict, name: str) -> Optional[float]:
    """Recursively find `name` anywhere in the (nested) train result dict.

    ScoreMetricsCallback logs red_score_rate / blue_prevention_rate via the
    MetricsLogger, which nests them under the env-runner results subtree; a
    recursive search avoids hard-coding a version-specific key path.

    Non-finite values read as absent: RLlib's windowed metrics emit NaN once a
    key's window drains (e.g. a matchup with no recent episodes), and a NaN
    winrate must mean "no data", not a number.
    """
    if (
        name in result
        and isinstance(result[name], (int, float))
        and math.isfinite(result[name])
    ):
        return float(result[name])
    for v in result.values():
        if isinstance(v, dict):
            found = read_metric(v, name)
            if found is not None:
                return found
    return None


def collect_winrates(result: dict, module_ids: Iterable[str]) -> dict[str, float]:
    """Current winrate estimate per frozen member, read off the train result.
    Members whose matchup has no logged data yet are omitted (callers default
    them to WINRATE_DEFAULT)."""
    out: dict[str, float] = {}
    for regex in (RED_POP_RE, BLUE_POP_RE):
        for opp in population_members(module_ids, regex):
            value = read_metric(result, winrate_key(opp))
            if value is not None:
                out[opp] = value
    return out


def prune_candidate(
    pop: list[str], winrates: dict[str, float], threshold: float
) -> Optional[str]:
    """The most-dominated member — highest main-winrate among those clearing
    `threshold` — or None when every member still puts up a fight (then the
    population is full of useful opponents and Step-3 stop-at-cap applies)."""
    dominated = [
        (winrates.get(m, WINRATE_DEFAULT), m) for m in pop
        if winrates.get(m, WINRATE_DEFAULT) >= threshold
    ]
    return max(dominated)[1] if dominated else None


def report_league(
    result: dict, module_ids: Iterable[str],
    winrates: dict[str, float], league_cfg: dict,
) -> None:
    """Write league diagnostics into the train result (surfaces in Tune/W&B):
    per-member winrate estimates and the PFSP probabilities actually in force."""
    league = result.setdefault("league", {})
    exponent = float(league_cfg.get("pfsp_exponent", 2.0))
    floor = float(league_cfg.get("pfsp_uniform_floor", 0.1))
    for regex in (RED_POP_RE, BLUE_POP_RE):
        pop = population_members(module_ids, regex)
        probs = pfsp_weights(
            {m: winrates.get(m, WINRATE_DEFAULT) for m in pop}, exponent, floor)
        for m in pop:
            league[f"wr_vs_{m}"] = winrates.get(m, WINRATE_DEFAULT)
            league[f"pfsp_p_{m}"] = probs[m]


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


def make_league_mapping_fn(
    module_ids: Iterable[str],
    league_cfg: dict,
    winrates: Optional[dict[str, float]] = None,
):
    """Build the per-episode matchup fn (PFSP sampling, Step 4).

    With prob `live_fraction`: main_red vs main_blue (both learn). Otherwise
    split 50/50: 'red exploits' (main_red vs a blue_pop member) or 'blue
    exploits' (a red_pop member vs main_blue). Frozen members are drawn PFSP:
    P(o) ∝ (1 − winrate_vs_o)^pfsp_exponent + uniform floor; opponents with no
    win-rate data yet weigh in at WINRATE_DEFAULT, so winrates=None degrades to
    uniform (the Step-3 behavior). An empty opposite population makes that
    exploit mode fall back to live. Closes over plain lists/floats/dicts only,
    so it pickles across the Ray boundary.
    """
    red_pop = population_members(module_ids, RED_POP_RE)
    blue_pop = population_members(module_ids, BLUE_POP_RE)
    live_fraction = float(league_cfg.get("live_fraction", 0.5))
    exponent = float(league_cfg.get("pfsp_exponent", 2.0))
    floor = float(league_cfg.get("pfsp_uniform_floor", 0.1))
    wr = winrates or {}
    red_probs = pfsp_weights(
        {m: wr.get(m, WINRATE_DEFAULT) for m in red_pop}, exponent, floor)
    blue_probs = pfsp_weights(
        {m: wr.get(m, WINRATE_DEFAULT) for m in blue_pop}, exponent, floor)

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
            return weighted_pick(blue_pop, blue_probs, member)
        # blue_exploits
        if agent_id == "red_0":
            return weighted_pick(red_pop, red_probs, member)
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
        # Two-phase pruning: victim module id -> iteration it was marked. A
        # pending member is out of the mapping fn immediately but only
        # physically removed after the grace period, because in-flight episodes
        # keep their old agent->module mapping until they end (RLlib contract)
        # and would KeyError on a module yanked mid-episode.
        self._pending_removal: dict[str, int] = {}
        # Highest snapshot version ever issued per side this run; prevents a
        # pruned id from being recycled (which would inherit the stale metric
        # window of the dead policy).
        self._version_floor = {"red": 0, "blue": 0}

    def _league_cfg(self, algorithm) -> dict:
        return dict(algorithm.config.env_config.get("league", {}))

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._last_snapshot_iter = {"red": it, "blue": it}
        # Restore note: pending removals and the version floor reset here. A
        # victim checkpointed mid-grace rejoins the active population (one
        # member over cap until the next prune) and a pruned max-version id can
        # be reused across a restore — both bounded, same class as the
        # cooldown reset.
        self._pending_removal = {}
        module_ids = _algo_module_ids(algorithm)
        self._version_floor = {
            "red": next_version(module_ids, RED_POP_RE) - 1,
            "blue": next_version(module_ids, BLUE_POP_RE) - 1,
        }
        _refresh_mapping_fn(algorithm, make_league_mapping_fn(module_ids, self._cfg))

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        """Log the main's win vs the specific frozen opponent it faced, feeding
        the PFSP winrate estimates. Reuses ScoreMetricsCallback's score_acc
        (always installed alongside this callback by config_builder); episodes
        without it are skipped rather than guessed at."""
        if metrics_logger is None:
            return
        acc = episode.custom_data.get("score_acc")
        if acc is None:
            return
        outcome = matchup_outcome(
            episode.module_for("red_0"),
            episode.module_for("blue_0"),
            bool(acc["scored"]),
        )
        if outcome is not None:
            key, win = outcome
            metrics_logger.log_value(
                key, win, reduce="mean", window=WINRATE_WINDOW)

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._process_pending_removals(algorithm, it)
        active_ids = _algo_module_ids(algorithm) - set(self._pending_removal)
        winrates = collect_winrates(result, active_ids)
        for side, metric_name, main_id, regex in _SIDES:
            # Prefer the dedicated eval-battery metric (clean, length-unconfounded,
            # never NaN); fall back to the windowed in-training metric when the
            # battery is disabled (Step-4 behavior, fully backward-compatible).
            metric = read_metric(result, f"eval_{metric_name}")
            if metric is None:
                metric = read_metric(result, metric_name)
            if metric is None:
                continue
            pop = population_members(active_ids, regex)
            victim = None
            if len(pop) >= int(self._cfg["population_cap"]):
                victim = prune_candidate(
                    pop, winrates,
                    float(self._cfg.get("prune_winrate_threshold", 0.8)))
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg.get(
                    f"snapshot_threshold_{side}", self._cfg["snapshot_threshold"])),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(pop) - (1 if victim else 0),
                cap=int(self._cfg["population_cap"]),
            ):
                if victim is not None:
                    self._pending_removal[victim] = it
                    active_ids = active_ids - {victim}
                self._snapshot(algorithm, main_id, side, regex, active_ids)
                self._last_snapshot_iter[side] = it
                active_ids = _algo_module_ids(algorithm) - set(self._pending_removal)
        # PFSP weights move every iteration -> rebuild + reinstall the mapping
        # fn each time (the same cheap config overwrite the restore path uses).
        # This also supersedes the uniform fn add_module just installed when a
        # snapshot fired above, and drops pending victims from new matchups.
        self._refresh(algorithm, active_ids, winrates)
        report_league(result, active_ids, winrates, self._cfg)
        result["league"]["pending_removals"] = len(self._pending_removal)

    def _refresh(self, algorithm, module_ids, winrates) -> None:
        _refresh_mapping_fn(
            algorithm, make_league_mapping_fn(module_ids, self._cfg, winrates))

    def _process_pending_removals(self, algorithm, it: int) -> None:
        """Physically remove victims whose grace period has elapsed. By now no
        env runner routes new episodes to them (excluded from the mapping fn
        since mark time) and in-flight episodes from before the mark have ended
        (grace_iters x batch >= episode length — see conf/league/default.yaml)."""
        grace = int(self._cfg.get("prune_grace_iters", 2))
        due = [m for m, marked in self._pending_removal.items()
               if it - marked >= grace]
        for victim in due:
            algorithm.remove_module(
                module_id=victim,
                new_should_module_be_updated=["main_red", "main_blue"],
            )
            del self._pending_removal[victim]

    def _snapshot(self, algorithm, main_id, side, regex, module_ids) -> None:
        ver = max(next_version(module_ids, regex), self._version_floor[side] + 1)
        self._version_floor[side] = ver
        new_id = f"{side}_pop_v{ver}"
        # Uniform fn at add time is fine — the trailing _refresh in
        # on_train_result reinstalls the PFSP-weighted fn in the same call.
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
