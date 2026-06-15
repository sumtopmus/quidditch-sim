"""Step-3 snapshot-population league: pure helpers + mapping fn + callback."""
from __future__ import annotations

import math

import rllib.league as L


def test_population_members_sorted_by_version():
    ids = {"main_red", "main_blue", "red_pop_v2", "red_pop_v1", "blue_pop_v1"}
    assert L.population_members(ids, L.RED_POP_RE) == ["red_pop_v1", "red_pop_v2"]
    assert L.population_members(ids, L.BLUE_POP_RE) == ["blue_pop_v1"]


def test_next_version():
    assert L.next_version({"main_red", "main_blue"}, L.RED_POP_RE) == 1
    assert L.next_version({"red_pop_v1", "red_pop_v3"}, L.RED_POP_RE) == 4
    assert L.next_version({"red_pop_v1"}, L.BLUE_POP_RE) == 1


def test_read_metric_finds_nested_key():
    result = {"env_runners": {"red_score_rate": 0.8, "other": 1}, "x": 2}
    assert L.read_metric(result, "red_score_rate") == 0.8
    assert L.read_metric(result, "blue_prevention_rate") is None


def test_should_snapshot_threshold_cooldown_and_cap():
    base = dict(threshold=0.7, cooldown=20, pop_size=0, cap=5)
    # at/above threshold + cooldown elapsed + room -> True
    assert L.should_snapshot(metric=0.7, iters_since_last=20, **base)
    assert L.should_snapshot(metric=0.9, iters_since_last=999, **base)
    # below threshold -> False
    assert not L.should_snapshot(metric=0.69, iters_since_last=20, **base)
    # within cooldown -> False
    assert not L.should_snapshot(metric=0.9, iters_since_last=19, **base)
    # population full -> False
    assert not L.should_snapshot(
        metric=0.9, iters_since_last=20, threshold=0.7, cooldown=20, pop_size=5, cap=5)


import types


def _episode(eid):
    return types.SimpleNamespace(id_=eid)


def test_pfsp_weights_prioritizes_hard_opponents():
    probs = L.pfsp_weights({"a": 0.9, "b": 0.1}, exponent=2.0, floor=0.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # raw weights (1-0.9)^2 = 0.01 vs (1-0.1)^2 = 0.81 -> b gets ~98.8%
    assert probs["b"] > 0.95 > probs["a"] > 0.0


def test_pfsp_weights_uniform_floor_is_anti_forgetting():
    probs = L.pfsp_weights({"a": 1.0, "b": 0.0}, exponent=2.0, floor=0.2)
    # 'a' is fully dominated; the floor still guarantees it floor/N mass
    assert abs(probs["a"] - 0.1) < 1e-9
    assert abs(probs["b"] - 0.9) < 1e-9


def test_pfsp_weights_all_dominated_falls_back_to_uniform():
    assert L.pfsp_weights({"a": 1.0, "b": 1.0}, exponent=2.0, floor=0.0) == {
        "a": 0.5, "b": 0.5}


def test_pfsp_weights_empty():
    assert L.pfsp_weights({}, exponent=2.0, floor=0.1) == {}


def test_weighted_pick_respects_cumulative_probs():
    items, probs = ["a", "b"], {"a": 0.25, "b": 0.75}
    assert L.weighted_pick(items, probs, 0.0) == "a"
    assert L.weighted_pick(items, probs, 0.24) == "a"
    assert L.weighted_pick(items, probs, 0.25) == "b"
    assert L.weighted_pick(items, probs, 0.999) == "b"
    # float-accumulation guard: a roll at/above the total still returns an item
    assert L.weighted_pick(items, probs, 1.0) == "b"


def test_episode_roll_deterministic_salted_and_bounded():
    ep = _episode("abc")
    r = L._episode_roll(ep, "mode")
    assert r == L._episode_roll(ep, "mode")          # deterministic
    assert 0.0 <= r < 1.0
    # different salts and different episode ids give different draws
    assert r != L._episode_roll(ep, "member")
    assert r != L._episode_roll(_episode("xyz"), "mode")


def test_mapping_is_live_when_populations_empty():
    fn = L.make_league_mapping_fn({"main_red", "main_blue"}, {"live_fraction": 0.5})
    for eid in range(50):
        ep = _episode(eid)
        assert fn("red_0", ep) == "main_red"
        assert fn("blue_0", ep) == "main_blue"


def test_mapping_both_agents_coherent_per_episode():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})  # always exploit
    for eid in range(50):
        ep = _episode(eid)
        red, blue = fn("red_0", ep), fn("blue_0", ep)
        # Exactly one side is a frozen pop member; the other is its live main.
        red_exploits = red == "main_red" and blue == "blue_pop_v1"
        blue_exploits = red == "red_pop_v1" and blue == "main_blue"
        assert red_exploits or blue_exploits


def test_mapping_respects_live_fraction():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.5})
    n = 4000
    live = sum(
        1 for eid in range(n)
        if fn("red_0", _episode(eid)) == "main_red"
        and fn("blue_0", _episode(eid)) == "main_blue"
    )
    assert 0.40 < live / n < 0.60   # ~0.5 within tolerance


def test_mapping_empty_pop_falls_back_to_live():
    # Only red_pop exists -> "red exploits" (needs blue_pop) can't happen;
    # blue can still exploit red_pop. Never routes to a nonexistent member.
    ids = {"main_red", "main_blue", "red_pop_v1"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})
    for eid in range(50):
        ep = _episode(eid)
        red, blue = fn("red_0", ep), fn("blue_0", ep)
        assert blue == "main_blue"                 # blue never frozen (no blue_pop)
        assert red in ("main_red", "red_pop_v1")   # live or blue-exploits


def test_mapping_pfsp_concentrates_on_hard_opponents():
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    cfg = {"live_fraction": 0.0, "pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.0}
    # main_red dominates v1 (wr 0.9) and struggles vs v2 (wr 0.1)
    fn = L.make_league_mapping_fn(
        ids, cfg, winrates={"blue_pop_v1": 0.9, "blue_pop_v2": 0.1})
    picks = [fn("blue_0", _episode(eid)) for eid in range(4000)]
    frozen = [p for p in picks if p != "main_blue"]
    assert frozen, "expected red-exploits episodes"
    assert frozen.count("blue_pop_v2") / len(frozen) > 0.9   # ~0.988 expected


def test_mapping_without_winrates_is_uniform():
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})
    picks = [fn("blue_0", _episode(eid)) for eid in range(4000)]
    frozen = [p for p in picks if p != "main_blue"]
    assert 0.4 < frozen.count("blue_pop_v1") / len(frozen) < 0.6


def test_matchup_outcome_main_vs_frozen():
    # red main vs frozen blue: red's win = it scored
    assert L.matchup_outcome("main_red", "blue_pop_v1", scored=True) == (
        "league_wr_vs_blue_pop_v1", 1.0)
    assert L.matchup_outcome("main_red", "blue_pop_v1", scored=False) == (
        "league_wr_vs_blue_pop_v1", 0.0)
    # blue main vs frozen red: blue's win = it prevented the score
    assert L.matchup_outcome("red_pop_v3", "main_blue", scored=False) == (
        "league_wr_vs_red_pop_v3", 1.0)
    assert L.matchup_outcome("red_pop_v3", "main_blue", scored=True) == (
        "league_wr_vs_red_pop_v3", 0.0)


def test_matchup_outcome_live_episode_is_none():
    assert L.matchup_outcome("main_red", "main_blue", scored=True) is None


class _FakeMetricsLogger:
    def __init__(self):
        self.logged = []

    def log_value(self, key, value, **kw):
        self.logged.append((key, value))


def _ended_episode(red_mod, blue_mod, scored):
    mapping = {"red_0": red_mod, "blue_0": blue_mod}
    return types.SimpleNamespace(
        id_="ep1",
        custom_data={"score_acc": {"scored": scored, "min_dist": 1.0}},
        module_for=lambda aid, m=mapping: m[aid],
    )


def test_on_episode_end_logs_main_vs_frozen_winrate():
    cb = L.LeagueCallback()
    logger = _FakeMetricsLogger()
    cb.on_episode_end(
        episode=_ended_episode("main_red", "blue_pop_v1", scored=True),
        metrics_logger=logger)
    assert ("league_wr_vs_blue_pop_v1", 1.0) in logger.logged


def test_on_episode_end_skips_live_and_unaccumulated_episodes():
    cb = L.LeagueCallback()
    logger = _FakeMetricsLogger()
    cb.on_episode_end(
        episode=_ended_episode("main_red", "main_blue", scored=True),
        metrics_logger=logger)
    ep = _ended_episode("main_red", "blue_pop_v1", scored=True)
    ep.custom_data = {}   # ScoreMetricsCallback absent -> no acc -> skip
    cb.on_episode_end(episode=ep, metrics_logger=logger)
    assert logger.logged == []


class _FakeModule:
    def __init__(self, ids):
        self._ids = set(ids)
    def keys(self):
        return set(self._ids)


class _FakeLearnerGroup:
    def foreach_learner(self, fn):
        return []


class _FakeEnvRunnerGroup:
    def __init__(self):
        self.synced = []
        self.refreshed = 0
    def sync_weights(self, **kw):
        self.synced.append(kw.get("policies"))
    def foreach_env_runner(self, fn, **kw):
        self.refreshed += 1
        return []


class _FakeAlgo:
    def __init__(self, ids, iteration, league_cfg):
        self._module = _FakeModule(ids)
        self.iteration = iteration
        self.config = types.SimpleNamespace(env_config={"league": league_cfg})
        self.learner_group = _FakeLearnerGroup()
        self.env_runner_group = _FakeEnvRunnerGroup()
        # The MultiRLModule (population store) lives on the local env runner.
        self.env_runner = types.SimpleNamespace(module=self._module)
        self.added = []
        self.set_states = []
        self.removed = []
    def get_module(self, module_id=None):
        if module_id is None:
            return self._module  # the MultiRLModule (has .keys())
        # A single sub-module: only get_state() is exercised by _snapshot.
        return types.SimpleNamespace(get_state=lambda mid=module_id: {"weights": mid})
    def add_module(self, *, module_id, module_spec, new_should_module_be_updated,
                   new_agent_to_module_mapping_fn, **kw):
        self.added.append(module_id)
        self._module._ids.add(module_id)  # reflect the new member
    def set_state(self, state):
        self.set_states.append(state)
    def remove_module(self, module_id, **kw):
        self.removed.append(module_id)
        self._module._ids.discard(module_id)


_LEAGUE_CFG = {"snapshot_threshold": 0.7, "min_iters_between_snapshots": 20,
               "population_cap": 5, "live_fraction": 0.5,
               "pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.1,
               "prune_winrate_threshold": 0.8, "prune_grace_iters": 2}


def test_collect_winrates_reads_only_population_keys():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    result = {"env_runners": {"league_wr_vs_red_pop_v1": 0.8, "noise": 1.0}}
    assert L.collect_winrates(result, ids) == {"red_pop_v1": 0.8}


def test_report_league_writes_winrates_and_pfsp_probs():
    result = {}
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    L.report_league(result, ids, {"blue_pop_v1": 0.9},
                    {"pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.0})
    league = result["league"]
    assert league["wr_vs_blue_pop_v1"] == 0.9
    assert league["wr_vs_blue_pop_v2"] == L.WINRATE_DEFAULT   # unseen
    total = league["pfsp_p_blue_pop_v1"] + league["pfsp_p_blue_pop_v2"]
    assert abs(total - 1.0) < 1e-9
    assert league["pfsp_p_blue_pop_v2"] > league["pfsp_p_blue_pop_v1"]


def test_on_train_result_refreshes_mapping_fn_every_iteration():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=5, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    before = algo.env_runner_group.refreshed
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.0,
                                               "blue_prevention_rate": 0.0}})
    assert algo.env_runner_group.refreshed > before   # PFSP weights reinstalled


def test_prune_candidate_picks_most_dominated():
    pop = ["red_pop_v1", "red_pop_v2", "red_pop_v3"]
    wr = {"red_pop_v1": 0.95, "red_pop_v2": 0.85, "red_pop_v3": 0.4}
    assert L.prune_candidate(pop, wr, threshold=0.8) == "red_pop_v1"


def test_prune_candidate_none_when_population_still_challenging():
    pop = ["red_pop_v1", "red_pop_v2"]
    wr = {"red_pop_v1": 0.6}   # v2 unseen -> WINRATE_DEFAULT (0.5)
    assert L.prune_candidate(pop, wr, threshold=0.8) is None


def test_callback_snapshots_red_when_threshold_cleared(monkeypatch):
    # Cloning a real RLModule spec needs a live module; the fake add_module
    # ignores the spec, so stub the spec builder with a sentinel.
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)            # baselines cooldown at iter 25
    algo.iteration = 50                              # 25 iters later (> cooldown)
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.8,
                                               "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added
    assert "blue_pop_v1" not in algo.added           # blue below threshold
    # Snapshot weights are pushed to the learner + all env runners via set_state.
    pushed = algo.set_states[-1]["learner_group"]["learner"]["rl_module"]
    assert "red_pop_v1" in pushed


def test_callback_respects_cooldown():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=10, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 19                               # < cooldown (20)
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.95}})
    assert algo.added == []


def _prune_cfg(**over):
    return {**_LEAGUE_CFG, "population_cap": 1,
            "min_iters_between_snapshots": 0, **over}


def test_callback_prunes_most_dominated_at_cap(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9,        # blue snapshot due, blue_pop at cap
        "red_score_rate": 0.0,              # red side quiet
        "league_wr_vs_blue_pop_v1": 0.95,   # v1 dominated -> prunable
    }})
    assert "blue_pop_v2" in algo.added                  # snapshot proceeded
    assert "blue_pop_v1" in cb._pending_removal         # victim marked...
    assert algo.removed == []                           # ...not yet removed


def test_pending_removal_executes_after_grace(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg(prune_grace_iters=2))
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.95}})
    quiet = {"env_runners": {"blue_prevention_rate": 0.0, "red_score_rate": 0.0}}
    algo.iteration = 2
    cb.on_train_result(algorithm=algo, result=dict(quiet))   # grace not elapsed
    assert algo.removed == []
    algo.iteration = 3
    cb.on_train_result(algorithm=algo, result=dict(quiet))   # 3 - 1 >= 2 -> due
    assert algo.removed == ["blue_pop_v1"]
    assert "blue_pop_v1" not in cb._pending_removal


def test_no_snapshot_at_cap_when_no_member_is_dominated(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.4}})   # still challenging -> keep
    assert algo.added == []                  # Step-3 stop-at-cap preserved
    assert cb._pending_removal == {}


def test_snapshot_version_never_reused_after_prune(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.95}})
    # the active pop at snapshot time was empty (v1 pending), yet the version
    # floor prevents recycling v1 for a brand-new policy
    assert "blue_pop_v2" in algo.added
    assert "blue_pop_v1" not in algo.added


def test_callback_reconstructs_membership_and_refreshes_on_init():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "red_pop_v1", "red_pop_v2"},
                     iteration=100, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    assert algo.env_runner_group.refreshed == 1       # mapping fn reinstalled
    # next red snapshot would be v3 (derived from restored module ids)
    assert L.next_version(algo.get_module().keys(), L.RED_POP_RE) == 3


def test_read_metric_ignores_non_finite_values():
    # RLlib's windowed metrics emit NaN when a matchup's window drains (no
    # recent episodes) — those must read as "no data", not as a winrate.
    nan = float("nan")
    assert L.read_metric({"env_runners": {"league_wr_vs_blue_pop_v1": nan}},
                         "league_wr_vs_blue_pop_v1") is None
    assert L.read_metric({"x": float("inf")}, "x") is None
    assert L.read_metric({"a": {"x": nan}, "x": 0.3}, "x") == 0.3


def test_collect_winrates_omits_nan_members():
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    result = {"env_runners": {"league_wr_vs_blue_pop_v1": float("nan"),
                              "league_wr_vs_blue_pop_v2": 0.4}}
    assert L.collect_winrates(result, ids) == {"blue_pop_v2": 0.4}


def test_pfsp_weights_treats_non_finite_winrate_as_default():
    # Defense in depth: a NaN that slips through must not poison every
    # probability (NaN cum-probs make weighted_pick always return the LAST
    # member, silently collapsing the league to a single opponent).
    probs = L.pfsp_weights({"a": float("nan"), "b": 0.5}, exponent=2.0, floor=0.0)
    assert all(math.isfinite(p) for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["a"] == probs["b"]   # NaN -> WINRATE_DEFAULT (= b's 0.5)


def test_snapshot_prefers_eval_metric_over_windowed(monkeypatch):
    """When the eval battery has written eval_red_score_rate, the gate uses it
    (not the confounded windowed red_score_rate)."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    # Windowed red_score_rate is high (0.9) but the clean eval says 0.2 -> below
    # the 0.7 threshold -> NO red snapshot. The eval metric wins.
    cb.on_train_result(algorithm=algo, result={
        "env_runners": {"red_score_rate": 0.9, "blue_prevention_rate": 0.1},
        "eval": {"eval_red_score_rate": 0.2, "eval_blue_prevention_rate": 0.1},
    })
    assert "red_pop_v1" not in algo.added


def test_snapshot_falls_back_to_windowed_without_eval(monkeypatch):
    """With no eval block (battery disabled), Step-4 windowed gating is intact."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={
        "env_runners": {"red_score_rate": 0.8, "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added


def _asym_cfg(**over):
    return {**_LEAGUE_CFG, "snapshot_threshold_red": 0.4,
            "snapshot_threshold_blue": 0.7, **over}


def test_per_side_snapshot_threshold(monkeypatch):
    """Red uses snapshot_threshold_red (0.4); Blue uses snapshot_threshold_blue
    (0.7). A metric of 0.5 snapshots Red but not Blue."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_asym_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "red_score_rate": 0.5, "blue_prevention_rate": 0.5}})
    assert "red_pop_v1" in algo.added        # 0.5 >= 0.4 (red threshold)
    assert "blue_pop_v1" not in algo.added   # 0.5 < 0.7 (blue threshold)


def test_per_side_threshold_falls_back_to_shared(monkeypatch):
    """Without per-side keys, both sides use the shared snapshot_threshold."""
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue"}, iteration=25, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 50
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "red_score_rate": 0.75, "blue_prevention_rate": 0.1}})
    assert "red_pop_v1" in algo.added        # 0.75 >= shared 0.7
