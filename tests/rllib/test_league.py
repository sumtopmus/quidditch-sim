"""Step-3 snapshot-population league: pure helpers + mapping fn + callback."""
from __future__ import annotations

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


_LEAGUE_CFG = {"snapshot_threshold": 0.7, "min_iters_between_snapshots": 20,
               "population_cap": 5, "live_fraction": 0.5}


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


def test_callback_reconstructs_membership_and_refreshes_on_init():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "red_pop_v1", "red_pop_v2"},
                     iteration=100, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    assert algo.env_runner_group.refreshed == 1       # mapping fn reinstalled
    # next red snapshot would be v3 (derived from restored module ids)
    assert L.next_version(algo.get_module().keys(), L.RED_POP_RE) == 3
