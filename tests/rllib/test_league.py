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
