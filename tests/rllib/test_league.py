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
