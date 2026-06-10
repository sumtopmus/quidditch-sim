"""Pure per-episode metric aggregation used by ScoreMetricsCallback."""
from __future__ import annotations

import math

from rllib.metrics import episode_metrics, init_episode_acc, update_episode_acc
from rllib.metrics import (
    blue_episode_metrics, init_blue_acc, update_blue_acc,
)


def test_init_acc_starts_empty():
    acc = init_episode_acc()
    assert acc["scored"] is False
    assert math.isinf(acc["min_dist"])


def test_update_tracks_running_min_distance():
    acc = init_episode_acc()
    update_episode_acc(acc, {"dist_red_to_hoop": 1.5, "scored": False})
    update_episode_acc(acc, {"dist_red_to_hoop": 0.4, "scored": False})
    update_episode_acc(acc, {"dist_red_to_hoop": 0.9, "scored": False})
    assert acc["min_dist"] == 0.4


def test_update_latches_scored():
    acc = init_episode_acc()
    update_episode_acc(acc, {"dist_red_to_hoop": 1.0, "scored": True})
    update_episode_acc(acc, {"dist_red_to_hoop": 1.0, "scored": False})
    assert acc["scored"] is True


def test_update_ignores_missing_distance():
    acc = init_episode_acc()
    update_episode_acc(acc, {"scored": False})        # no dist key
    assert math.isinf(acc["min_dist"])


def test_episode_metrics_scored_and_min_dist():
    acc = init_episode_acc()
    update_episode_acc(acc, {"dist_red_to_hoop": 0.3, "scored": True})
    m = episode_metrics(acc)
    assert m["red_score_rate"] == 1.0
    assert m["red_min_dist_to_hoop"] == 0.3


def test_episode_metrics_not_scored_is_zero_rate():
    acc = init_episode_acc()
    update_episode_acc(acc, {"dist_red_to_hoop": 1.2, "scored": False})
    m = episode_metrics(acc)
    assert m["red_score_rate"] == 0.0
    assert m["red_min_dist_to_hoop"] == 1.2


def test_episode_metrics_omits_min_dist_when_never_seen():
    """No distance ever observed → don't log a spurious inf."""
    m = episode_metrics(init_episode_acc())
    assert m["red_score_rate"] == 0.0
    assert "red_min_dist_to_hoop" not in m


def test_blue_acc_tracks_min_dist_to_red_and_prevention():
    acc = init_blue_acc()
    update_blue_acc(acc, {"dist_b2r": 1.2, "scored": False})
    update_blue_acc(acc, {"dist_b2r": 0.3, "scored": False})
    m = blue_episode_metrics(acc)
    assert m["blue_min_dist_to_red"] == 0.3
    assert m["blue_prevention_rate"] == 1.0   # Red did not score this episode


def test_blue_prevention_rate_zero_when_red_scores():
    acc = init_blue_acc()
    update_blue_acc(acc, {"dist_b2r": 0.8, "scored": True})
    m = blue_episode_metrics(acc)
    assert m["blue_prevention_rate"] == 0.0
