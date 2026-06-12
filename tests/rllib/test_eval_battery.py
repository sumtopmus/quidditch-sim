"""Step-5b eval battery: pure aggregation + rollout + callback wiring."""
from __future__ import annotations

import rllib.eval_battery as EB


def test_battery_metrics_honest_prevention_is_length_independent():
    acc = EB.init_battery_acc()
    # 4 episodes: 1 score, 3 prevented. Episode lengths vary wildly; prevention
    # must depend ONLY on the binary scored flag, not on length.
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",    length=900)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout",  length=10)
    EB.fold_episode(acc, scored=False, take_down=True,  bucket="drone_drone_crash", length=50)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="red_oob",  length=5)
    m = EB.battery_metrics(acc)
    assert m["eval_red_score_rate"] == 0.25
    assert m["eval_blue_prevention_rate"] == 0.75
    assert m["eval_takedown_rate"] == 0.25
    assert abs(m["eval_mean_ep_len"] - (900 + 10 + 50 + 5) / 4) < 1e-9


def test_battery_metrics_terminal_histogram():
    acc = EB.init_battery_acc()
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",   length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    m = EB.battery_metrics(acc)
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 2
    # every bucket is represented (zeros included) so W&B panels are stable
    assert m["eval_terminal_red_oob"] == 0


def test_battery_metrics_empty_is_safe():
    m = EB.battery_metrics(EB.init_battery_acc())
    assert m["eval_red_score_rate"] == 0.0
    assert m["eval_blue_prevention_rate"] == 0.0
    assert m["eval_mean_ep_len"] == 0.0
