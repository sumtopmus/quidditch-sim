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


class _StubEnv:
    """Scripted multi-agent env: each episode runs `script` steps, then ends
    with the given terminal infos. Mimics QuidditchMultiAgentEnv's step shape
    (obs/rew/term/trunc/infos dicts with __all__) without MuJoCo."""

    def __init__(self, episodes):
        # episodes: list of (n_steps, scored, take_down, terminal_cause)
        self._episodes = episodes
        self._idx = -1

    def reset(self, *, seed=None, options=None):
        self._idx += 1
        self._step = 0
        return {"red_0": [0.0], "blue_0": [0.0]}, {}

    def step(self, action_dict):
        n_steps, scored, take_down, cause = self._episodes[self._idx]
        self._step += 1
        done = self._step >= n_steps
        obs = {"red_0": [0.0], "blue_0": [0.0]}
        rew = {"red_0": 0.0, "blue_0": 0.0}
        term = {"red_0": done, "blue_0": done, "__all__": done}
        trunc = {"red_0": False, "blue_0": False, "__all__": False}
        infos = {"red_0": {}, "blue_0": {}}
        if done:
            # Stamp the terminal info so _classify_terminal returns `cause` and
            # `scored`/`take_down_fired` read correctly.
            if cause == "score":
                infos["red_0"]["scored"] = True
                infos["blue_0"]["scored"] = True
            elif cause == "drone_drone_crash":
                infos["red_0"]["drone_drone_crash"] = True
                infos["blue_0"]["drone_drone_crash"] = True
            elif cause == "red_oob":
                infos["red_0"]["red_oob"] = True
            infos["red_0"]["take_down_fired"] = take_down
            infos["blue_0"]["take_down_fired"] = take_down
        return obs, rew, term, trunc, infos


def test_rollout_battery_aggregates_outcomes():
    env = _StubEnv([
        (3, True,  False, "score"),
        (5, False, False, "timeout"),
        (2, False, True,  "drone_drone_crash"),
    ])
    # Action callables are never inspected by the stub; identity is fine.
    m = EB.rollout_battery(env, lambda o: o, lambda o: o, n_episodes=3, seed=0)
    assert m["eval_red_score_rate"] == 1 / 3
    assert m["eval_blue_prevention_rate"] == 2 / 3
    assert m["eval_takedown_rate"] == 1 / 3
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 1
    assert m["eval_terminal_drone_drone_crash"] == 1
    assert abs(m["eval_mean_ep_len"] - (3 + 5 + 2) / 3) < 1e-9
