"""Dedicated, deterministic eval battery for the RLlib league (Step 5b).

Three layers, mirroring rllib/metrics.py:
  1. Pure aggregation (init/fold/finalize) — unit-tested, no RLlib deps.
  2. rollout_battery(env, act_red, act_blue, ...) — deterministic head-to-head
     rollouts on a QuidditchMultiAgentEnv; takes two obs->action callables.
  3. EvalBatteryCallback — runs the battery on a cadence inside on_train_result
     and writes flat eval_* keys into the train result.

Why a dedicated battery: Step 4 gates snapshots on RLlib's WINDOWED
red_score_rate / blue_prevention_rate, which emit NaN when a window drains and
are confounded by the PFSP matchup mix. The battery plays a fixed number of
deterministic main_red-vs-main_blue episodes and reports clean, length-
unconfounded metrics. "Honest prevention" = fraction of episodes Red did NOT
score — a binary per-episode question, independent of episode length.
"""
from __future__ import annotations

from core.eval_core import TERMINAL_BUCKETS, _classify_terminal


# ── Pure aggregation ────────────────────────────────────────────────────────
def init_battery_acc() -> dict:
    """Fresh battery accumulator."""
    return {
        "n": 0,
        "scored": 0,
        "take_down": 0,
        "len_sum": 0,
        "buckets": {b: 0 for b in TERMINAL_BUCKETS},
    }


def fold_episode(
    acc: dict, *, scored: bool, take_down: bool, bucket: str, length: int
) -> None:
    """Fold one finished episode's outcome into the accumulator (in place)."""
    acc["n"] += 1
    acc["scored"] += 1 if scored else 0
    acc["take_down"] += 1 if take_down else 0
    acc["len_sum"] += int(length)
    if bucket in acc["buckets"]:
        acc["buckets"][bucket] += 1


def battery_metrics(acc: dict) -> dict:
    """Flat eval_* metrics. Red score-rate, honest Blue prevention (1 - score-
    rate), takedown-rate, per-bucket terminal histogram, mean episode length."""
    n = acc["n"]
    score_rate = acc["scored"] / n if n else 0.0
    # Honest prevention = fraction of episodes Red did NOT score. Computed
    # directly from the complement count (n - scored)/n rather than 1 - score_rate
    # so it is exact (no float rounding) and srate + prevention == 1.0 holds.
    out: dict[str, float] = {
        "eval_red_score_rate": score_rate,
        "eval_blue_prevention_rate": ((n - acc["scored"]) / n) if n else 0.0,
        "eval_takedown_rate": (acc["take_down"] / n) if n else 0.0,
        "eval_mean_ep_len": (acc["len_sum"] / n) if n else 0.0,
        "eval_episodes": float(n),
    }
    for bucket, count in acc["buckets"].items():
        out[f"eval_terminal_{bucket}"] = count
    return out


import numpy as np


# ── Deterministic head-to-head rollouts ─────────────────────────────────────
def rollout_battery(env, act_red, act_blue, *, n_episodes: int, seed: int) -> dict:
    """Play `n_episodes` deterministic episodes; aggregate into eval_* metrics.

    `env` is a QuidditchMultiAgentEnv (RLlib MultiAgentEnv): step returns
    (obs, rew, term, trunc, infos) dicts with an "__all__" whole-episode flag.
    `act_red` / `act_blue` map an agent's obs array -> action array. A per-
    episode seed derived from `seed` keeps the battery reproducible run-to-run.
    """
    acc = init_battery_acc()
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        length = 0
        red_info: dict = {}
        blue_info: dict = {}
        while True:
            actions = {}
            if "red_0" in obs:
                actions["red_0"] = act_red(obs["red_0"])
            if "blue_0" in obs:
                actions["blue_0"] = act_blue(obs["blue_0"])
            obs, _, term, trunc, infos = env.step(actions)
            length += 1
            red_info = infos.get("red_0", red_info) or red_info
            blue_info = infos.get("blue_0", blue_info) or blue_info
            if term.get("__all__") or trunc.get("__all__"):
                break
        scored = bool(red_info.get("scored") or blue_info.get("scored"))
        take_down = bool(
            red_info.get("take_down_fired") or blue_info.get("take_down_fired")
        )
        bucket = _classify_terminal(red_info, blue_info)
        fold_episode(acc, scored=scored, take_down=take_down,
                     bucket=bucket, length=length)
    return battery_metrics(acc)
