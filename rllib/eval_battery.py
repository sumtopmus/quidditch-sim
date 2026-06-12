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
    out: dict[str, float] = {
        "eval_red_score_rate": score_rate,
        "eval_blue_prevention_rate": (1.0 - score_rate) if n else 0.0,
        "eval_takedown_rate": (acc["take_down"] / n) if n else 0.0,
        "eval_mean_ep_len": (acc["len_sum"] / n) if n else 0.0,
        "eval_episodes": float(n),
    }
    for bucket, count in acc["buckets"].items():
        out[f"eval_terminal_{bucket}"] = count
    return out
