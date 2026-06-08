"""Per-episode behavioral metrics for the RLlib skeleton.

The aggregation is split into pure functions (unit-tested in
tests/rllib/test_metrics.py) and a thin RLlibCallback that drives them from
the env-runner episode hooks and logs the result to the MetricsLogger so the
numbers reach Tune / W&B under env_runners/.

What this answers that episode return alone cannot: does Red actually score,
and how close does it get to the hoop?  Return is a confounded proxy; these
are the objective directly.
"""
from __future__ import annotations

import math

from ray.rllib.callbacks.callbacks import RLlibCallback

_LEARNER = "red_0"


# ── Pure aggregation ────────────────────────────────────────────────────────
def init_episode_acc() -> dict:
    """Fresh per-episode accumulator."""
    return {"scored": False, "min_dist": math.inf}


def update_episode_acc(acc: dict, red_info: dict) -> None:
    """Fold one step's red_0 info dict into the accumulator (in place)."""
    dist = red_info.get("dist_red_to_hoop")
    if dist is not None:
        acc["min_dist"] = min(acc["min_dist"], float(dist))
    if red_info.get("scored"):
        acc["scored"] = True


def episode_metrics(acc: dict) -> dict:
    """Per-episode metric values to log.  Omits min-dist if never observed."""
    out: dict[str, float] = {"red_score_rate": 1.0 if acc["scored"] else 0.0}
    if math.isfinite(acc["min_dist"]):
        out["red_min_dist_to_hoop"] = acc["min_dist"]
    return out


# ── RLlib wiring ────────────────────────────────────────────────────────────
def _red_info(episode) -> dict | None:
    """Extract red_0's latest info dict, tolerating both return shapes of
    MultiAgentEpisode.get_infos (the agent's dict directly, or {agent: dict})."""
    info = episode.get_infos(-1, _LEARNER)
    if isinstance(info, dict) and _LEARNER in info and isinstance(info[_LEARNER], dict):
        info = info[_LEARNER]
    return info if isinstance(info, dict) else None


class ScoreMetricsCallback(RLlibCallback):
    """Logs red_score_rate (mean over the metrics window → score fraction) and
    red_min_dist_to_hoop (mean of per-episode closest approach)."""

    def on_episode_start(self, *, episode, **kwargs) -> None:
        episode.custom_data["score_acc"] = init_episode_acc()

    def on_episode_step(self, *, episode, **kwargs) -> None:
        acc = episode.custom_data.get("score_acc")
        if acc is None:
            acc = episode.custom_data["score_acc"] = init_episode_acc()
        info = _red_info(episode)
        if info is not None:
            update_episode_acc(acc, info)

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        acc = episode.custom_data.get("score_acc") or init_episode_acc()
        if metrics_logger is None:
            return
        for key, value in episode_metrics(acc).items():
            metrics_logger.log_value(key, value, reduce="mean")
