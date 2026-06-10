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
_BLUE = "blue_0"


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


# ── Blue (defender) aggregation ─────────────────────────────────────────────
# Self-play (Step 2) wants Blue's defense legible too: how often it prevents a
# score, and how close it gets to Red (its takedown opportunity).
def init_blue_acc() -> dict:
    return {"scored": False, "min_dist_to_red": math.inf}


def update_blue_acc(acc: dict, blue_info: dict) -> None:
    dist = blue_info.get("dist_b2r")
    if dist is not None:
        acc["min_dist_to_red"] = min(acc["min_dist_to_red"], float(dist))
    if blue_info.get("scored"):
        acc["scored"] = True


def blue_episode_metrics(acc: dict) -> dict:
    out: dict[str, float] = {"blue_prevention_rate": 0.0 if acc["scored"] else 1.0}
    if math.isfinite(acc["min_dist_to_red"]):
        out["blue_min_dist_to_red"] = acc["min_dist_to_red"]
    return out


# ── RLlib wiring ────────────────────────────────────────────────────────────
def _red_info(episode) -> dict | None:
    """Extract red_0's latest info dict, tolerating both return shapes of
    MultiAgentEpisode.get_infos (the agent's dict directly, or {agent: dict})."""
    info = episode.get_infos(-1, _LEARNER)
    if isinstance(info, dict) and _LEARNER in info and isinstance(info[_LEARNER], dict):
        info = info[_LEARNER]
    return info if isinstance(info, dict) else None


def _blue_info(episode) -> dict | None:
    info = episode.get_infos(-1, _BLUE)
    if isinstance(info, dict) and _BLUE in info and isinstance(info[_BLUE], dict):
        info = info[_BLUE]
    return info if isinstance(info, dict) else None


class ScoreMetricsCallback(RLlibCallback):
    """Logs Red offense (red_score_rate, red_min_dist_to_hoop) and Blue defense
    (blue_prevention_rate, blue_min_dist_to_red), each a mean over the metrics
    window."""

    def on_episode_start(self, *, episode, **kwargs) -> None:
        episode.custom_data["score_acc"] = init_episode_acc()
        episode.custom_data["blue_acc"] = init_blue_acc()

    def on_episode_step(self, *, episode, **kwargs) -> None:
        red_acc = episode.custom_data.setdefault("score_acc", init_episode_acc())
        blue_acc = episode.custom_data.setdefault("blue_acc", init_blue_acc())
        red = _red_info(episode)
        if red is not None:
            update_episode_acc(red_acc, red)
        blue = _blue_info(episode)
        if blue is not None:
            update_blue_acc(blue_acc, blue)

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        if metrics_logger is None:
            return
        red_acc = episode.custom_data.get("score_acc") or init_episode_acc()
        blue_acc = episode.custom_data.get("blue_acc") or init_blue_acc()
        metrics = {**episode_metrics(red_acc), **blue_episode_metrics(blue_acc)}
        for key, value in metrics.items():
            metrics_logger.log_value(key, value, reduce="mean")
