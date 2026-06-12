"""Step-5a league curriculum: dense->sparse reward anneal + difficulty levers.

A pure scheduled_value interpolator + a thin CurriculumCallback that pushes the
scheduled reward dense-scale and difficulty-lever values onto every env-runner's
live QuidditchTeamEnv each on_train_result. The env reads red_action_scale /
red_start_r_max live (in step / _sample_red_start) and dense_scale live (in
RewardStack.compute_step), so a runtime push takes effect on the next step.
"""
from __future__ import annotations

from typing import Optional

from ray.rllib.callbacks.callbacks import RLlibCallback


def scheduled_value(schedule, t) -> Optional[float]:
    """Piecewise-linear interpolation of a [[t0, v0], [t1, v1], ...] schedule.

    Clamps to the first value below t0 and the last value above the final knot.
    Returns None when `schedule` is falsy (no override configured), so callers
    can distinguish "leave the field alone" from "set it to 0".
    """
    if not schedule:
        return None
    knots = [(float(t_i), float(v_i)) for t_i, v_i in schedule]
    knots.sort()
    t = float(t)
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    for (t0, v0), (t1, v1) in zip(knots, knots[1:]):
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return v0 + frac * (v1 - v0)
    return knots[-1][1]


def _read_lifetime_steps(result: dict) -> float:
    """Lifetime env-steps sampled, the schedule's time axis. Recursive so it
    works wherever RLlib nests the counter."""
    key = "num_env_steps_sampled_lifetime"
    if key in result and isinstance(result[key], (int, float)):
        return float(result[key])
    for v in result.values():
        if isinstance(v, dict):
            found = _read_lifetime_steps(v)
            if found is not None:
                return found
    return 0.0


def _inner_team_env(runner):
    """Reach the QuidditchTeamEnv from an env runner. RLlib's env runner holds
    the QuidditchMultiAgentEnv at runner.env (single env per runner); ._inner is
    the wrapped QuidditchTeamEnv. Returns None if the shape is unexpected."""
    env = getattr(runner, "env", None)
    env = getattr(env, "unwrapped", env)
    return getattr(env, "_inner", None)


def _apply_to_team_envs(algorithm, fn) -> None:
    def _set(runner, _fn=fn):
        inner = _inner_team_env(runner)
        if inner is not None:
            _fn(inner)
    algorithm.env_runner_group.foreach_env_runner(_set, local_env_runner=True)


class CurriculumCallback(RLlibCallback):
    """Pushes scheduled reward dense-scale + difficulty-lever values onto every
    env-runner's live QuidditchTeamEnv each iteration. No-op for any lever
    without a schedule (its static value holds)."""

    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        cur = dict(algorithm.config.env_config.get("curriculum", {}))
        if not cur:
            return
        t = _read_lifetime_steps(result)
        dense = scheduled_value(cur.get("dense_scale_schedule"), t)
        ras = scheduled_value(cur.get("red_action_scale_schedule"), t)
        rsm = scheduled_value(cur.get("red_start_r_max_schedule"), t)
        if dense is None and ras is None and rsm is None:
            return

        def _push(inner, _d=dense, _a=ras, _r=rsm):
            if _d is not None:
                inner._reward_stack.set_dense_scale(_d)
            if _a is not None:
                inner.cfg.red_action_scale = _a
            if _r is not None:
                inner.cfg.red_start_r_max = _r

        _apply_to_team_envs(algorithm, _push)
        # Surface the live values in the result for W&B visibility.
        cd = result.setdefault("curriculum", {})
        if dense is not None:
            cd["dense_scale"] = dense
        if ras is not None:
            cd["red_action_scale"] = ras
        if rsm is not None:
            cd["red_start_r_max"] = rsm
