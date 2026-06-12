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
