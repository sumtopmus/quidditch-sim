"""Step-5a curriculum: reward-anneal scaling, scheduler, and CurriculumCallback."""
from __future__ import annotations

from envs.quidditch.rewards.stack import RewardStack, StepState
from envs.quidditch.rewards.terms import HoopApproachShaping, ScoreEvent


def _state(scored: bool, prev: float, cur: float) -> StepState:
    return StepState(agent_ids=("red_0", "blue_0"), scored=scored,
                     dist_red_to_hoop=cur, dist_red_to_hoop_prev=prev)


def test_dense_scale_scales_shaping_not_outcomes():
    stack = RewardStack(terms=[
        HoopApproachShaping(scale=2.0, agent="red_0"),   # dense -> scaled
        ScoreEvent(magnitude=10.0, scorer="red_0", zero_sum_opponent="blue_0"),
    ])
    # progress 1.0 -> shaping = 2.0; score -> +10 red / -10 blue.
    full = stack.compute_step(_state(scored=True, prev=2.0, cur=1.0))
    assert abs(full["red_0"] - (2.0 + 10.0)) < 1e-9

    stack.set_dense_scale(0.5)
    half = stack.compute_step(_state(scored=True, prev=2.0, cur=1.0))
    # shaping halved (1.0); score untouched (+10) -> 11.0 red
    assert abs(half["red_0"] - (1.0 + 10.0)) < 1e-9
    # Blue's zero-sum score mirror (-10) is a sparse outcome, NOT scaled.
    assert abs(half["blue_0"] - (-10.0)) < 1e-9


def test_dense_scale_zero_removes_all_shaping():
    stack = RewardStack(terms=[HoopApproachShaping(scale=2.0, agent="red_0")])
    stack.set_dense_scale(0.0)
    out = stack.compute_step(_state(scored=False, prev=2.0, cur=1.0))
    assert out["red_0"] == 0.0


def test_default_dense_scale_is_identity():
    stack = RewardStack(terms=[HoopApproachShaping(scale=2.0, agent="red_0")])
    assert stack.dense_scale == 1.0
    out = stack.compute_step(_state(scored=False, prev=2.0, cur=1.0))
    assert abs(out["red_0"] - 2.0) < 1e-9


import rllib.curriculum as C


def test_scheduled_value_linear_interpolation():
    sched = [[0, 1.0], [100, 0.0]]
    assert C.scheduled_value(sched, 0) == 1.0
    assert abs(C.scheduled_value(sched, 50) - 0.5) < 1e-9
    assert C.scheduled_value(sched, 100) == 0.0


def test_scheduled_value_clamps_outside_range():
    sched = [[10, 0.5], [20, 1.0]]
    assert C.scheduled_value(sched, 0) == 0.5     # before first knot -> first value
    assert C.scheduled_value(sched, 999) == 1.0   # past last knot -> last value


def test_scheduled_value_none_and_constant():
    assert C.scheduled_value(None, 5) is None          # no schedule -> no override
    assert C.scheduled_value([[0, 0.7]], 999) == 0.7   # single knot -> constant


import types


class _FakeRewardStack:
    def __init__(self):
        self.dense_scale = 1.0
    def set_dense_scale(self, s):
        self.dense_scale = s


class _FakeTeamEnv:
    def __init__(self):
        self._reward_stack = _FakeRewardStack()
        self.cfg = types.SimpleNamespace(red_action_scale=1.0, red_start_r_max=None)


class _FakeEnvRunnerGroup:
    def __init__(self, inner):
        # runner.env._inner is the QuidditchTeamEnv
        self._runner = types.SimpleNamespace(
            env=types.SimpleNamespace(_inner=inner))
    def foreach_env_runner(self, fn, **kw):
        return [fn(self._runner)]


def _curr_algo(curriculum, lifetime_steps, inner):
    return types.SimpleNamespace(
        iteration=1,
        config=types.SimpleNamespace(env_config={"curriculum": curriculum}),
        env_runner_group=_FakeEnvRunnerGroup(inner),
        env_runner=types.SimpleNamespace(),
        # result carries the lifetime step count (read by the callback)
        _lifetime=lifetime_steps,
    )


def test_curriculum_callback_pushes_scheduled_values():
    inner = _FakeTeamEnv()
    algo = _curr_algo({
        "dense_scale_schedule": [[0, 1.0], [1000, 0.0]],
        "red_action_scale_schedule": [[0, 0.5], [1000, 1.0]],
    }, lifetime_steps=500, inner=inner)
    cb = C.CurriculumCallback()
    cb.on_train_result(algorithm=algo,
                       result={"num_env_steps_sampled_lifetime": 500})
    assert abs(inner._reward_stack.dense_scale - 0.5) < 1e-9   # halfway anneal
    assert abs(inner.cfg.red_action_scale - 0.75) < 1e-9       # halfway 0.5->1.0


def test_curriculum_callback_noop_without_schedules():
    inner = _FakeTeamEnv()
    algo = _curr_algo({}, lifetime_steps=500, inner=inner)
    cb = C.CurriculumCallback()
    cb.on_train_result(algorithm=algo,
                       result={"num_env_steps_sampled_lifetime": 500})
    assert inner._reward_stack.dense_scale == 1.0   # untouched
    assert inner.cfg.red_action_scale == 1.0
