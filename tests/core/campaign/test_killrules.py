"""Kill-rule evaluator — pure function over synthetic W&B history rows."""
from __future__ import annotations

import math

from core.campaign.killrules import evaluate


def _rows(*steps_metrics):
    """steps_metrics: (step, runtime, {metric: value}) tuples."""
    return [{"_step": s, "_runtime": rt, **m} for s, rt, m in steps_metrics]


def test_no_rules_never_kills():
    v = evaluate(_rows((100, 1.0, {"eval/success_rate": 0.5})), [])
    assert v.kill is False
    assert v.reasons == []
    assert v.latest["_step"] == 100


def test_nan_rule_fires_on_nan_metric():
    rows = _rows((100, 1.0, {"train/loss": math.nan}))
    v = evaluate(rows, [{"type": "nan", "metrics": ["train/loss"]}])
    assert v.kill is True
    assert v.reasons[0].rule == "nan"


def test_nan_rule_silent_when_finite():
    rows = _rows((100, 1.0, {"train/loss": 0.3}))
    v = evaluate(rows, [{"type": "nan", "metrics": ["train/loss"]}])
    assert v.kill is False


def test_floor_at_step_fires_after_gate():
    rows = _rows((3_000_000, 10.0, {"eval/success_rate": 0.04}))
    rule = {"type": "floor_at_step", "metric": "eval/success_rate",
            "min": 0.10, "at_step": 3_000_000}
    v = evaluate(rows, [rule])
    assert v.kill is True
    assert v.reasons[0].rule == "floor_at_step"


def test_floor_at_step_dormant_before_gate():
    rows = _rows((1_000_000, 10.0, {"eval/success_rate": 0.04}))
    rule = {"type": "floor_at_step", "metric": "eval/success_rate",
            "min": 0.10, "at_step": 3_000_000}
    assert evaluate(rows, [rule]).kill is False


def test_collapse_vs_baseline_fires():
    rows = _rows((2_000_000, 10.0, {"eval/success_rate": 0.20}))
    base = _rows((1_000_000, 5.0, {"eval/success_rate": 0.30}),
                 (2_000_000, 9.0, {"eval/success_rate": 0.45}))
    rule = {"type": "collapse_vs_baseline", "metric": "eval/success_rate",
            "baseline_run": "b", "margin": 0.15}
    v = evaluate(rows, [rule], baselines={"b": base})
    assert v.kill is True
    assert v.reasons[0].rule == "collapse_vs_baseline"


def test_collapse_vs_baseline_silent_when_close():
    rows = _rows((2_000_000, 10.0, {"eval/success_rate": 0.40}))
    base = _rows((2_000_000, 9.0, {"eval/success_rate": 0.45}))
    rule = {"type": "collapse_vs_baseline", "metric": "eval/success_rate",
            "baseline_run": "b", "margin": 0.15}
    assert evaluate(rows, [rule], baselines={"b": base}).kill is False


def test_budget_step_and_wallclock():
    rows = _rows((10_000_000, 7200.0, {"eval/success_rate": 0.5}))
    v = evaluate(rows, [{"type": "budget", "max_step": 10_000_000,
                         "max_wallclock_s": 7200}])
    assert v.kill is True
    assert {r.rule for r in v.reasons} == {"budget"}


def test_unknown_rule_type_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown kill-rule"):
        evaluate(_rows((1, 1.0, {})), [{"type": "bogus"}])
