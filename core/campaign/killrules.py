"""Deterministic kill-rule evaluation over W&B history rows.

Pure functions only — no network, no wandb import.  The controller skill
passes already-fetched history (via dsim campaign-status, which uses
core.campaign.telemetry) so this module stays trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True)
class KillReason:
    rule: str
    detail: str


@dataclass(frozen=True)
class KillVerdict:
    kill: bool
    reasons: list[KillReason]
    latest: Row


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and x != x


def _latest_value(rows: list[Row], metric: str) -> tuple[Any, Any]:
    """Most recent non-null value of metric and the step it occurred at."""
    for row in reversed(rows):
        v = row.get(metric)
        if v is not None:
            return v, row.get("_step")
    return None, None


def _value_at_step(rows: list[Row], metric: str, step: Any) -> Any:
    """Baseline value at the latest baseline step <= step (step-aligned compare)."""
    if step is None:
        return None
    best = None
    for row in rows:
        s, v = row.get("_step"), row.get(metric)
        if s is None or v is None:
            continue
        if s <= step:
            best = v
        else:
            break
    return best


def evaluate(
    rows: list[Row],
    rules: list[dict],
    *,
    baselines: dict[str, list[Row]] | None = None,
) -> KillVerdict:
    """Evaluate declared kill-rules against history rows.

    rows: oldest-first list of W&B history dicts (each with `_step`, `_runtime`).
    rules: list of rule dicts (see kill-rule schema in the plan/spec).
    baselines: run-path -> baseline history rows, for collapse_vs_baseline.
    """
    baselines = baselines or {}
    reasons: list[KillReason] = []
    latest = rows[-1] if rows else {}

    for rule in rules:
        kind = rule["type"]
        if kind == "nan":
            for m in rule["metrics"]:
                val, _ = _latest_value(rows, m)
                if _is_nan(val):
                    reasons.append(KillReason("nan", f"{m} is NaN"))
        elif kind == "floor_at_step":
            m, floor, gate = rule["metric"], rule["min"], rule["at_step"]
            cur_step = latest.get("_step") or 0
            if cur_step >= gate:
                val, step = _latest_value(rows, m)
                if val is not None and val < floor:
                    reasons.append(KillReason(
                        "floor_at_step",
                        f"{m} {val:.3f} < {floor} at step {step}"))
        elif kind == "collapse_vs_baseline":
            m, margin, b = rule["metric"], rule["margin"], rule["baseline_run"]
            val, step = _latest_value(rows, m)
            base_val = _value_at_step(baselines.get(b, []), m, step)
            if val is not None and base_val is not None and val < base_val - margin:
                reasons.append(KillReason(
                    "collapse_vs_baseline",
                    f"{m} {val:.3f} < baseline {base_val:.3f} - {margin} at step {step}"))
        elif kind == "budget":
            max_step, max_wc = rule.get("max_step"), rule.get("max_wallclock_s")
            cur_step, cur_wc = latest.get("_step"), latest.get("_runtime")
            if max_step is not None and cur_step is not None and cur_step >= max_step:
                reasons.append(KillReason("budget",
                                          f"step {cur_step} >= max_step {max_step}"))
            if max_wc is not None and cur_wc is not None and cur_wc >= max_wc:
                reasons.append(KillReason("budget",
                                          f"runtime {cur_wc}s >= max {max_wc}s"))
        else:
            raise ValueError(f"unknown kill-rule type: {kind!r}")

    return KillVerdict(kill=bool(reasons), reasons=reasons, latest=latest)
