"""Telemetry reader — uses an injected api_factory, never the network."""
from __future__ import annotations

from core.campaign.telemetry import read_telemetry


class _FakeRun:
    def __init__(self):
        self.summary = {"eval/success_rate": 0.42, "_step": 5_000_000}

    def history(self, keys=None, pandas=True):
        assert pandas is False, "telemetry must request pandas=False"
        return [
            {"_step": 1_000_000, "_runtime": 60.0, "eval/success_rate": 0.20},
            {"_step": 2_000_000, "_runtime": 120.0, "eval/success_rate": 0.42},
        ]


class _FakeApi:
    def __init__(self):
        self.requested = None

    def run(self, path):
        self.requested = path
        return _FakeRun()


def test_read_telemetry_returns_rows_and_summary():
    fake = _FakeApi()
    tel = read_telemetry("ent/proj/abc123", api_factory=lambda: fake)
    assert fake.requested == "ent/proj/abc123"
    assert len(tel.rows) == 2
    assert tel.rows[-1]["eval/success_rate"] == 0.42
    assert tel.summary["_step"] == 5_000_000


def test_read_telemetry_rows_are_plain_dicts():
    tel = read_telemetry("ent/proj/x", api_factory=lambda: _FakeApi())
    assert all(isinstance(r, dict) for r in tel.rows)


class _FakeSparseRun:
    """Mimics wandb: history(keys=...) returns [] if ANY requested metric was
    never logged, and metrics live on disjoint steps (eval vs train)."""

    SERIES = {
        "eval/success_rate": [(50_000, 0.0), (100_000, 0.1)],
        "train/loss": [(40_000, 2.0), (90_000, 1.5)],
        # "rollout/ep_rew_mean" intentionally never logged
    }

    def __init__(self):
        self.summary = {"_step": 100_000}

    def history(self, keys=None, pandas=True):
        assert pandas is False
        metrics = [k for k in keys if k not in ("_step", "_runtime")]
        # The wandb gotcha: a single absent key zeroes the whole call.
        if any(m not in self.SERIES for m in metrics):
            return []
        out = []
        for m in metrics:
            for step, val in self.SERIES[m]:
                out.append({"_step": step, "_runtime": float(step), m: val})
        return out


class _FakeSparseApi:
    def run(self, path):
        return _FakeSparseRun()


def test_absent_metric_key_does_not_blind_the_monitor():
    """A never-logged key in `keys` must not collapse the fetch to zero rows."""
    tel = read_telemetry("ent/proj/x", api_factory=lambda: _FakeSparseApi())
    # Per-key fetch + merge recovers all logged points despite ep_rew_mean
    # (absent) and eval/train living on disjoint steps.
    assert len(tel.rows) == 4
    sr = [(r["_step"], r["eval/success_rate"]) for r in tel.rows
          if "eval/success_rate" in r]
    assert sr == [(50_000, 0.0), (100_000, 0.1)]
    assert tel.rows[-1]["_step"] == 100_000  # oldest-first, highest step last
