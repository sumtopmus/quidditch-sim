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
