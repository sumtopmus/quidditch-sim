"""Read snapshots of ``tui_progress.json`` written by TUIProgressCallback.

Tolerant to: missing file, partial JSON (a mid-write race despite the callback
using atomic os.replace — defensive), and unsupported newer schema versions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION_SUPPORTED = 1


@dataclass(frozen=True)
class ProgressSnapshot:
    schema_version: int
    ts: float
    run_name: str
    trial: str
    kind: str
    learner: str | None
    opponent: str | None
    step: int
    total_steps: int
    fps: float
    elapsed_sec: float
    ep_rew_mean: float | None
    ep_len_mean: float | None
    best_so_far: dict[str, Any] | None
    recent_rewards: list[float]


def read_snapshot(path: Path) -> ProgressSnapshot | None:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("schema_version") != SCHEMA_VERSION_SUPPORTED:
        return None
    try:
        return ProgressSnapshot(**data)
    except TypeError:
        return None
