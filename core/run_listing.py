"""Enumerate `runs/` for resume / promote / list-runs workflows.

Each `runs/<run_name>/<YYYYMMDD_HHMMSS>/` is a trial; per the Hydra Part 1
convention.  `list_runs` returns one row per `<run_name>` with its latest
trial and that trial's latest checkpoint resolved.

Public API:
    list_runs(runs_dir=Path("runs"), run_filter=None) -> list[RunEntry]
    resolve_trial(run_name, trial=None, runs_dir=Path("runs")) -> Path
    resolve_checkpoint(trial_dir, ckpt=None) -> Path
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path("runs")
_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class RunEntry:
    run_name: str
    run_dir: Path
    latest_trial: Path
    latest_checkpoint: Path | None


def list_runs(runs_dir: Path = RUNS_DIR, run_filter: str | None = None) -> list[RunEntry]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    out: list[RunEntry] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if run_filter and run_filter not in d.name:
            continue
        trials = sorted([t for t in d.iterdir() if t.is_dir()])
        if not trials:
            continue
        latest = trials[-1]
        ckpt = _latest_checkpoint(latest)
        out.append(RunEntry(
            run_name=d.name, run_dir=d.resolve(),
            latest_trial=latest.resolve(),
            latest_checkpoint=ckpt.resolve() if ckpt else None,
        ))
    return out


def resolve_trial(
    run_name: str,
    *,
    trial: str | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    run_dir = Path(runs_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"no such run: {run_dir}")
    if trial is not None:
        td = run_dir / trial
        if not td.exists():
            raise FileNotFoundError(f"no such trial: {td}")
        return td.resolve()
    trials = sorted([t for t in run_dir.iterdir() if t.is_dir()])
    if not trials:
        raise FileNotFoundError(f"no trials under {run_dir}")
    return trials[-1].resolve()


def resolve_checkpoint(trial_dir: Path, *, ckpt: str | None = None) -> Path:
    cks = Path(trial_dir) / "checkpoints"
    if not cks.exists():
        raise FileNotFoundError(f"no checkpoints/ under {trial_dir}")
    if ckpt is not None:
        p = cks / (ckpt if ckpt.endswith(".zip") else ckpt + ".zip")
        if not p.exists():
            raise FileNotFoundError(f"no such checkpoint: {p}")
        return p.resolve()
    p = _latest_checkpoint(trial_dir)
    if p is None:
        raise FileNotFoundError(f"no .zip checkpoints under {cks}")
    return p


def _latest_checkpoint(trial_dir: Path) -> Path | None:
    cks = Path(trial_dir) / "checkpoints"
    if not cks.exists():
        return None
    best: tuple[int, Path] | None = None
    for f in cks.glob("*.zip"):
        m = _CKPT_STEPS_RE.search(f.name)
        if not m:
            continue
        steps = int(m.group(1))
        if best is None or steps > best[0]:
            best = (steps, f)
    return best[1].resolve() if best else None
