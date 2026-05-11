"""Fresh-read disk scanner for the TUI: promoted models, runs, trials, checkpoints.

No caching — every call is a fresh ``os.scandir``. Cheap for low-hundreds of dirs.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

_LIVE_WINDOW_SEC = 30.0
# Prefix-agnostic: handles legacy single-agent files (ppo_hoop_<step>_steps.zip)
# alongside new ones (ppo_<step>_steps.zip) and any future prefix. The regex
# anchors on `_<digits>_steps.zip` at end of name.
_CKPT_RE = re.compile(r"_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class PromotedModel:
    name: str
    path: Path
    alias: str | None = None


@dataclass(frozen=True)
class Trial:
    run_name: str
    name: str
    path: Path
    has_best_model: bool
    is_live: bool


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


def promoted_models(models_root: Path = Path("models")) -> list[PromotedModel]:
    if not models_root.is_dir():
        return []
    out: list[PromotedModel] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "best_model.zip").is_file():
            continue
        alias_file = entry / "alias.txt"
        alias = alias_file.read_text().strip() if alias_file.is_file() else None
        out.append(PromotedModel(name=entry.name, path=entry, alias=alias))
    return out


def run_names(runs_root: Path = Path("runs")) -> list[str]:
    if not runs_root.is_dir():
        return []
    return sorted(p.name for p in runs_root.iterdir() if p.is_dir())


def trials_in_run(run_name: str, runs_root: Path = Path("runs")) -> list[Trial]:
    run_dir = runs_root / run_name
    if not run_dir.is_dir():
        return []
    trials: list[Trial] = []
    for t in run_dir.iterdir():
        if not t.is_dir():
            continue
        progress = t / "tui_progress.json"
        is_live = False
        if progress.is_file():
            is_live = (time.time() - progress.stat().st_mtime) < _LIVE_WINDOW_SEC
        trials.append(Trial(
            run_name=run_name,
            name=t.name,
            path=t,
            has_best_model=(t / "best_model.zip").is_file(),
            is_live=is_live,
        ))
    return sorted(trials, key=lambda x: x.name, reverse=True)


def checkpoints(run_name: str, trial_name: str,
                runs_root: Path = Path("runs")) -> list[Checkpoint]:
    ckpt_dir = runs_root / run_name / trial_name / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    ckpts: list[Checkpoint] = []
    for f in ckpt_dir.iterdir():
        m = _CKPT_RE.search(f.name)
        if m and f.is_file():
            ckpts.append(Checkpoint(step=int(m.group(1)), path=f))
    return sorted(ckpts, key=lambda c: c.step, reverse=True)


def latest_trial(runs_root: Path = Path("runs")) -> Trial | None:
    if not runs_root.is_dir():
        return None
    candidates: list[Trial] = []
    for run in run_names(runs_root):
        candidates.extend(trials_in_run(run, runs_root))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.name)
