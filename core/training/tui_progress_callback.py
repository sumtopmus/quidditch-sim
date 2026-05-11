"""SB3 callback that writes a structured JSON progress file consumed by the TUI launcher.

The file is written atomically (tmp + os.replace) every `write_every` env steps.
Schema is documented in docs/superpowers/specs/2026-05-11-tui-launcher-design.md
section 3.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Literal

from stable_baselines3.common.callbacks import BaseCallback

SCHEMA_VERSION = 1
_RECENT_LEN = 16


class TUIProgressCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path | str,
        total_timesteps: int,
        kind: Literal["single", "team"],
        learner: str | None,
        opponent_spec: str | None,
        write_every: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._run_dir = Path(run_dir)
        self._total = int(total_timesteps)
        self._kind = kind
        self._learner = learner
        self._opponent_spec = opponent_spec
        self._write_every = (
            write_every if write_every is not None
            else max(1, self._total // 500)
        )
        self._recent: deque[float] = deque(maxlen=_RECENT_LEN)
        self._best_reward: float | None = None
        self._best_step: int = 0
        self._target = self._run_dir / "tui_progress.json"
        self._tmp = self._run_dir / "tui_progress.json.tmp"

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step % self._write_every != 0:
            return True
        self._write_snapshot(step)
        return True

    def _write_snapshot(self, step: int) -> None:
        ep_rew_mean = _safe_mean(self.model.ep_info_buffer, "r")
        ep_len_mean = _safe_mean(self.model.ep_info_buffer, "l")
        if ep_rew_mean is not None:
            self._recent.append(ep_rew_mean)
            if self._best_reward is None or ep_rew_mean > self._best_reward:
                self._best_reward = ep_rew_mean
                self._best_step = step

        elapsed_sec = max(time.time() - (self.model.start_time / 1e9), 0.0) \
            if isinstance(self.model.start_time, int) else 0.0
        fps = (step / elapsed_sec) if elapsed_sec > 0 else 0.0

        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "ts": time.time(),
            "run_name": self._run_dir.parent.name if self._run_dir.parent else "",
            "trial": self._run_dir.name,
            "kind": self._kind,
            "learner": self._learner,
            "opponent": self._opponent_spec,
            "step": step,
            "total_steps": self._total,
            "fps": fps,
            "elapsed_sec": elapsed_sec,
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "best_so_far": (
                {"reward": self._best_reward, "step": self._best_step}
                if self._best_reward is not None else None
            ),
            "recent_rewards": list(self._recent),
        }
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._tmp.write_text(json.dumps(snapshot))
        os.replace(self._tmp, self._target)


def _safe_mean(buf, key: str) -> float | None:
    if not buf:
        return None
    vals = [e[key] for e in buf if key in e]
    if not vals:
        return None
    return sum(vals) / len(vals)
