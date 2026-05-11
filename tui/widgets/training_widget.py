"""Docked-bottom training widget — reads the JSON status file every 500ms."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

from tui.process.progress import read_snapshot, ProgressSnapshot

_POLL_MS = 500


def _fmt_secs(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = max(hi - lo, 1e-9)
    return "".join(bars[1 + int((v - lo) / rng * 7)] for v in values)


class TrainingWidget(Widget):
    """Reads ``<run_dir>/tui_progress.json`` and renders progress + stats."""

    def __init__(self) -> None:
        super().__init__()
        self._run_dir: Path | None = None
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("", id="train-title"),
            ProgressBar(id="train-bar", show_eta=False, total=100),
            Static("", id="train-stats"),
            Static("", id="train-spark"),
            id="train-body",
        )

    def follow(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        if self._timer is None:
            self._timer = self.set_interval(_POLL_MS / 1000, self._refresh)
        self._refresh()

    def stop_following(self) -> None:
        self._run_dir = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.query_one("#train-title", Static).update("(no active training)")
        self.query_one("#train-stats", Static).update("")
        self.query_one("#train-spark", Static).update("")

    def _refresh(self) -> None:
        if self._run_dir is None:
            return
        snap = read_snapshot(self._run_dir / "tui_progress.json")
        if snap is None:
            self.query_one("#train-title", Static).update("waiting for first checkpoint…")
            return
        self._update_from(snap)

    def _update_from(self, s: ProgressSnapshot) -> None:
        title = f"⚑ Training — {s.run_name}/{s.trial}"
        if s.kind == "team":
            title += f"  ({s.learner} vs {s.opponent})"
        self.query_one("#train-title", Static).update(title)

        bar = self.query_one("#train-bar", ProgressBar)
        pct = 100 * s.step / max(s.total_steps, 1)
        bar.update(total=100, progress=pct)

        eta_sec = (s.elapsed_sec / max(s.step, 1)) * max(s.total_steps - s.step, 0)
        best = s.best_so_far or {"reward": float("nan"), "step": 0}
        rew = "n/a" if s.ep_rew_mean is None else f"{s.ep_rew_mean:.2f}"
        stats = (
            f"step {s.step:,} / {s.total_steps:,}   "
            f"ep_rew_mean {rew}   best {best.get('reward', 0):.2f} @ {best.get('step', 0):,}   "
            f"fps {s.fps:,.0f}   elapsed {_fmt_secs(s.elapsed_sec)}   "
            f"ETA {_fmt_secs(eta_sec)}"
        )
        self.query_one("#train-stats", Static).update(stats)
        self.query_one("#train-spark", Static).update(_sparkline(s.recent_rewards))
