"""Right-pane state: promoted models + recent trials. Refreshes every 5s."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from tui import theme
from tui.state import scan

_REFRESH_SEC = 5.0
_TRIALS_SHOWN = 10


class StatePane(Widget):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="state-scroll")

    def on_mount(self) -> None:
        self._refresh()
        self._timer: Timer = self.set_interval(_REFRESH_SEC, self._refresh)

    def _refresh(self) -> None:
        scroll = self.query_one("#state-scroll", VerticalScroll)
        scroll.remove_children()

        scroll.mount(Static("Promoted models", classes="accent"))
        models = scan.promoted_models()
        if not models:
            scroll.mount(Static("  (none)", classes="dim"))
        else:
            for m in models:
                label = f"  • {m.alias}" if m.alias else f"  • {m.name}"
                scroll.mount(Static(label))
                if m.alias:
                    scroll.mount(Static(f"    {m.name}", classes="dim"))

        scroll.mount(Static(""))
        scroll.mount(Static("Recent trials", classes="accent"))
        all_trials = []
        for run in scan.run_names():
            all_trials.extend(scan.trials_in_run(run))
        recent = sorted(all_trials, key=lambda t: t.name, reverse=True)[:_TRIALS_SHOWN]
        if not recent:
            scroll.mount(Static("  (none)", classes="dim"))
        else:
            for t in recent:
                live = f"  {theme.ACTIVE.LIVE} live" if t.is_live else ""
                scroll.mount(Static(f"  {t.run_name}"))
                scroll.mount(Static(f"    {t.name}{live}",
                                    classes="good" if t.is_live else "dim"))
