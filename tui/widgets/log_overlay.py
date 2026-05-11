"""Modal log overlay — last 200 lines of either subprocess slot.

Toggled by the `l` hotkey on the main app.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, RichLog


class LogOverlay(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "close"), ("l", "dismiss", "close")]

    def __init__(self, title: str, lines: list[str]) -> None:
        super().__init__()
        self._title = title
        self._lines = lines

    def compose(self) -> ComposeResult:
        log = RichLog(highlight=False, markup=False, wrap=True, id="log-body")
        yield Vertical(
            Static(self._title, classes="accent"),
            log,
            id="log-shell",
        )

    def on_mount(self) -> None:
        log = self.query_one("#log-body", RichLog)
        for line in self._lines:
            log.write(line)
