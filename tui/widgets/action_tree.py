"""Left-pane action tree — grouped, keyboard-navigable, disable-aware."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from tui import theme
from tui.actions import ACTIONS
from tui.actions.base import Action

_GROUP_ORDER = ("Demo", "Train", "Eval", "Manage")


class ActionTree(Widget):
    """Renders ACTIONS grouped by `group`, with selection + disabled state."""

    # Make the tree itself focusable so up/down/enter land here, not on the
    # inner VerticalScroll or its children.
    can_focus = True

    cursor: reactive[int] = reactive(0)
    training_disabled: reactive[bool] = reactive(False)

    class Selected(Message):
        def __init__(self, action: Action) -> None:
            super().__init__()
            self.action = action

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="tree-scroll")

    def on_mount(self) -> None:
        self._refresh()

    def watch_cursor(self, _old: int, _new: int) -> None:
        self._refresh()
        self.post_message(self.Selected(self._flat()[self.cursor]))

    def watch_training_disabled(self, _old: bool, _new: bool) -> None:
        self._refresh()

    def key_down(self) -> None:
        flat = self._flat()
        if self.cursor < len(flat) - 1:
            self.cursor += 1

    def key_up(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def key_enter(self) -> None:
        # Hand off to the app — it knows where the form pane is.
        self.app.action_focus_form()

    def key_right(self) -> None:
        # Mirror of Enter — pane-level navigation muscle memory.
        self.app.action_focus_form()

    def _flat(self) -> list[Action]:
        return [a for g in _GROUP_ORDER for a in ACTIONS if a.group == g]

    def _is_disabled(self, action: Action) -> bool:
        return self.training_disabled and action.is_training

    def _refresh(self) -> None:
        scroll = self.query_one("#tree-scroll", VerticalScroll)
        scroll.remove_children()
        flat = self._flat()
        ix = 0
        for group in _GROUP_ORDER:
            members = [a for a in ACTIONS if a.group == group]
            if not members:
                continue
            scroll.mount(Static(f"  {group}", classes="tree-group"))
            for a in members:
                glyph = getattr(theme.ACTIVE, a.glyph_attr, "·")
                lock = "  " + theme.ACTIVE.LOCK if self._is_disabled(a) else ""
                arrow = "▸" if ix == self.cursor else " "
                cls_parts: list[str] = []
                if self._is_disabled(a):
                    cls_parts.append("tree-disabled")
                if ix == self.cursor:
                    cls_parts.append("tree-cursor")
                scroll.mount(Static(
                    f"  {arrow} {glyph}  {a.label}{lock}",
                    classes=" ".join(cls_parts),
                ))
                ix += 1
