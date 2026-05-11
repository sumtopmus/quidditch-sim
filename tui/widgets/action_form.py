"""Centre-pane form: renders a selected Action's fields, validates, builds argv."""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch

from tui import theme
from tui.actions.base import Action, FieldKind, FieldSpec
from tui.state import scan


class ActionForm(Widget):
    """Renders the currently-selected action's form fields and Run / Dry-run buttons."""

    class Run(Message):
        def __init__(self, action: Action, values: dict[str, Any]) -> None:
            super().__init__()
            self.action = action
            self.values = values

    class DryRun(Message):
        def __init__(self, action: Action, argv: list[str]) -> None:
            super().__init__()
            self.action = action
            self.argv = argv

    def __init__(self) -> None:
        super().__init__()
        self._action: Action | None = None
        self._values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="form-body")

    def set_action(self, action: Action) -> None:
        self._action = action
        self._values = {f.name: f.default for f in action.fields}
        self._render()

    def _render(self) -> None:
        body = self.query_one("#form-body", Vertical)
        body.remove_children()
        a = self._action
        if a is None:
            body.mount(Static("Select an action ↑↓", classes="dim"))
            return

        glyph = getattr(theme.ACTIVE, a.glyph_attr, "·")
        body.mount(Static(f"{glyph}  {a.label}", classes="accent"))
        body.mount(Static(""))  # spacer

        for f in a.fields:
            body.mount(Static(f.label + (" *" if f.required else "")))
            body.mount(self._build_widget(f))

        body.mount(Static(""))
        body.mount(Horizontal(
            Button("Run", id="btn-run", variant="success"),
            Button("Dry-run", id="btn-dryrun"),
        ))

    def _build_widget(self, f: FieldSpec) -> Widget:
        wid = f"field-{f.name}"
        if f.kind == FieldKind.TEXT:
            w = Input(value=str(f.default or ""), id=wid)
            return w
        if f.kind == FieldKind.INT:
            return Input(value=str(f.default or ""), id=wid, type="integer")
        if f.kind == FieldKind.BOOL:
            return Switch(value=bool(f.default), id=wid)
        # picker:* — fill choices from scan
        return Select(self._picker_choices(f), id=wid, prompt="(none)")

    def _picker_choices(self, f: FieldSpec) -> list[tuple[str, str]]:
        if f.kind == FieldKind.PICKER_MODELS:
            return [(m.alias or m.name, str(m.path)) for m in scan.promoted_models()]
        if f.kind == FieldKind.PICKER_RUNS:
            return [(n, n) for n in scan.run_names()]
        if f.kind == FieldKind.PICKER_TRIALS_IN_RUN:
            run = self._values.get("RUN_NAME") or ""
            return [(t.name, t.name) for t in scan.trials_in_run(run)] if run else []
        if f.kind == FieldKind.PICKER_CHECKPOINTS_IN_TRIAL:
            run = self._values.get("RUN_NAME") or ""
            trial = self._values.get("TRIAL") or ""
            if run and trial:
                return [
                    (c.path.with_suffix("").name, str(c.path.with_suffix("")))
                    for c in scan.checkpoints(run, trial)
                ]
            return []
        return []

    # ------------------------------------------------------------------
    # Event handlers — keep field state in self._values and refresh
    # dependent pickers when their source field changes.

    def on_input_changed(self, event: Input.Changed) -> None:
        field_name = event.input.id.removeprefix("field-")
        self._values[field_name] = event.value
        self._refresh_dependents_of(field_name)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        field_name = event.switch.id.removeprefix("field-")
        self._values[field_name] = event.value

    def on_select_changed(self, event: Select.Changed) -> None:
        field_name = event.select.id.removeprefix("field-")
        self._values[field_name] = event.value
        self._refresh_dependents_of(field_name)

    def _refresh_dependents_of(self, source: str) -> None:
        a = self._action
        if a is None:
            return
        for f in a.fields:
            if source in f.depends_on and f.kind.value.startswith("picker:"):
                try:
                    sel = self.query_one(f"#field-{f.name}", Select)
                except Exception:
                    continue
                sel.set_options(self._picker_choices(f))
                self._values[f.name] = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._action is None:
            return
        if event.button.id == "btn-run":
            if self._validate():
                self.post_message(self.Run(self._action, dict(self._values)))
        elif event.button.id == "btn-dryrun":
            argv = self._action.build_argv(self._values)
            self.post_message(self.DryRun(self._action, argv))

    def _validate(self) -> bool:
        for f in self._action.fields:  # type: ignore[union-attr]
            if f.required and not self._values.get(f.name):
                self.app.notify(f"required: {f.label}", severity="error")
                return False
        return True
