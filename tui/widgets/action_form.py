"""Centre-pane form: renders a selected Action's fields, validates, builds argv."""
from __future__ import annotations

from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch


class FormInput(Input):
    """Text input with vim-style nav / edit modes.

    On focus, the field starts in **nav mode**: the cursor is hidden,
    character input is suppressed, and ↑/↓/←/→/Tab/Shift+Tab bubble up to
    ``ActionForm`` for field navigation.  Pressing **Enter** switches to
    **edit mode** — the cursor appears, the field accepts typing, and the
    standard Input bindings apply.  **Esc** returns to nav mode.  Losing
    focus always resets to nav mode.

    The cursor is hidden in nav mode via the ``nav-mode`` class, which
    overrides ``.input--cursor`` styling to be transparent.
    """

    DEFAULT_CSS = """
    FormInput.nav-mode .input--cursor {
        background: transparent;
        color: transparent;
        text-style: none;
    }
    """

    edit_mode: reactive[bool] = reactive(False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Don't blink — visibility is governed by the nav-mode class instead.
        self.cursor_blink = False

    def on_focus(self) -> None:
        self.edit_mode = False

    def on_blur(self) -> None:
        self.edit_mode = False

    def watch_edit_mode(self, _old: bool, new: bool) -> None:
        if new:
            self.remove_class("nav-mode")
        else:
            self.add_class("nav-mode")

    def on_mount(self) -> None:
        # Apply the initial class so cursor is hidden before first focus.
        self.add_class("nav-mode")

    def on_key(self, event: events.Key) -> None:
        if self.edit_mode:
            if event.key == "escape":
                event.stop()
                self.edit_mode = False
                return
            # Otherwise: default Input behavior (cursor, typing, etc.).
            return
        # Nav mode below.
        if event.key == "enter":
            event.stop()
            self.edit_mode = True
            return
        # Let navigation keys bubble — ActionForm / Input bindings handle them.
        if event.key in ("up", "down", "left", "right",
                         "tab", "shift+tab", "escape"):
            return
        # Swallow everything else so characters don't insert in nav mode.
        event.stop()
        event.prevent_default()


class FormSelect(Select):
    """A Select where ↑/↓ navigate form fields instead of opening the menu.

    Textual's default Select binds ``enter,down,space,up`` to ``show_overlay``.
    Subclass BINDINGS in Textual **accumulate** through MRO rather than
    replace — so adding ``enter,space`` here would not remove the inherited
    ``down`` / ``up`` triggers.  The fix is to bind ``down`` and ``up``
    explicitly to a different action; Textual matches per-key during dispatch,
    so the subclass's per-key entry wins over the parent's comma-string entry.

    Enter and space continue to open the overlay via the inherited binding.
    Once the overlay is open, focus moves into it and these bindings no
    longer apply — overlay's own ↑/↓ (option navigation) take over.
    """

    BINDINGS = [
        Binding("down", "app.focus_next",     "next field", show=False),
        Binding("up",   "app.focus_previous", "prev field", show=False),
    ]

from tui import theme
from tui.actions.base import Action, FieldKind, FieldSpec
from tui.state import scan


class ActionForm(Widget):
    """Renders the currently-selected action's form fields and Run / Dry-run buttons."""

    # Keep field navigation consistent with the action tree: ↑/↓ moves between
    # focusable form children.  These fire as the key event bubbles up from
    # the focused child (Input / FormSelect / Switch / Button), so the user
    # doesn't have to Tab through fields.
    #
    # Left arrow: jump to the previous sibling in a Horizontal row (e.g.
    # Run ← Dry-run); if there's no previous sibling, hop back to the action
    # tree.  Inside Input, the Input's own ``cursor_left`` binding wins (it's
    # on the focused widget so it dispatches first), so text editing is
    # unaffected.
    BINDINGS = [
        Binding("up",   "app.focus_previous", "prev field", show=False),
        Binding("down", "app.focus_next",     "next field", show=False),
        Binding("left", "leftmost_or_tree",      "tree",       show=False),
    ]

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._action: Action | None = None
        self._values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="form-body")

    async def set_action(self, action: Action) -> None:
        self._action = action
        self._values = {f.name: f.default for f in action.fields}
        await self._rebuild()

    async def _rebuild(self) -> None:
        # NOTE: this method intentionally avoids the name `_render` — that
        # collides with Textual's Widget._render, which must return a Visual.
        # Overriding it (to return None and mount children) breaks the render
        # pipeline with AttributeError on NoneType.render_strips.
        #
        # remove_children() is async — it returns an AwaitRemove that the
        # event loop must process before subsequent mount() calls run.
        # Without `await`, the new mounts race ahead of the removal and
        # DuplicateIds fires whenever two actions share a field name like
        # RUN_NAME.
        body = self.query_one("#form-body", Vertical)
        await body.remove_children()
        a = self._action
        if a is None:
            await body.mount(Static("Select an action ↑↓", classes="dim"))
            return

        glyph = getattr(theme.ACTIVE, a.glyph_attr, "·")
        children: list[Widget] = [
            Static(f"{glyph}  {a.label}", classes="accent"),
            Static(""),
        ]
        for f in a.fields:
            children.append(Static(f.label + (" *" if f.required else "")))
            children.append(self._build_widget(f))
        children.append(Static(""))
        children.append(Horizontal(
            Button("Run", id="btn-run", variant="success"),
            Button("Dry-run", id="btn-dryrun"),
        ))
        await body.mount(*children)

    def _build_widget(self, f: FieldSpec) -> Widget:
        wid = f"field-{f.name}"
        if f.kind == FieldKind.TEXT:
            return FormInput(value=str(f.default or ""), id=wid)
        if f.kind == FieldKind.INT:
            return FormInput(value=str(f.default or ""), id=wid, type="integer")
        if f.kind == FieldKind.BOOL:
            return Switch(value=bool(f.default), id=wid)
        # picker:* — fill choices from scan
        return FormSelect(self._picker_choices(f), id=wid, prompt="(none)")

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

    # ------------------------------------------------------------------
    # Left-arrow navigation: previous-in-row, or back to the tree.

    def action_leftmost_or_tree(self) -> None:
        focused = self.app.focused
        if focused is None:
            return
        parent = focused.parent
        if isinstance(parent, Horizontal):
            siblings = [c for c in parent.children if c.can_focus]
            if focused in siblings:
                ix = siblings.index(focused)
                if ix > 0:
                    siblings[ix - 1].focus()
                    return
        # Leftmost in its row (or not in a row) — hop back to the action tree.
        from tui.widgets.action_tree import ActionTree
        self.app.query_one(ActionTree).focus()
