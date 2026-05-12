"""DroneSimApp — composes the three panes, training widget, and hotkeys."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from tui import theme
from tui.actions.base import Action
from tui.process.manager import ProcessManager
from tui.widgets.action_form import ActionForm
from tui.widgets.action_tree import ActionTree
from tui.widgets.log_overlay import LogOverlay
from tui.widgets.state_pane import StatePane
from tui.widgets.training_widget import TrainingWidget


class DroneSimApp(App):
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit_with_prompt", "quit"),
        ("ctrl+c", "stop_selected", "stop"),
        ("l", "show_logs", "logs"),
        ("t", "toggle_tensorboard", "tensorboard"),
        # Pane navigation: Escape from anywhere returns to the action tree.
        # Forward direction (tree → form) is bound on ActionTree itself
        # (Enter / Right) so it only fires when the tree has focus.
        ("escape", "focus_actions", "back to tree"),
    ]

    def __init__(self, *, initial_group: str | None = None) -> None:
        super().__init__()
        self._pm = ProcessManager()
        self._initial_group = initial_group

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            ActionTree(id="actions", classes="panel"),
            Vertical(
                ActionForm(id="form"),
                id="form-wrap",
                classes="panel",
            ),
            StatePane(id="state", classes="panel"),
        )
        yield TrainingWidget(id="training", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        # Hide training widget until a training subprocess is alive.
        self.query_one(TrainingWidget).stop_following()
        if self._initial_group:
            tree = self.query_one(ActionTree)
            # Naive: scan flat order, set cursor to first action whose group matches.
            for i, a in enumerate(tree._flat()):
                if a.group.lower() == self._initial_group.lower():
                    tree.cursor = i
                    break
        # Land focus on the tree so ↑/↓ navigation works without a prior click.
        self.query_one(ActionTree).focus()

    # ------------------------------------------------------------------
    # Tree → form coupling

    async def on_action_tree_selected(self, message: ActionTree.Selected) -> None:
        await self.query_one(ActionForm).set_action(message.action)

    # ------------------------------------------------------------------
    # Form → run / dry-run

    def on_action_form_run(self, message: ActionForm.Run) -> None:
        argv = message.action.build_argv(message.values)
        self._launch(message.action, argv)

    def on_action_form_dry_run(self, message: ActionForm.DryRun) -> None:
        rendered = " ".join(shlex.quote(s) for s in message.argv)
        self.push_screen(LogOverlay(title=f"Dry-run: {message.action.label}",
                                    lines=[rendered]))

    def _launch(self, action: Action, argv: list[str]) -> None:
        slot = "training" if action.is_training else "aux"
        if self._pm.is_running(slot):
            self.notify(f"{slot} slot is busy", severity="warning")
            return
        try:
            proc = self._pm.start(slot, argv)
        except Exception as e:  # noqa: BLE001
            self.notify(f"failed to start: {e}", severity="error")
            return
        self.notify(f"▶ {action.label} ({' '.join(argv[:3])} …)")
        if slot == "training":
            run_dir = self._guess_run_dir(action, argv)
            if run_dir is not None:
                self.query_one(TrainingWidget).follow(run_dir)
            self.query_one(ActionTree).training_disabled = True
        self.set_interval(1.0, lambda: self._poll_slot(slot, action))

    def _poll_slot(self, slot: str, action: Action) -> None:
        if self._pm.is_running(slot):
            return
        proc = self._pm._slots[slot]
        if proc is None:
            return
        rc = proc.returncode()
        sev = "information" if rc == 0 else "error"
        self.notify(f"{action.label} exited with code {rc} (press 'l' for logs)",
                    severity=sev)
        if slot == "training":
            self.query_one(TrainingWidget).stop_following()
            self.query_one(ActionTree).training_disabled = False

    def _guess_run_dir(self, action: Action, argv: list[str]) -> Path | None:
        """Heuristic: after launching, look for the newest trial under runs/."""
        from tui.state.scan import latest_trial
        # Poll the filesystem briefly — the run dir is created in the first
        # second of training. Caller is in the Textual event loop; a sync
        # sleep would block, so instead the training widget's first
        # refresh will surface the right dir via latest_trial once it lands.
        return latest_trial().path if latest_trial() is not None else None

    # ------------------------------------------------------------------
    # Hotkeys

    def action_stop_selected(self) -> None:
        # Stop the most-recent running slot — prefer aux if both alive.
        for slot in ("aux", "training"):
            if self._pm.is_running(slot):
                self._pm.stop(slot)
                self.notify(f"sent SIGINT to {slot}")
                return
        self.notify("nothing to stop")

    def action_show_logs(self) -> None:
        # Show whichever slot is alive — preference: aux first.
        for slot in ("aux", "training"):
            proc = self._pm.current(slot)
            if proc is not None:
                self.push_screen(LogOverlay(f"{slot} logs (last 200)", proc.tail(200)))
                return
        self.notify("no active subprocess")

    def action_toggle_tensorboard(self) -> None:
        # Quick-launch / kill TB without going through the form.
        proc = self._pm.current("aux")
        if proc is not None and "tensorboard" in proc.argv[0]:
            self._pm.stop("aux")
            self.notify("tensorboard stopped")
            return
        if self._pm.is_running("aux"):
            self.notify("aux slot busy — cannot start tensorboard", severity="warning")
            return
        try:
            self._pm.start("aux", ["tensorboard", "--logdir", "runs"])
            self.notify("tensorboard at http://localhost:6006")
        except Exception as e:  # noqa: BLE001
            self.notify(f"could not start tensorboard: {e}", severity="error")

    def action_focus_actions(self) -> None:
        self.query_one(ActionTree).focus()

    def action_focus_form(self) -> None:
        # Focus the first focusable descendant of the form pane (Input / Select /
        # Switch / Button).  If the current action has no fields, there's
        # nothing to focus inside — fall back to leaving focus on the tree.
        form = self.query_one(ActionForm)
        for child in form.query("Input, Select, Switch, Button"):
            child.focus()
            return

    def action_quit_with_prompt(self) -> None:
        if self._pm.is_running("training"):
            self.notify(
                "Training still running — quitting leaves it alive in the background. "
                "Press Ctrl-C first to stop, then q again.",
                severity="warning",
            )
            return
        self._pm.stop_all()
        self.exit()
