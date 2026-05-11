"""Regression: widgets with custom __init__ must forward kwargs to super().

Textual's ``Widget.__init__`` accepts ``id``, ``classes``, ``name``, and
``disabled`` as keyword args.  A subclass that overrides ``__init__`` without
forwarding ``**kwargs`` breaks those — and the resulting ``TypeError`` only
surfaces at compose-time, when the app is mounted in production code, not at
import-time.  This test fails fast at unit-test time instead.
"""
from __future__ import annotations

from tui.widgets.action_form import ActionForm
from tui.widgets.training_widget import TrainingWidget


def test_action_form_accepts_id_and_classes_kwargs() -> None:
    w = ActionForm(id="form", classes="panel")
    assert w.id == "form"
    assert "panel" in w.classes


def test_training_widget_accepts_id_and_classes_kwargs() -> None:
    w = TrainingWidget(id="training", classes="panel")
    assert w.id == "training"
    assert "panel" in w.classes
