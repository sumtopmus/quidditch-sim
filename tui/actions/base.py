"""Declarative action types for the TUI's action catalog.

An Action describes one form-driven invocation: name, glyph, group, fields,
and a ``build_argv`` callable that turns form values into a subprocess argv.
Actions are read by the action tree (left pane) and form (centre pane).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class FieldKind(str, Enum):
    TEXT = "text"
    INT = "int"
    BOOL = "bool"
    PICKER_MODELS = "picker:models"
    PICKER_RUNS = "picker:runs"
    PICKER_TRIALS_IN_RUN = "picker:trials_in_run"
    PICKER_CHECKPOINTS_IN_TRIAL = "picker:checkpoints_in_trial"


@dataclass(frozen=True)
class FieldSpec:
    name: str                       # form-field id, also passed to build_argv
    label: str                      # user-facing label
    kind: FieldKind
    required: bool = False
    default: Any = None
    depends_on: tuple[str, ...] = ()  # other field names whose values seed this picker


@dataclass(frozen=True)
class Action:
    key: str                        # unique id used in tree / hotkeys / CLI
    label: str                      # user-facing name shown in tree
    group: Literal["Demo", "Train", "Eval", "Manage"]
    glyph_attr: str                 # name of an attribute on tui.theme.ACTIVE (e.g. "DEMO")
    fields: tuple[FieldSpec, ...] = ()
    requires_mjpython: bool = False
    is_training: bool = False       # → occupies the training slot in ProcessManager
    is_tensorboard: bool = False    # → the tensorboard daemon
    build_argv: Callable[[dict[str, Any]], list[str]] = field(
        default=lambda _: [],
        repr=False,
    )
