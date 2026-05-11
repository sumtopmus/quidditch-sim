"""Single source of truth: ordered list of all actions."""
from __future__ import annotations

from tui.actions.demos import DEMOS
from tui.actions.eval import EVAL
from tui.actions.manage import MANAGE
from tui.actions.train import TRAIN

ACTIONS = [*DEMOS, *TRAIN, *EVAL, *MANAGE]
