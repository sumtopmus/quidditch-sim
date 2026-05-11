"""Visual constants for the TUI: Nerd Font glyphs + ASCII fallback.

Nerd Font codepoints used (private-use area U+E000+):
- nf-fa-gamepad           : demos
- nf-fa-rocket            : single-agent train
- nf-fa-flag              : team train (red / blue)
- nf-fa-refresh           : resume / repro
- nf-fa-crosshairs        : eval
- nf-fa-bar_chart         : tensorboard
- nf-fa-trophy            : promote
- nf-fa-folder_open       : list-runs
- nf-fa-link              : lineage
- nf-fa-lock              : disabled action

These glyphs render correctly only with a Nerd Font installed. When
``Glyphs.ASCII`` is active (selected via the ``--ascii`` CLI flag) the
class attributes are replaced with bracket-letter forms.
"""
from __future__ import annotations


class Glyphs:
    DEMO        = ""
    TRAIN       = ""
    TRAIN_TEAM  = ""
    RESUME      = ""
    EVAL        = ""
    TENSORBOARD = ""
    PROMOTE     = ""
    LIST_RUNS   = ""
    LINEAGE     = ""
    REPRO       = ""
    LOCK        = ""
    LIVE        = "●"
    RUNNING     = "⏵"
    UP_ARROW    = "↑"
    DOWN_ARROW  = "↓"


class GlyphsASCII:
    DEMO        = "[D]"
    TRAIN       = "[T]"
    TRAIN_TEAM  = "[Tm]"
    RESUME      = "[R]"
    EVAL        = "[E]"
    TENSORBOARD = "[B]"
    PROMOTE     = "[P]"
    LIST_RUNS   = "[L]"
    LINEAGE     = "[Ln]"
    REPRO       = "[Rp]"
    LOCK        = "[x]"
    LIVE        = "*"
    RUNNING     = ">"
    UP_ARROW    = "^"
    DOWN_ARROW  = "v"


# Default — overridden in tui/app.py based on --ascii flag.
ACTIVE: type = Glyphs


def use_ascii() -> None:
    global ACTIVE
    ACTIVE = GlyphsASCII


def use_nerd() -> None:
    global ACTIVE
    ACTIVE = Glyphs
