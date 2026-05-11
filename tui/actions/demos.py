"""The 4 demo actions — each shells out to ``mjpython demo/menu.py <key>``."""
from __future__ import annotations

from tui.actions.base import Action


def _demo_argv(key: str):
    def _f(_values: dict) -> list[str]:
        return ["mjpython", "demo/menu.py", key]
    return _f


DEMOS = [
    Action(
        key=k,
        label=k,
        group="Demo",
        glyph_attr="DEMO",
        fields=(),
        requires_mjpython=True,
        build_argv=_demo_argv(k),
    )
    for k in ("hover", "waypoint", "takedown", "score-through-tag")
]
