"""Interactive demo selector.

Lists the available demos, prompts for a choice, and runs it.  Invoked by
`make demo` (under mjpython, since each demo opens the MuJoCo viewer).

Add a new demo by appending a (key, label, module_path) entry to DEMOS;
the module must expose a top-level main() function.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# (key, one-line description, module path) — module must expose main()
DEMOS: list[tuple[str, str, str]] = [
    ("hover",    "Hover at 1 m in the Quidditch arena (hoop + wall)", "demo.hover_demo"),
    ("waypoint", "Fly a triangular waypoint loop in empty space",      "demo.waypoint_demo"),
]


def _prompt() -> int | None:
    """Print the menu and return the chosen index, or None to quit."""
    print("Available demos:")
    width = max(len(k) for k, _, _ in DEMOS)
    for i, (key, desc, _) in enumerate(DEMOS, 1):
        print(f"  {i}) {key:<{width}}  {desc}")
    print()

    while True:
        raw = input(f"Choose a demo [1-{len(DEMOS)} or name, q to quit]: ").strip().lower()
        if raw in ("", "q", "quit", "exit"):
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(DEMOS):
                return idx
        for i, (key, _, _) in enumerate(DEMOS):
            if raw == key:
                return i
        print(f"  invalid choice: {raw!r} — try again.")


def main() -> None:
    idx = _prompt()
    if idx is None:
        print("No demo selected.")
        return
    key, _, module_path = DEMOS[idx]
    print(f"\n>>> running '{key}' ({module_path})\n")
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise RuntimeError(f"demo module {module_path!r} has no main()")
    module.main()


if __name__ == "__main__":
    main()
