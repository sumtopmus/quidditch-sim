"""Entrypoint for `python -m tui` and `make ui`."""
from __future__ import annotations

import argparse
import sys

from tui import theme
from tui.app import DroneSimApp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tui",
                                description="Drone-Quidditch interactive launcher")
    p.add_argument("--group", default=None,
                   help="Initial action group to focus (Demo | Train | Eval | Manage)")
    p.add_argument("--ascii", action="store_true",
                   help="Use ASCII fallback glyphs (no Nerd Font required)")
    args = p.parse_args(argv)

    if args.ascii:
        theme.use_ascii()

    DroneSimApp(initial_group=args.group).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
