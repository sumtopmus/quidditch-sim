"""Smoke test — confirms the project's top-level packages import cleanly.

Lives at the tests/ root (not under unit/ or integration/) because it
exercises pytest discovery itself; if this can't find the modules,
nothing else will.
"""
from __future__ import annotations


def test_imports_envs_quidditch() -> None:
    import envs.quidditch.simple_env  # noqa: F401
    import envs.quidditch.team_env    # noqa: F401


def test_imports_core() -> None:
    import core.world          # noqa: F401
    import core.drone.cf2x     # noqa: F401
