"""Smoke test: the TUI mounts and renders one frame without raising.

Unit tests on individual widgets miss bugs that only surface during Textual's
render path — e.g. accidentally overriding ``Widget._render`` (a private
Textual API) and breaking the visual pipeline.  This test exercises the full
mount + render once, headlessly, so those bugs fail in CI.
"""
from __future__ import annotations

import asyncio

import pytest

from tui.app import DroneSimApp


def test_app_mounts_and_renders_one_frame_without_error() -> None:
    async def go() -> None:
        app = DroneSimApp()
        async with app.run_test(size=(160, 40)) as pilot:
            # Two pauses let Textual run mount → first compose → first render.
            await pilot.pause()
            await pilot.pause()

    # Surface any exception from the Textual render loop as a test failure.
    asyncio.run(go())
