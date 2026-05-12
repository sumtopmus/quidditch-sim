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


def test_switching_between_actions_with_overlapping_field_ids() -> None:
    """Action navigation rebuilds the form pane.  Multiple actions share
    field names like ``RUN_NAME`` (single-agent train, both team variants,
    resume, …) — so each rebuild must fully drop the previous form before
    mounting the new one, or Textual raises ``DuplicateIds``.

    This test switches between every pair of actions in a chain; if any
    transition leaves stale widgets behind, the next mount raises.
    """
    from tui.actions import ACTIONS
    from tui.widgets.action_form import ActionForm

    async def go() -> None:
        app = DroneSimApp()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            form = app.query_one(ActionForm)
            # Hit a variety of actions, especially the ones with form fields
            # that collide on names like field-RUN_NAME / field-TRIAL.
            key_order = [
                "hover", "train", "train-team-red", "train-team-blue",
                "resume", "eval", "eval-team", "promote", "lineage", "repro",
                "train",  # back to one with field-RUN_NAME / field-PRETRAIN
            ]
            by_key = {a.key: a for a in ACTIONS}
            for k in key_order:
                await form.set_action(by_key[k])
                await pilot.pause()

    asyncio.run(go())
