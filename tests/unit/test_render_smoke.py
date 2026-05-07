"""One-frame offscreen render smoke test — catches asset/mesh/camera
regressions that headless episode loops never exercise.

New test, no source script.  Builds a minimal single-drone world,
renders one 240x320 frame, asserts shape + dtype.
"""
from __future__ import annotations

import mujoco
import numpy as np

from core.drone.cf2x import cf2x_assets, cf2x_fragment
from core.world import World
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment


def test_offscreen_render_one_frame_no_crash() -> None:
    fragments = [
        cf2x_assets(with_collision_meshes=False),
        cf2x_fragment(),  # default prefix "drone", visual-only
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=False),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    try:
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        try:
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data)
            frame = renderer.render()
            assert frame.shape == (240, 320, 3), f"unexpected shape {frame.shape}"
            assert frame.dtype == np.uint8, f"unexpected dtype {frame.dtype}"
        finally:
            renderer.close()
    finally:
        world.disconnect()
