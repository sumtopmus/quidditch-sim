"""MJCF assembly: required geoms exist with correct contype/conaffinity bits.

Ported from scripts/check_team_mjcf.py.
"""
from __future__ import annotations

import mujoco

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


def test_team_env_mjcf_geoms_present_with_correct_collision_bits() -> None:
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True,
                      with_tag_sphere=True, tag_sphere_rgba=(1.0, 0.0, 0.0, 0.15)),
        cf2x_fragment(prefix="blue_0", with_collisions=True,
                      with_tag_sphere=True, tag_sphere_rgba=(0.0, 0.0, 1.0, 0.15)),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    try:
        m = world.model

        def gid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert i >= 0, f"missing geom: {name!r}"
            return i

        for name in (
            "red_0_probe",
            "blue_0_probe",
            "red_0_tag_sphere",
            "blue_0_tag_sphere",
            "hoop_0_score_tube",
        ):
            g = gid(name)
            assert m.geom_contype[g]    == 0, f"{name}: contype != 0"
            assert m.geom_conaffinity[g] == 0, f"{name}: conaffinity != 0"

        for i in range(16):
            name = f"arena_wall_seg_{i:02d}"
            g = gid(name)
            assert m.geom_contype[g]    == 1, f"{name}: contype != 1"
            assert m.geom_conaffinity[g] == 1, f"{name}: conaffinity != 1"
    finally:
        world.disconnect()
