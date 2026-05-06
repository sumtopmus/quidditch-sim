"""Asserts the team-env MJCF builds with the expected geoms.

Run:
    conda activate uav
    python scripts/check_team_mjcf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco

from core.world import World
from core.drone.cf2x import cf2x_assets, cf2x_fragment
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)


def main() -> None:
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True,
                       with_tag_sphere=True, tag_sphere_rgba=(1, 0, 0, 0.15)),
        cf2x_fragment(prefix="blue_0", with_collisions=True,
                       with_tag_sphere=True, tag_sphere_rgba=(0, 0, 1, 0.15)),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    w = World(fragments)
    m = w.model

    def gid(name: str) -> int:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert i >= 0, f"missing geom: {name!r}"
        return i

    g_red_probe       = gid("red_0_probe")
    g_blue_probe      = gid("blue_0_probe")
    g_red_tag_sphere  = gid("red_0_tag_sphere")
    g_blue_tag_sphere = gid("blue_0_tag_sphere")
    g_score_tube      = gid("hoop_0_score_tube")

    for g, name in [
        (g_red_probe,       "red_0_probe"),
        (g_blue_probe,      "blue_0_probe"),
        (g_red_tag_sphere,  "red_0_tag_sphere"),
        (g_blue_tag_sphere, "blue_0_tag_sphere"),
        (g_score_tube,      "hoop_0_score_tube"),
    ]:
        assert m.geom_contype[g]    == 0, f"{name}: contype != 0"
        assert m.geom_conaffinity[g] == 0, f"{name}: conaffinity != 0"

    for i in range(16):
        g = gid(f"arena_wall_seg_{i:02d}")
        assert m.geom_contype[g]    == 1, f"wall seg {i}: contype != 1"
        assert m.geom_conaffinity[g] == 1, f"wall seg {i}: conaffinity != 1"

    print(f"OK team-env MJCF: ngeom={m.ngeom}  red_probe={g_red_probe}  "
          f"blue_probe={g_blue_probe}  red_tag={g_red_tag_sphere}  "
          f"blue_tag={g_blue_tag_sphere}  score_tube={g_score_tube}")
    w.disconnect()


if __name__ == "__main__":
    main()
