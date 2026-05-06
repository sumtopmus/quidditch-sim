"""Drives drone-vs-{floor, wall, drone} contacts and asserts CrashDetector
classifies them correctly with the expected |v_rel · normal|.

Run:
    conda activate uav
    python scripts/check_team_crash.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from core.world import World
from core.drone.cf2x import cf2x_assets, cf2x_fragment
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment
from envs.quidditch.crash import CrashDetector
from envs.quidditch.constants import (
    ARENA_RADIUS, ARENA_WALL_HEIGHT, HOOP_CENTER, HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS, CRASH_VEL_THR,
)


def _build_world() -> tuple[World, CrashDetector]:
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True, with_tag_sphere=True),
        cf2x_fragment(prefix="blue_0", with_collisions=True, with_tag_sphere=True),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    w = World(fragments)
    return w, CrashDetector(w, ["red_0", "blue_0"])


def _qa(w: World, body_name: str) -> int:
    """qpos address of the body's first joint (free joint = 7 entries: xyz + quat)."""
    bid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt = int(w.model.body_jntadr[bid])
    return int(w.model.jnt_qposadr[jnt])


def _set_state(w: World, body: str, pos: tuple[float, float, float],
               vel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    qa = _qa(w, body)
    w.data.qpos[qa : qa + 3] = pos
    w.data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
    bid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body)
    jnt = int(w.model.body_jntadr[bid])
    qva = int(w.model.jnt_dofadr[jnt])
    w.data.qvel[qva : qva + 3] = vel
    w.data.qvel[qva + 3 : qva + 6] = 0.0


def case_drone_into_wall() -> None:
    print("=== [1/3] Drone-into-wall at 2.0 m/s ===")
    w, detector = _build_world()
    _set_state(w, "blue_0", pos=(0.0, 0.0, 1.5))
    _set_state(w, "red_0",  pos=(ARENA_RADIUS - 0.1, 0.0, 1.0), vel=(2.0, 0.0, 0.0))
    mujoco.mj_forward(w.model, w.data)
    for _ in range(20):
        mujoco.mj_step(w.model, w.data)
        ev = detector.events()
        if ev.wall["red_0"] > 0.0:
            print(f"   red_0 wall vrel={ev.wall['red_0']:.3f} m/s")
            assert ev.wall["red_0"] > 0.5, f"expected meaningful wall vrel, got {ev.wall['red_0']:.3f}"
            print("PASSED")
            w.disconnect()
            return
    raise AssertionError("never made contact with wall in 20 steps")


def case_slow_drone_drone() -> None:
    print("=== [2/3] Drone-drone at 0.5 m/s (below threshold, no crash) ===")
    w, detector = _build_world()
    _set_state(w, "red_0",  pos=(-0.03, 0.0, 1.0), vel=(0.25, 0.0, 0.0))
    _set_state(w, "blue_0", pos=( 0.03, 0.0, 1.0), vel=(-0.25, 0.0, 0.0))
    mujoco.mj_forward(w.model, w.data)
    saw_contact = False
    saw_above_thr = False
    for _ in range(30):
        mujoco.mj_step(w.model, w.data)
        ev = detector.events()
        if ev.drone_drone is not None:
            saw_contact = True
            if ev.drone_drone[2] > CRASH_VEL_THR:
                saw_above_thr = True
    assert saw_contact,         "expected at least one drone-drone contact"
    assert not saw_above_thr,   "did not expect any |v_rel| > CRASH_VEL_THR"
    print("PASSED (contact observed below threshold)")
    w.disconnect()


def case_fast_drone_drone() -> None:
    print("=== [3/3] Drone-drone at 2.0 m/s (above threshold, crash) ===")
    w, detector = _build_world()
    _set_state(w, "red_0",  pos=(-0.5, 0.0, 1.0), vel=(2.0, 0.0, 0.0))
    _set_state(w, "blue_0", pos=( 0.5, 0.0, 1.0), vel=(-2.0, 0.0, 0.0))
    mujoco.mj_forward(w.model, w.data)
    for step in range(120):
        mujoco.mj_step(w.model, w.data)
        ev = detector.events()
        if ev.drone_drone is not None and ev.drone_drone[2] > CRASH_VEL_THR:
            print(f"   step {step}: drone_drone vrel={ev.drone_drone[2]:.3f} m/s — CRASH")
            print("PASSED")
            w.disconnect()
            return
    raise AssertionError("never observed |v_rel| > CRASH_VEL_THR")


if __name__ == "__main__":
    case_drone_into_wall()
    print()
    case_slow_drone_drone()
    print()
    case_fast_drone_drone()
    print()
    print("All crash sub-cases PASSED.")
