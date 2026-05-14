"""CrashDetector classification: drone-into-wall, slow drone-drone (no crash),
fast drone-drone (crash). Drives contacts via manual qpos/qvel writes.

Ported from scripts/check_team_crash.py.
"""
from __future__ import annotations

import mujoco

from envs.quidditch.constants import ARENA_RADIUS, CRASH_VEL_THR
from tests.conftest import build_team_world, set_body_state


def test_drone_into_wall_above_threshold() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "blue_0", pos=(0.0, 0.0, 1.5))
        set_body_state(world, "red_0",  pos=(ARENA_RADIUS - 0.1, 0.0, 1.0),
                       vel=(2.0, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        for _ in range(20):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.wall["red_0"] > 0.0:
                assert ev.wall["red_0"] > 0.5, (
                    f"expected meaningful wall vrel, got {ev.wall['red_0']:.3f}"
                )
                return
        raise AssertionError("never made contact with wall in 20 steps")
    finally:
        world.disconnect()


def test_drone_drone_below_threshold_no_crash() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "red_0",  pos=(-0.03, 0.0, 1.0), vel=( 0.25, 0.0, 0.0))
        set_body_state(world, "blue_0", pos=( 0.03, 0.0, 1.0), vel=(-0.25, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        saw_contact = False
        saw_above_thr = False
        for _ in range(30):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.drone_drone is not None:
                saw_contact = True
                if ev.drone_drone[2] > CRASH_VEL_THR:
                    saw_above_thr = True
        assert saw_contact, "expected at least one drone-drone contact"
        assert not saw_above_thr, "did not expect any |v_rel| > CRASH_VEL_THR"
    finally:
        world.disconnect()


def test_drone_drone_above_threshold_crash() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "red_0",  pos=(-0.5, 0.0, 1.0), vel=( 2.0, 0.0, 0.0))
        set_body_state(world, "blue_0", pos=( 0.5, 0.0, 1.0), vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        for _ in range(120):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.drone_drone is not None and ev.drone_drone[2] > CRASH_VEL_THR:
                return
        raise AssertionError("never observed |v_rel| > CRASH_VEL_THR")
    finally:
        world.disconnect()
