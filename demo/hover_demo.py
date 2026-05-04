"""Smoke test: hover a cf2x quadrotor in MuJoCo for ~10 seconds.

Opens an interactive MuJoCo viewer window (use mouse to orbit/pan/zoom).
The drone climbs to 1 m and holds in the Quidditch arena (hoop + wall visible).

Run:  make demo  (and pick "hover")   or:  mjpython demo/hover_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.quadrotor import Quadrotor, CONTROL_HZ
from envs.quidditch.scene import hoop_fragment, arena_wall_fragment
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)


HOVER_SECONDS = 10.0
START_POS = np.array([[0.0, 0.0, 0.0]])
START_ORN = np.array([[0.0, 0.0, 0.0]])
SETPOINT = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, yaw, z) — hold at (0,0,1m)


def main() -> None:
    quad = Quadrotor.standalone(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=True,
        extra_fragments=[
            arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT),
            hoop_fragment("hoop", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
        ],
    )
    quad.set_mode(7)
    quad.set_setpoint(SETPOINT)

    print(f"Hovering for {HOVER_SECONDS:.0f} s at (0,0,1) — orbit the window with the mouse.")
    print(f"  step_period = {quad.step_period*1000:.2f} ms  ({CONTROL_HZ} Hz control)")

    n_steps = int(CONTROL_HZ * HOVER_SECONDS)
    for i in range(n_steps):
        quad.step()
        if i % CONTROL_HZ == 0:  # print once per second
            pos = quad.state()[3]
            print(f"  t={i/CONTROL_HZ:4.1f}s  pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    quad.idle()
    quad.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
