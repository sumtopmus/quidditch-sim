"""Smoke test: hover a cf2x quadrotor in MuJoCo for ~10 seconds.

Opens an interactive MuJoCo viewer window (use mouse to orbit/pan/zoom).
The drone climbs to 1 m and holds in the Quidditch arena (hoop + wall visible).

Run:  make demo  (and pick "hover")   or:  mjpython demo/hover_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.quadrotor import Quadrotor, CONTROL_HZ


HOVER_SECONDS = 10.0
START_POS = np.array([[0.0, 0.0, 0.0]])
START_ORN = np.array([[0.0, 0.0, 0.0]])
SETPOINT = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, yaw, z) — hold at (0,0,1m)


def main() -> None:
    quad = Quadrotor(start_pos=START_POS, start_orn=START_ORN, render=True)
    quad.set_mode(7)
    quad.set_setpoint(0, SETPOINT)

    print(f"Hovering for {HOVER_SECONDS:.0f} s at (0,0,1) — orbit the window with the mouse.")
    print(f"  step_period = {quad.step_period*1000:.2f} ms  ({CONTROL_HZ} Hz control)")

    n_steps = int(CONTROL_HZ * HOVER_SECONDS)
    for i in range(n_steps):
        quad.step()
        if i % CONTROL_HZ == 0:  # print once per second
            pos = quad.state(0)[3]
            print(f"  t={i/CONTROL_HZ:4.1f}s  pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        time.sleep(quad.step_period)  # pace to roughly real-time

    quad.idle()
    quad.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
