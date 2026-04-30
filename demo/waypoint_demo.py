"""Fly a single QuadX through a sequence of waypoints, with visible target markers.

Stepping stone toward the Phase-2 hoop env: exercises position-setpoint flight
mode (mode 7 — x, y, yaw, z), waypoint switching, and translucent-sphere markers.

Unlike PyFlyt's Aviary, MuJoCo's launch_passive viewer does NOT auto-pace to
real time, so we sleep `quad.step_period` after each step when rendering.

Run:  make waypoint   (or:  mjpython demo/waypoint_demo.py)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.quadrotor import Quadrotor

# --------------------------------------------------------------------------- #
# Waypoints — (x, y, z). Triangle around the origin, ~2 m radius, 1.5 m up.
# --------------------------------------------------------------------------- #
WAYPOINTS = np.array(
    [
        [2.0, 0.0, 1.5],
        [-1.0, 1.732, 1.5],
        [-1.0, -1.732, 1.5],
        [2.0, 0.0, 1.5],  # close the loop
    ],
    dtype=np.float32,
)

SETTLE_SECONDS = 2.0        # climb to start altitude before first waypoint
SECONDS_PER_WAYPOINT = 5.0  # give the controller enough time
REACH_TOL = 0.3
FINAL_HOVER_SECONDS = 5.0

START_POS = np.array([[0.0, 0.0, 1.0]])
START_ORN = np.array([[0.0, 0.0, 0.0]])


def fly_to(quad: Quadrotor, wp: np.ndarray, yaw: float, seconds: float) -> np.ndarray:
    """Command a position setpoint and step for `seconds` of sim time."""
    setpoint = np.array([wp[0], wp[1], yaw, wp[2]], dtype=np.float32)
    quad.set_setpoint(0, setpoint)

    steps = int(seconds / quad.step_period)
    log_every = max(1, steps // int(seconds * 2))
    pos = quad.state(0)[-1]
    for step in range(steps):
        quad.step()
        time.sleep(quad.step_period)  # pace to real-time so the viewer is watchable
        pos = quad.state(0)[-1]
        if step % log_every == 0:
            dist = float(np.linalg.norm(pos - wp))
            print(
                f"    t={step * quad.step_period:4.1f}s  "
                f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  dist={dist:.2f}"
            )
    return pos


def main() -> None:
    # Translucent blue spheres at each waypoint position
    markers = [
        ((float(wp[0]), float(wp[1]), float(wp[2])), "0.4 0.7 1.0 0.35", 0.15)
        for wp in WAYPOINTS
    ]

    quad = Quadrotor(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=True,
        markers=markers,
    )
    quad.set_mode(7)

    # Settle at start altitude before racing to the first waypoint.
    print(f"[settle] holding at start position for {SETTLE_SECONDS}s")
    fly_to(quad, np.array([0.0, 0.0, 1.5]), yaw=0.0, seconds=SETTLE_SECONDS)

    for i, wp in enumerate(WAYPOINTS):
        next_wp = WAYPOINTS[i + 1] if i + 1 < len(WAYPOINTS) else wp
        dx, dy = next_wp[0] - wp[0], next_wp[1] - wp[1]
        yaw = float(np.arctan2(dy, dx)) if (dx or dy) else 0.0

        print(f"[waypoint {i}] -> ({wp[0]:+.2f}, {wp[1]:+.2f}, {wp[2]:+.2f})  "
              f"yaw={np.degrees(yaw):+.0f}deg")
        final = fly_to(quad, wp, yaw, SECONDS_PER_WAYPOINT)
        dist = float(np.linalg.norm(final - wp))
        status = "reached" if dist < REACH_TOL else "timeout"
        print(f"[waypoint {i}] {status}  final_dist={dist:.2f}\n")

    print(f"[done] hovering {FINAL_HOVER_SECONDS}s before closing")
    fly_to(quad, WAYPOINTS[-1], yaw=0.0, seconds=FINAL_HOVER_SECONDS)

    quad.disconnect()


if __name__ == "__main__":
    main()
