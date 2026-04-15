"""Fly a single QuadX through a sequence of waypoints, with visible target markers.

Stepping stone toward the Phase-2 hoop env: exercises position-setpoint flight
mode (mode 7 — x, y, yaw, z), waypoint switching, and PyBullet visual markers.

Aviary step() already sleeps to match real time when render=True, so we just
need to give each waypoint enough simulated seconds to be reachable.

Run:  python waypoint_demo.py
"""

import numpy as np
import pybullet as p

from PyFlyt.core import Aviary

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

SETTLE_SECONDS = 2.0       # climb to start altitude before first waypoint
SECONDS_PER_WAYPOINT = 5.0 # give the controller enough time
REACH_TOL = 0.3
FINAL_HOVER_SECONDS = 5.0

START_POS = np.array([[0.0, 0.0, 1.0]])
START_ORN = np.array([[0.0, 0.0, 0.0]])


def add_marker(env: Aviary, position, rgba=(0.4, 0.7, 1.0, 0.25), radius=0.15):
    """Drop a non-colliding coloured sphere at `position` as a visual cue.

    Aviary subclasses pybullet_utils.bullet_client.BulletClient, so we call
    pybullet functions directly on `env` (no physicsClientId kwarg needed).
    """
    visual_id = env.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=rgba,
    )
    env.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=list(position),
    )


def fly_to(env: Aviary, wp: np.ndarray, yaw: float, seconds: float) -> np.ndarray:
    """Command a position setpoint and step for `seconds` (sim time).

    Returns final position. Breaks early if within REACH_TOL and visibly stable.
    """
    setpoint = np.array([wp[0], wp[1], yaw, wp[2]], dtype=np.float32)
    env.set_setpoint(0, setpoint)

    steps = int(seconds * env.physics_hz / env.updates_per_step)
    pos = env.state(0)[-1]
    for step in range(steps):
        env.step()
        pos = env.state(0)[-1]
        # log every ~0.5 s of sim time
        if step % max(1, steps // int(seconds * 2)) == 0:
            dist = float(np.linalg.norm(pos - wp))
            print(f"    t={step * env.updates_per_step / env.physics_hz:4.1f}s  "
                  f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  dist={dist:.2f}")
    return pos


def main() -> None:
    env = Aviary(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=True,
        drone_type="quadx",
    )
    env.set_mode(7)  # position setpoint: [x, y, yaw, z]

    for wp in WAYPOINTS:
        add_marker(env, wp)

    # Settle at start altitude before racing to the first waypoint.
    print(f"[settle] holding at start position for {SETTLE_SECONDS}s")
    fly_to(env, np.array([0.0, 0.0, 1.5]), yaw=0.0, seconds=SETTLE_SECONDS)

    for i, wp in enumerate(WAYPOINTS):
        next_wp = WAYPOINTS[i + 1] if i + 1 < len(WAYPOINTS) else wp
        dx, dy = next_wp[0] - wp[0], next_wp[1] - wp[1]
        yaw = float(np.arctan2(dy, dx)) if (dx or dy) else 0.0

        print(f"[waypoint {i}] -> ({wp[0]:+.2f}, {wp[1]:+.2f}, {wp[2]:+.2f})  yaw={np.degrees(yaw):+.0f}deg")
        final = fly_to(env, wp, yaw, SECONDS_PER_WAYPOINT)
        dist = float(np.linalg.norm(final - wp))
        status = "reached" if dist < REACH_TOL else "timeout"
        print(f"[waypoint {i}] {status}  final_dist={dist:.2f}\n")

    print(f"[done] hovering {FINAL_HOVER_SECONDS}s before closing")
    fly_to(env, WAYPOINTS[-1], yaw=0.0, seconds=FINAL_HOVER_SECONDS)

    env.close()


if __name__ == "__main__":
    main()
