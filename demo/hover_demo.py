"""Smoke test: spin up PyFlyt, hover a QuadX drone for a few seconds in the GUI."""

import time
import numpy as np
from PyFlyt.core import Aviary

# Drone starts 1m above origin, level
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,  # opens PyBullet GUI window
    drone_type="quadx",
)

# Flight mode 7 = position setpoint (x, y, yaw, z)
env.set_mode(7)
setpoint = np.array([0.0, 0.0, 0.0, 1.0])  # hold at origin, 1m altitude
env.set_setpoint(0, setpoint)

# Step physics for ~10 seconds (PyFlyt default = 240 Hz physics)
for _ in range(240 * 10):
    env.step()
    time.sleep(1 / 240)  # roughly real-time so you can see it

env.close()
