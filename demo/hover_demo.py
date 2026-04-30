"""Smoke test: hover a cf2x quadrotor in MuJoCo for ~10 seconds.

Opens an interactive MuJoCo viewer window (use mouse to orbit/pan/zoom).
The drone should climb to 1 m and hold position indefinitely.

Run:
    conda activate uav
    cd quidditch-sim
    python demo/hover_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.quadrotor import Quadrotor, CONTROL_HZ

start_pos = np.array([[0.0, 0.0, 0.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

quad = Quadrotor(start_pos=start_pos, start_orn=start_orn, render=True)
quad.set_mode(7)

# Hold at origin, 1 m altitude
setpoint = np.array([0.0, 0.0, 0.0, 1.0])
quad.set_setpoint(0, setpoint)

print(f"Hovering for 10 s at (0,0,1) — orbit the window with the mouse.")
print(f"  step_period = {quad.step_period*1000:.2f} ms  ({CONTROL_HZ} Hz control)")

n_steps = CONTROL_HZ * 10  # 10 s
for i in range(n_steps):
    quad.step()
    if i % CONTROL_HZ == 0:  # print once per second
        s = quad.state(0)
        pos = s[3]
        print(f"  t={i/CONTROL_HZ:4.1f}s  pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    time.sleep(quad.step_period)  # pace to roughly real-time

quad.disconnect()
print("Done.")
