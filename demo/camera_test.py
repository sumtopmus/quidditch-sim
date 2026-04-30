"""Render the waypoint flight through the *fixed* camera and write an mp4.

Use this to iterate on config/camera.toml — edit the camera position/lookat,
re-run `make camera-test`, watch the result.  This is what the
VideoRecorderCallback sees during training (same MJCF "fixed" camera), so
getting the angle right here gets it right for checkpoint videos too.

Output:
    runs/camera_test/waypoint_camera_test.mp4   ← full flight video
    runs/camera_test/first_frame.png            ← still preview (faster to inspect)

Run:  make camera-test    (or:  python demo/camera_test.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.quadrotor import Quadrotor, load_camera_config
from demo.waypoint_demo import (
    WAYPOINTS,
    SETTLE_SECONDS,
    SECONDS_PER_WAYPOINT,
    FINAL_HOVER_SECONDS,
    START_POS,
    START_ORN,
)


VIDEO_W, VIDEO_H = 960, 540
FPS = 30
OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "camera_test"


def fly_record(
    quad: Quadrotor,
    wp: np.ndarray,
    yaw: float,
    seconds: float,
    frames: list[np.ndarray],
    every: int,
) -> None:
    """Hold a setpoint for `seconds` of sim time, capturing every `every`-th step."""
    setpoint = np.array([wp[0], wp[1], yaw, wp[2]], dtype=np.float32)
    quad.set_setpoint(0, setpoint)
    steps = int(seconds / quad.step_period)
    for i in range(steps):
        quad.step()
        if i % every == 0:
            frames.append(quad.render_frame(VIDEO_W, VIDEO_H))


def main() -> None:
    import imageio.v2 as imageio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = load_camera_config()
    print(f"[camera] eye={cam['eye']}  lookat={cam['lookat']}")

    markers = [
        ((float(wp[0]), float(wp[1]), float(wp[2])), "0.4 0.7 1.0 0.35", 0.15)
        for wp in WAYPOINTS
    ]

    quad = Quadrotor(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=False,
        markers=markers,
        camera=cam,
    )
    quad.set_mode(7)

    frames: list[np.ndarray] = []
    every = max(1, int((1.0 / FPS) / quad.step_period))

    print(f"[settle] {SETTLE_SECONDS}s at start altitude")
    fly_record(quad, np.array([0.0, 0.0, 1.5]), 0.0, SETTLE_SECONDS, frames, every)

    for i, wp in enumerate(WAYPOINTS):
        nxt = WAYPOINTS[i + 1] if i + 1 < len(WAYPOINTS) else wp
        dx, dy = nxt[0] - wp[0], nxt[1] - wp[1]
        yaw = float(np.arctan2(dy, dx)) if (dx or dy) else 0.0
        print(f"[waypoint {i}] -> ({wp[0]:+.2f}, {wp[1]:+.2f}, {wp[2]:+.2f})")
        fly_record(quad, wp, yaw, SECONDS_PER_WAYPOINT, frames, every)

    print(f"[done] hovering {FINAL_HOVER_SECONDS}s")
    fly_record(quad, WAYPOINTS[-1], 0.0, FINAL_HOVER_SECONDS, frames, every)

    quad.disconnect()

    mp4 = OUT_DIR / "waypoint_camera_test.mp4"
    png = OUT_DIR / "first_frame.png"
    with imageio.get_writer(str(mp4), fps=FPS, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    imageio.imwrite(str(png), frames[0])
    print(f"\nWrote {len(frames)} frames @ {FPS} fps -> {mp4}")
    print(f"First-frame preview                  -> {png}")


if __name__ == "__main__":
    main()
