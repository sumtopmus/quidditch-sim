"""Render the hover demo through the *fixed* camera and write an mp4.

Use this to iterate on config/camera.toml — edit the camera position/lookat,
re-run `make camera-test`, watch the result.  This is what the
VideoRecorderCallback sees during training (same MJCF "fixed" camera in the
Quidditch arena), so getting the angle right here gets it right for
checkpoint videos too.

Output:
    runs/camera_test/hover_camera_test.mp4   ← full hover video
    runs/camera_test/first_frame.png         ← still preview (faster to inspect)

Run:  make camera-test    (or:  python demo/camera_test.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.quadrotor import Quadrotor, load_camera_config
from demo.hover_demo import HOVER_SECONDS, START_POS, START_ORN, SETPOINT


VIDEO_W, VIDEO_H = 960, 540
FPS = 30
OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "camera_test"


def main() -> None:
    import imageio.v2 as imageio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = load_camera_config()
    print(f"[camera] eye={cam['eye']}  lookat={cam['lookat']}")

    quad = Quadrotor(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=False,
        camera=cam,
    )
    quad.set_mode(7)
    quad.set_setpoint(0, SETPOINT)

    frames: list[np.ndarray] = []
    every = max(1, int((1.0 / FPS) / quad.step_period))
    steps = int(HOVER_SECONDS / quad.step_period)

    print(f"[hover] recording {HOVER_SECONDS:.0f}s at (0,0,1) in Quidditch arena")
    for i in range(steps):
        quad.step()
        if i % every == 0:
            frames.append(quad.render_frame(VIDEO_W, VIDEO_H))

    quad.disconnect()

    mp4 = OUT_DIR / "hover_camera_test.mp4"
    png = OUT_DIR / "first_frame.png"
    with imageio.get_writer(str(mp4), fps=FPS, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    imageio.imwrite(str(png), frames[0])
    print(f"\nWrote {len(frames)} frames @ {FPS} fps -> {mp4}")
    print(f"First-frame preview                  -> {png}")


if __name__ == "__main__":
    main()
