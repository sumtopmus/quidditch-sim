"""Render the hover demo and write an mp4 — defaults to the 2x2 grid.

The default mirrors what training videos record (``south``, ``east``,
``top``, and ``tpv`` stitched at 1080p), so this is the canonical
"is the next checkpoint video going to look right?" check.  Pass
``--cam NAME`` to preview a single named camera instead.

Use this to iterate on config/camera.toml (only affects the "fixed"
cam) or on the chase-cam offsets in core/quadrotor.py.  Output filenames
embed the cam name so multiple previews can co-exist.

Outputs (per --cam choice):
    runs/camera_test/hover_<cam>.mp4   ← full hover video
    runs/camera_test/hover_<cam>.png   ← still preview (last frame)

Available cams:  grid | fixed | north | east | south | west | top
                 | fpv | tpv | port | starboard

The per-drone chase cams (fpv, tpv, port, starboard) are prefix-scoped
in the model (this demo uses the ``drone`` prefix → ``drone_tpv`` etc.);
``World.resolve_cam_name`` auto-prefixes the bare names to whichever
drone is registered, so they Just Work here.

Run:  make camera-test                  # → hover_grid.mp4 (default; 1080p 2x2)
      make camera-test CAM=fixed        # → hover_fixed.mp4
      make camera-test CAM=tpv          # → hover_tpv.mp4 (drone_tpv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.quadrotor import Quadrotor
from demo.hover_demo import HOVER_SECONDS, START_POS, START_ORN, SETPOINT
from envs.quidditch.scene import hoop_fragment, arena_wall_fragment
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)

# Single-cam preview resolution.
SINGLE_W, SINGLE_H = 960, 540
# Per-cell resolution for grid preview — matches the training-callback
# default in templates/training.toml so this preview shows what gets
# recorded during checkpoint videos.
GRID_CELL_W, GRID_CELL_H = 960, 540
GRID_CAMS = ("south", "east", "top", "tpv")
FPS = 120
OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "camera_test"

VALID_CAMS = (
    "grid",
    "fixed",
    "north",
    "east",
    "south",
    "west",
    "top",
    "fpv",
    "tpv",
    "port",
    "starboard",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--cam",
        default="grid",
        choices=VALID_CAMS,
        help='Which camera to preview (default: "grid" — the 2x2 1080p stitch '
        "matching the training video callback).",
    )
    return p.parse_args()


def main() -> None:
    import imageio.v2 as imageio

    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[preview] rendering through: {args.cam}")

    quad = Quadrotor.standalone(
        start_pos=START_POS,
        start_orn=START_ORN,
        render=False,
        extra_fragments=[
            arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT),
            hoop_fragment("hoop", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
        ],
    )
    quad.set_mode(7)
    quad.set_setpoint(SETPOINT)

    if args.cam == "grid":
        capture = lambda: quad.render_grid(GRID_CAMS, GRID_CELL_W, GRID_CELL_H)
    else:
        # Single-cam path: drive the renderer directly so we can pick the
        # camera by name (World.render_frame is hardcoded to "fixed").
        # Resolve through the World so bare per-drone names (tpv, fpv,
        # port, starboard) auto-prefix to the registered drone.
        renderer = quad._world.get_renderer(SINGLE_W, SINGLE_H)
        cam_name = quad._world.resolve_cam_name(args.cam)
        if cam_name is None:
            raise SystemExit(
                f"camera-test: no camera named {args.cam!r} in the model"
            )

        def capture() -> np.ndarray:
            renderer.update_scene(quad._world.data, camera=cam_name)
            return renderer.render()[:, :, :3]

    frames: list[np.ndarray] = []
    every = max(1, int((1.0 / FPS) / quad.step_period))
    steps = int(HOVER_SECONDS / quad.step_period)

    print(f"[hover] recording {HOVER_SECONDS:.0f}s at (0,0,1) in Quidditch arena")
    for i in range(steps):
        quad.step()
        if i % every == 0:
            frames.append(capture())

    quad.disconnect()

    mp4 = OUT_DIR / f"hover_{args.cam}.mp4"
    png = OUT_DIR / f"hover_{args.cam}.png"
    with imageio.get_writer(
        str(mp4), fps=FPS, macro_block_size=None, ffmpeg_log_level="error"
    ) as w:
        for f in frames:
            w.append_data(f)  # type: ignore[attr-defined]
    imageio.imwrite(str(png), frames[-1])
    print(f"\nWrote {len(frames)} frames @ {FPS} fps -> {mp4}")
    print(f"Last-frame preview                   -> {png}")


if __name__ == "__main__":
    main()
