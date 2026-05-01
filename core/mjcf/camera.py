"""Camera math: load config, derive MJCF xyaxes, derive live-viewer params.

Used by `build_mjcf` (offscreen "fixed" camera) and by the World/Quadrotor
viewer setup (live mujoco.viewer).  Single source of truth for both.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ── camera config ────────────────────────────────────────────────────────────
# Eye and lookat in world coords (metres).  Used for both the offscreen "fixed"
# camera (videos) and the live viewer.  Override per-construction by passing
# `camera={"eye": (...), "lookat": (...)}` to the World; otherwise we try
# config/camera.toml and fall back to the hardcoded sideline view below.
_FALLBACK_CAMERA: dict = {
    "eye":    (2.9, -4.7, 2.9),
    "lookat": (0.5,  0.0, 1.3),
}


def load_camera_config(path: str | Path | None = None) -> dict:
    """Load {'eye': (x,y,z), 'lookat': (x,y,z)} from a TOML file.

    Defaults to ``<repo>/config/camera.toml``.  Falls back to the hardcoded
    sideline view if the file is missing or malformed.
    """
    import tomllib

    if path is None:
        # repo root = three levels up from this file: core/mjcf/camera.py -> core -> repo
        path = Path(__file__).resolve().parents[2] / "config" / "camera.toml"
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        cam = data["camera"]
        return {"eye": tuple(cam["eye"]), "lookat": tuple(cam["lookat"])}
    except (FileNotFoundError, KeyError):
        return dict(_FALLBACK_CAMERA)


def _camera_xyaxes(eye: tuple, lookat: tuple) -> tuple[str, str]:
    """Compute MJCF ``<camera>`` ``pos`` and ``xyaxes`` strings from eye+lookat.

    Returns (pos_str, xyaxes_str).  Raises if the look direction is parallel
    to world up (degenerate camera).
    """
    eye_a    = np.asarray(eye,    dtype=np.float64)
    lookat_a = np.asarray(lookat, dtype=np.float64)
    forward = lookat_a - eye_a
    fnorm = float(np.linalg.norm(forward))
    if fnorm < 1e-9:
        raise ValueError(f"Degenerate camera: eye == lookat ({eye!r})")
    forward /= fnorm

    # MuJoCo camera frame: +X right, +Y up, camera looks along -Z (OpenGL).
    # xaxis = right = cross(forward, up); yaxis = up_cam = cross(xaxis, forward).
    cam_x = np.cross(forward, [0.0, 0.0, 1.0])
    xnorm = float(np.linalg.norm(cam_x))
    if xnorm < 1e-6:
        raise ValueError(
            f"Degenerate camera: look direction parallel to world up "
            f"(eye={eye!r}, lookat={lookat!r}). Add a horizontal offset."
        )
    cam_x /= xnorm
    cam_y = np.cross(cam_x, forward)  # already unit length

    pos_str = f"{eye_a[0]:.4f} {eye_a[1]:.4f} {eye_a[2]:.4f}"
    xyaxes_str = (
        f"{cam_x[0]:.5f} {cam_x[1]:.5f} {cam_x[2]:.5f}  "
        f"{cam_y[0]:.5f} {cam_y[1]:.5f} {cam_y[2]:.5f}"
    )
    return pos_str, xyaxes_str


def _viewer_params(eye: tuple, lookat: tuple) -> tuple[float, float, float, np.ndarray]:
    """Convert (eye, lookat) → MuJoCo viewer (azimuth°, elevation°, distance, lookat).

    Matches the spherical convention used by ``mujoco.viewer``:
        azimuth   = angle of look-direction in xy-plane, measured from +x CCW
        elevation = arcsin of look-direction's z component (negative = looking down)
    """
    eye_a    = np.asarray(eye,    dtype=np.float64)
    lookat_a = np.asarray(lookat, dtype=np.float64)
    vec = eye_a - lookat_a                  # camera offset from lookat
    distance = float(np.linalg.norm(vec))
    azimuth = math.degrees(math.atan2(-vec[1], -vec[0]))
    elevation = math.degrees(math.asin(-vec[2] / distance))
    return azimuth, elevation, distance, lookat_a
