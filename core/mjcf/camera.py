"""Camera math: load config, derive MJCF xyaxes, derive live-viewer params.

Used by `build_mjcf` (offscreen "Fixed" camera + axis-aligned broadcast cams)
and by the World/Quadrotor viewer setup (live mujoco.viewer).  Single source
of truth for both.

Camera definitions live in ``config/camera.toml`` (copied from
``templates/camera.toml`` by ``make configs``, which is also invoked
by ``make install``).  The file is REQUIRED: ``load_camera_config``
raises FileNotFoundError if it's missing or any required cam below is
absent, with a hint to run ``make configs``.

Required cams in the toml: Fixed | North | East | South | West | Top.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# Cams the toml MUST declare; the loaders below raise if any are missing.
_REQUIRED_CAMS: tuple[str, ...] = (
    "Fixed", "North", "East", "South", "West", "Top",
)

_INSTALL_HINT = (
    "Run `make configs` to copy templates/camera.toml → config/camera.toml."
)


def _default_camera_config_path() -> Path:
    """Resolve to ``<repo>/config/camera.toml`` from this file's location."""
    # core/mjcf/camera.py → core → repo
    return Path(__file__).resolve().parents[2] / "config" / "camera.toml"


def _load_camera_toml(path: str | Path | None) -> dict:
    """Open + parse the toml.  Raises FileNotFoundError with install hint."""
    import tomllib

    p = Path(path) if path is not None else _default_camera_config_path()
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{p} not found — {_INSTALL_HINT}"
        ) from e

    cams = data.get("camera")
    if not isinstance(cams, dict):
        raise KeyError(
            f"{p}: expected a top-level [camera.<Name>] section per cam; "
            f"got {type(cams).__name__}.  {_INSTALL_HINT}"
        )

    missing = [n for n in _REQUIRED_CAMS if n not in cams]
    if missing:
        raise KeyError(
            f"{p}: missing required cam(s): {', '.join(missing)}.  "
            f"Required: {', '.join(_REQUIRED_CAMS)}.  {_INSTALL_HINT}"
        )
    return cams


def _cam_entry(name: str, entry: dict) -> tuple[tuple, tuple, float | None]:
    """Validate one [camera.<Name>] entry → (eye, lookat, fovy_or_None)."""
    for key in ("eye", "lookat"):
        if key not in entry:
            raise KeyError(f"camera.{name}: missing required key '{key}'.")
    fovy = float(entry["fovy"]) if "fovy" in entry else None
    return tuple(entry["eye"]), tuple(entry["lookat"]), fovy


# Type alias — one cam entry as it travels through WorldOptions / build_mjcf.
CameraSpec = tuple[str, tuple, tuple, float | None]   # (name, eye, lookat, fovy)


def load_camera_config(
    path: str | Path | None = None,
) -> tuple[CameraSpec, ...]:
    """Load all cameras from config/camera.toml as a tuple of (name, eye, lookat, fovy).

    Required cams: Fixed | North | East | South | West | Top.  Order in the
    returned tuple matches ``_REQUIRED_CAMS`` (Fixed first, compass cardinals
    in NESW order, Top last).

    Raises ``FileNotFoundError`` if the toml is missing, or ``KeyError`` if
    any required cam is absent or missing eye/lookat.  Both errors include
    a hint to run ``make configs`` to copy templates/camera.toml.
    """
    cams = _load_camera_toml(path)
    out: list[CameraSpec] = []
    for name in _REQUIRED_CAMS:
        eye, lookat, fovy = _cam_entry(name, cams[name])
        out.append((name, eye, lookat, fovy))
    return tuple(out)


def find_camera(cams: tuple[CameraSpec, ...], name: str) -> CameraSpec:
    """Look up one camera by name.  Raises KeyError if not present."""
    for entry in cams:
        if entry[0] == name:
            return entry
    raise KeyError(
        f"Camera {name!r} not in cam set; have: {[c[0] for c in cams]!r}."
    )


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
