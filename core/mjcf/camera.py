"""Camera math: load config, derive MJCF xyaxes, derive live-viewer params.

Used by `build_mjcf` (offscreen "fixed" camera + axis-aligned broadcast cams)
and by the World/Quadrotor viewer setup (live mujoco.viewer).  Single source
of truth for both.

Camera definitions live in ``conf/camera/default.yaml`` (tracked in git, edit
in place — the pre-Hydra TOML setup had a templates/ → config/ copy ceremony,
which is gone now).  Loaded directly via PyYAML; not part of Hydra config
composition since cameras aren't an experiment axis.

Required cams: fixed | north | east | south | west | top.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# Cams the YAML MUST declare; the loaders below raise if any are missing.
_REQUIRED_CAMS: tuple[str, ...] = (
    "fixed", "north", "east", "south", "west", "top",
)

_INSTALL_HINT = (
    "Edit conf/camera/default.yaml (tracked in git — no copy ceremony required)."
)


def _default_camera_config_path() -> Path:
    """Resolve to ``<repo>/conf/camera/default.yaml`` from this file's location."""
    # core/mjcf/camera.py → core → repo
    return Path(__file__).resolve().parents[2] / "conf" / "camera" / "default.yaml"


def _load_camera_yaml(path: str | Path | None) -> dict:
    """Open + parse the YAML.  Raises FileNotFoundError with install hint."""
    import yaml

    p = Path(path) if path is not None else _default_camera_config_path()
    try:
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{p} not found — {_INSTALL_HINT}"
        ) from e

    cams = data.get("cameras")
    if not isinstance(cams, dict):
        raise KeyError(
            f"{p}: expected a top-level `cameras:` map of cam-name → entry; "
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
    """Validate one cameras.<name> entry → (eye, lookat, fovy_or_None)."""
    for key in ("eye", "lookat"):
        if key not in entry:
            raise KeyError(f"cameras.{name}: missing required key '{key}'.")
    fovy = float(entry["fovy"]) if "fovy" in entry else None
    return tuple(entry["eye"]), tuple(entry["lookat"]), fovy


# Type alias — one cam entry as it travels through WorldOptions / build_mjcf.
CameraSpec = tuple[str, tuple, tuple, float | None]   # (name, eye, lookat, fovy)


def load_camera_config(
    path: str | Path | None = None,
) -> tuple[CameraSpec, ...]:
    """Load all cameras from conf/camera/default.yaml as a tuple of (name, eye, lookat, fovy).

    Required cams: fixed | north | east | south | west | top.  Order in the
    returned tuple matches ``_REQUIRED_CAMS`` (fixed first, compass cardinals
    in NESW order, top last).

    Raises ``FileNotFoundError`` if the YAML is missing, or ``KeyError`` if
    any required cam is absent or missing eye/lookat.
    """
    cams = _load_camera_yaml(path)
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
