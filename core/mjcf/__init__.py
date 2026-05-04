"""MJCF composition primitives — assemble a MuJoCo scene from fragments.

Public API:
    SceneFragment      — composable bundle of MJCF children for top-level sections
    merge_all          — reduce an iterable of fragments
    WorldOptions       — configuration for the top-level <mujoco> document
    build_mjcf         — assemble fragments into a complete MJCF string
    load_camera_config — load all cameras from config/camera.toml as a tuple
                         of (name, eye, lookat, fovy_or_None); raises
                         FileNotFoundError if the toml is missing
    find_camera        — look up one camera by name in a loaded cam tuple

Internal helpers (used by fragment factories) live in submodules:
    core.mjcf.camera    — camera math (xyaxes, viewer params, toml loader)
    core.mjcf.meshes    — procedural mesh data (torus, cylinder shell, markers)
"""

from core.mjcf.fragment import SceneFragment, merge_all
from core.mjcf.document import WorldOptions, build_mjcf
from core.mjcf.camera import find_camera, load_camera_config

__all__ = [
    "SceneFragment",
    "merge_all",
    "WorldOptions",
    "build_mjcf",
    "load_camera_config",
    "find_camera",
]
