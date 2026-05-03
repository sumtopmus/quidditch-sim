"""MJCF composition primitives — assemble a MuJoCo scene from fragments.

Public API:
    SceneFragment   — composable bundle of MJCF children for top-level sections
    merge_all       — reduce an iterable of fragments
    WorldOptions    — configuration for the top-level <mujoco> document
    build_mjcf      — assemble fragments into a complete MJCF string
    load_camera_config — load camera eye/lookat from config/camera.toml

Internal helpers (used by fragment factories) live in submodules:
    core.mjcf.camera    — camera math (xyaxes, viewer params, fallback)
    core.mjcf.meshes    — procedural mesh data (torus, cylinder shell, markers)
"""

from core.mjcf.fragment import SceneFragment, merge_all
from core.mjcf.document import WorldOptions, build_mjcf
from core.mjcf.camera import load_camera_config

__all__ = [
    "SceneFragment",
    "merge_all",
    "WorldOptions",
    "build_mjcf",
    "load_camera_config",
]
