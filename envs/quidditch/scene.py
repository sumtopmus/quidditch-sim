"""Quidditch scene fragments — arena wall, hoop, score tube.

Each function returns a self-contained `SceneFragment` that composes with
a drone fragment (`core.drone.cf2x.cf2x_fragment`) via `core.mjcf.build_mjcf`
to produce a complete MJCF.

Multi-hoop scenes: call `hoop_fragment` once per hoop with distinct prefixes
(e.g. "red_hoop", "blue_hoop").  Each hoop's mesh asset, material, body, and
score-tube geom are independently namespaced.
"""

from __future__ import annotations

import numpy as np

from core.mjcf import SceneFragment
from core.mjcf.meshes import _torus_mesh_data, _arena_wall_mesh_data
from envs.quidditch.constants import (
    ARENA_WALL_HEIGHT,
    HOOP_SCORE_TUBE_HALF_LEN,
)


def hoop_fragment(
    prefix: str,
    center: tuple[float, float, float] | np.ndarray,
    outward_normal: tuple[float, float, float] | np.ndarray,
    radius: float,
) -> SceneFragment:
    """Goal hoop: ring mesh + pole + invisible scoring trigger volume.

    Args:
        prefix: namespace.  Mesh asset / material / body / score-tube geom
            are named ``f"{prefix}_ring"``, ``f"{prefix}_metal"``,
            ``f"{prefix}"``, ``f"{prefix}_score_tube"``.  For multi-hoop
            scenes use distinct prefixes ("red_hoop", "blue_hoop", ...).
        center: world-frame xyz of the ring's centre.
        outward_normal: unit vector pointing from arena centre outward
            through the ring.  Currently only ``(1, 0, 0)`` is supported;
            generalising to arbitrary normals would require building an
            euler rotation from the normal — not needed yet.
        radius: ring major radius (m).

    Returns:
        SceneFragment carrying one mesh+material asset and one body
        (containing the ring, pole, and score tube).
    """
    cx, cy, cz = (float(v) for v in center)
    nx, ny, nz = (float(v) for v in outward_normal)

    if (nx, ny, nz) != (1.0, 0.0, 0.0):
        raise NotImplementedError(
            f"hoop_fragment currently supports outward_normal=(1, 0, 0) "
            f"only (got {outward_normal!r}); extend to compute euler from "
            f"normal when adding hoops along other axes."
        )

    # ── mesh + material asset (namespaced) ────────────────────────────────
    v, n, f = _torus_mesh_data(radius, 0.012, n_major=64, n_minor=12)
    mesh_asset = (
        f'<mesh name="{prefix}_ring" vertex="{v}" normal="{n}" face="{f}"/>\n'
        f'    <material name="{prefix}_metal" rgba="1.0 0.45 0.0 1" '
        f'specular="0.5" shininess="0.5" reflectance="0.15"/>'
    )

    # ── body: ring + pole + invisible score tube ─────────────────────────
    base_z = cz - radius
    body_xml = (
        f'<body name="{prefix}" pos="0 0 0">\n'
        f'      <geom type="mesh" mesh="{prefix}_ring" '
        f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" material="{prefix}_metal" '
        f'contype="0" conaffinity="0"/>\n'
        f'      <geom type="cylinder" size="0.005" '
        f'fromto="{cx:.4f} {cy:.4f} 0  {cx:.4f} {cy:.4f} {base_z:.4f}" '
        f'rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>\n'
        f'      <!-- Invisible cylinder along the hoop normal — non-colliding;\n'
        f'           geometric query target for mj_geomDistance.  rgba alpha=0\n'
        f'           keeps it out of every render; bump alpha to e.g. 0.08\n'
        f'           for debugging. -->\n'
        f'      <geom name="{prefix}_score_tube" type="cylinder" '
        f'size="{radius:.4f} {HOOP_SCORE_TUBE_HALF_LEN:.4f}" '
        f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" euler="0 1.5708 0" '
        f'contype="0" conaffinity="0" rgba="0.1 1.0 0.3 0"/>\n'
        f'    </body>'
    )

    return SceneFragment(assets=(mesh_asset,), worldbody=(body_xml,))


def arena_wall_fragment(
    radius: float,
    height: float = ARENA_WALL_HEIGHT,
) -> SceneFragment:
    """Translucent cylindrical arena wall (closed thin-shell mesh).

    A single mesh — inner + outer surfaces + top + bottom caps — with a
    polycarbonate-like material.  Only one wall per scene; no prefix
    parameter (the mesh asset, material, body, and geom are all named
    "arena_wall" / "arena_glass" / "arena").
    """
    v, n, f = _arena_wall_mesh_data(radius, height, thickness=0.016, n=64)
    mesh_asset = (
        f'<mesh name="arena_wall" vertex="{v}" normal="{n}" face="{f}"/>\n'
        f'    <material name="arena_glass" rgba="0.6 0.85 1.0 0.35" '
        f'specular="0.3" shininess="0.5"/>'
    )
    body_xml = (
        '<body name="arena" pos="0 0 0">\n'
        '      <geom type="mesh" mesh="arena_wall" material="arena_glass" '
        'contype="0" conaffinity="0"/>\n'
        '    </body>'
    )
    return SceneFragment(assets=(mesh_asset,), worldbody=(body_xml,))
