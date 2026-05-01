"""Procedural mesh generators for inline MJCF assets.

Each generator returns space-separated strings suitable for direct use in
``<mesh vertex="..." normal="..." face="..."/>``.  Pure geometry — no MuJoCo
or Quidditch knowledge — so these can be reused by any fragment factory.
"""

from __future__ import annotations

import math


def _torus_mesh_data(
    major_r: float,
    minor_r: float,
    n_major: int = 64,
    n_minor: int = 12,
) -> tuple[str, str, str]:
    """Generate inline-MJCF torus mesh data, centred at origin, axis = +x.

    Parameterised as: ring lies in the YZ plane, tube cross-section in the
    plane spanned by (radial direction in YZ, +x).

    Returns (vertex_str, normal_str, face_str) — space-separated strings
    suitable for ``<mesh vertex="..." normal="..." face="..."/>``.
    """
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    for i in range(n_major):
        theta = 2.0 * math.pi * i / n_major
        ct, st = math.cos(theta), math.sin(theta)
        for j in range(n_minor):
            phi = 2.0 * math.pi * j / n_minor
            cp, sp = math.cos(phi), math.sin(phi)
            verts.append((
                minor_r * sp,
                ct * (major_r + minor_r * cp),
                st * (major_r + minor_r * cp),
            ))
            norms.append((sp, ct * cp, st * cp))

    faces: list[tuple[int, int, int]] = []
    for i in range(n_major):
        for j in range(n_minor):
            v00 = i * n_minor + j
            v01 = i * n_minor + (j + 1) % n_minor
            v10 = ((i + 1) % n_major) * n_minor + j
            v11 = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    vertex_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    normal_str = " ".join(f"{nx:.5f} {ny:.5f} {nz:.5f}" for nx, ny, nz in norms)
    face_str   = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex_str, normal_str, face_str


def _arena_wall_mesh_data(
    radius: float,
    height: float,
    thickness: float = 0.016,
    n: int = 64,
) -> tuple[str, str, str]:
    """Generate inline-MJCF mesh data for a closed thin-shell cylindrical wall.

    The wall extends from z=0 to z=height with inner radius (radius - t/2)
    and outer radius (radius + t/2).  Includes inner + outer surfaces and
    top + bottom edge caps so the shell is closed and visible from any angle.

    Returns (vertex_str, normal_str, face_str).
    """
    R_in  = radius - thickness / 2.0
    R_out = radius + thickness / 2.0
    H     = height

    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []

    def add(v: tuple[float, float, float], nrm: tuple[float, float, float]) -> int:
        idx = len(verts)
        verts.append(v)
        norms.append(nrm)
        return idx

    # Per-angle vertex indices for each of the four surfaces.  We duplicate
    # vertices at shared positions (e.g. (R_out·cosθ, R_out·sinθ, 0) appears
    # in both the outer surface and the bottom cap) so each surface gets its
    # own vertex normal — required for crisp edges where surfaces meet.
    outer_b: list[int] = []
    outer_t: list[int] = []
    inner_b: list[int] = []
    inner_t: list[int] = []
    top_o:   list[int] = []
    top_i:   list[int] = []
    bot_o:   list[int] = []
    bot_i:   list[int] = []

    for i in range(n):
        theta = 2.0 * math.pi * i / n
        ct, st = math.cos(theta), math.sin(theta)
        outer_b.append(add((R_out * ct, R_out * st, 0.0), ( ct,  st, 0.0)))
        outer_t.append(add((R_out * ct, R_out * st, H  ), ( ct,  st, 0.0)))
        inner_b.append(add((R_in  * ct, R_in  * st, 0.0), (-ct, -st, 0.0)))
        inner_t.append(add((R_in  * ct, R_in  * st, H  ), (-ct, -st, 0.0)))
        top_o.append(  add((R_out * ct, R_out * st, H  ), (0.0, 0.0,  1.0)))
        top_i.append(  add((R_in  * ct, R_in  * st, H  ), (0.0, 0.0,  1.0)))
        bot_o.append(  add((R_out * ct, R_out * st, 0.0), (0.0, 0.0, -1.0)))
        bot_i.append(  add((R_in  * ct, R_in  * st, 0.0), (0.0, 0.0, -1.0)))

    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Outer surface: CCW from outside → normal radially outward.
        faces.append((outer_b[i], outer_b[j], outer_t[j]))
        faces.append((outer_b[i], outer_t[j], outer_t[i]))
        # Inner surface: opposite winding → normal radially inward.
        faces.append((inner_b[i], inner_t[j], inner_b[j]))
        faces.append((inner_b[i], inner_t[i], inner_t[j]))
        # Top cap: normal +z.
        faces.append((top_o[i], top_o[j], top_i[j]))
        faces.append((top_o[i], top_i[j], top_i[i]))
        # Bottom cap: normal -z.
        faces.append((bot_o[i], bot_i[j], bot_o[j]))
        faces.append((bot_o[i], bot_i[i], bot_i[j]))

    vertex_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    normal_str = " ".join(f"{nx:.5f} {ny:.5f} {nz:.5f}" for nx, ny, nz in norms)
    face_str   = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex_str, normal_str, face_str


def _markers_xml(markers: list[tuple] | None) -> str:
    """Render a list of (pos, rgba_str, radius) tuples as non-colliding spheres.

    Returns an XML fragment suitable for a SceneFragment.worldbody slot
    (multiple <geom> lines joined by indented newlines).  Returns "" when
    markers is None or empty.
    """
    if not markers:
        return ""
    lines = []
    for (pos, rgba, radius) in markers:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        lines.append(
            f'<geom type="sphere" size="{radius}" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'rgba="{rgba}" contype="0" conaffinity="0"/>'
        )
    return "\n      ".join(lines)
