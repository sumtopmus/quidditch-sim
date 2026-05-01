"""cf2x quadrotor — physical constants and MJCF fragment factories.

The cf2x is a small (27 g) Crazyflie-style quadrotor.

Physical-constant provenance (partial adoption from MuJoCo Menagerie):

  - **Mass**: 0.027 kg — agreed across Menagerie's `cf2.xml` and PyFlyt's
    `cf2x.yaml`.
  - **Inertia tensor** (`IXX = IYY = 2.3951e-5`, `IZZ = 3.2347e-5`):
    Menagerie's value, more physically accurate per the Crazyflie 2
    mechanical specs than PyFlyt's older defaults (1.4e-5, 2.17e-5).
  - **Motor coefficients** (`THRUST_COEF`, `TORQUE_COEF`, `MAX_RPM`,
    `ARM`): unchanged from PyFlyt's `cf2x.yaml`.  Menagerie expresses
    motors via `<motor gear=...>` actuator gears that aren't directly
    comparable; our RPM-squared thrust + reaction-torque model is
    independent of how the visual mesh was made and stays.

Two factories:

  - `cf2x_assets()` returns a `SceneFragment` carrying the seven mesh
    declarations + seven materials + the `.obj` file payloads.  Call it
    **once per scene**, regardless of how many drones share the world —
    mesh + material names are global in MJCF.
  - `cf2x_fragment(prefix)` returns a per-drone `SceneFragment`: one
    `<body>` (with its inertial, seven mesh geoms, IMU site, scoring
    probe) plus four sensors (gyro, velocimeter, framequat, framepos),
    all namespaced by `prefix`.  References meshes by name — `cf2_0`
    through `cf2_6` — declared by `cf2x_assets()`.
"""

from __future__ import annotations

import math
from pathlib import Path

from core.mjcf import SceneFragment


# ── cf2x physical constants ──────────────────────────────────────────────────
# Mass + inertia: from Menagerie (commit affef08, file bitcraze_crazyflie_2/cf2.xml).
# Motor coefficients: from PyFlyt cf2x.yaml — see module docstring.
MASS: float = 0.027         # kg
IXX:  float = 2.3951e-5     # kg⋅m²  (Menagerie)
IYY:  float = 2.3951e-5     # kg⋅m²  (Menagerie)
IZZ:  float = 3.2347e-5     # kg⋅m²  (Menagerie)
ARM:  float = 0.028         # m (motor distance from CoM along each diagonal)
THRUST_COEF: float = 3.16e-10  # N / RPM²
TORQUE_COEF: float = 7.94e-12  # N⋅m / RPM²
# max_rpm = sqrt(total_max_thrust / (4 × kF));  total_max_thrust = 2.0 N (cf2x.yaml)
MAX_RPM: float = math.sqrt(2.0 / (4.0 * THRUST_COEF))   # ≈ 39 775 RPM


# ── cf2x mesh assets ─────────────────────────────────────────────────────────
# Visual meshes vendored verbatim from MuJoCo Menagerie at
# repo/assets/cf2x/.  See assets/cf2x/README.md for source + commit hash.
_CF2X_ASSETS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "cf2x"
)
_CF2X_MESH_FILES: tuple[str, ...] = (
    "cf2_0.obj",
    "cf2_1.obj",
    "cf2_2.obj",
    "cf2_3.obj",
    "cf2_4.obj",
    "cf2_5.obj",
    "cf2_6.obj",
)
# Collision meshes — 32 convex hulls vendored from Menagerie's
# bitcraze_crazyflie_2/assets/cf2_collision_*.obj.  Loaded only when
# `cf2x_assets(with_collision_meshes=True)` is called.
_CF2X_COLLISION_FILES: tuple[str, ...] = tuple(
    f"cf2_collision_{i}.obj" for i in range(32)
)
# (mesh_name, material_name) per mesh — order + names match Menagerie's cf2.xml.
_CF2X_MESH_MATERIALS: tuple[tuple[str, str], ...] = (
    ("cf2_0", "propeller_plastic"),
    ("cf2_1", "medium_gloss_plastic"),
    ("cf2_2", "polished_gold"),
    ("cf2_3", "polished_plastic"),
    ("cf2_4", "burnished_chrome"),
    ("cf2_5", "body_frame_plastic"),
    ("cf2_6", "white"),
)


def cf2x_assets(*, with_collision_meshes: bool = False) -> SceneFragment:
    """Mesh + material assets for the Crazyflie 2 visuals.

    Call this **exactly once per scene** — the seven mesh names (cf2_0…
    cf2_6) and seven material names (propeller_plastic, …, white) are
    global in MJCF, so a multi-drone scene declares them once and every
    drone body refers to them by name.

    Args:
        with_collision_meshes: When True, also load Menagerie's 32
            cf2_collision_*.obj meshes and emit their <mesh> declarations.
            Required for any drone that uses cf2x_fragment(..., with_collisions=True).
            Default False (single-drone, non-colliding scenes don't need them
            and the .obj load adds tiny startup overhead).

    Returns a `SceneFragment` whose:
      - `assets` block carries seven `<material/>` declarations (rgba
        values copied verbatim from Menagerie's cf2.xml) followed by
        seven `<mesh file="cf2_<i>.obj"/>` declarations.  When
        ``with_collision_meshes=True``, 32 additional ``<mesh
        name="cf2_collision_<i>" file="cf2_collision_<i>.obj"/>``
        declarations are appended.
      - `asset_files` block carries the seven .obj files as
        ``(filename, bytes)`` tuples (plus the 32 collision .obj files
        when the flag is on).  `World` forwards them to
        ``mujoco.MjModel.from_xml_string(assets=...)`` so MuJoCo
        resolves the mesh `file="..."` references against in-memory
        bytes — no filesystem lookup at sim-init time.

    Per-drone bodies are emitted by `cf2x_fragment(prefix)`; they
    reference `cf2_0`…`cf2_6` declared here.
    """
    materials: tuple[str, ...] = (
        '<material name="propeller_plastic"    rgba="0.792 0.820 0.933 1"/>',
        '<material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>',
        '<material name="polished_gold"        rgba="0.969 0.878 0.6 1"/>',
        '<material name="polished_plastic"     rgba="0.631 0.659 0.678 1"/>',
        '<material name="burnished_chrome"     rgba="0.898 0.898 0.898 1"/>',
        '<material name="body_frame_plastic"   rgba="0.102 0.102 0.102 1"/>',
        '<material name="white"                rgba="1 1 1 1"/>',
    )
    meshes: tuple[str, ...] = tuple(
        f'<mesh name="{name}" file="{filename}"/>'
        for filename, (name, _mat) in zip(_CF2X_MESH_FILES, _CF2X_MESH_MATERIALS)
    )
    file_bytes: tuple[tuple[str, bytes], ...] = tuple(
        (filename, (_CF2X_ASSETS_DIR / filename).read_bytes())
        for filename in _CF2X_MESH_FILES
    )
    if with_collision_meshes:
        collision_meshes: tuple[str, ...] = tuple(
            f'<mesh name="cf2_collision_{i}" file="{filename}"/>'
            for i, filename in enumerate(_CF2X_COLLISION_FILES)
        )
        collision_file_bytes: tuple[tuple[str, bytes], ...] = tuple(
            (filename, (_CF2X_ASSETS_DIR / filename).read_bytes())
            for filename in _CF2X_COLLISION_FILES
        )
        meshes = meshes + collision_meshes
        file_bytes = file_bytes + collision_file_bytes
    return SceneFragment(
        assets=materials + meshes,
        asset_files=file_bytes,
    )


def cf2x_fragment(
    prefix: str = "drone",
    *,
    with_collisions: bool = False,
) -> SceneFragment:
    """A single cf2x drone (body + sensors) as a composable MJCF fragment.

    Args:
        prefix: namespace for every named element — body, freejoint, IMU
            site, scoring probe, and the four sensors.  Default "drone"
            preserves the historical single-drone names.  For multi-drone
            scenes pass distinct prefixes (e.g. "red_0", "blue_0").
        with_collisions: When True, append 32 collision-mesh geoms with
            contype=1 conaffinity=1 to the body.  Requires cf2x_assets(
            with_collision_meshes=True) to be present in the same scene.
            Default False — drone is fully non-colliding, matching the
            historical single-drone setup.

            The contype/conaffinity bits 1/1 mean the drone collides with
            anything else carrying bit 1 (currently only the floor —
            hoop and arena_wall use bit 0).  Multi-drone scenes get
            drone-drone + drone-floor collisions for free.  Override per-
            fragment if you want fancier team-bitmask schemes (e.g.
            bit 2 = "blue team", bit 3 = "obstacle").

    Returns:
        SceneFragment with one <body> in worldbody and four sensors
        (gyro, velocimeter, framequat, framepos), all prefixed.

    Notes:
        - The body's `<pos>` is hardcoded to `(0, 0, 0.03)` so the drone
          rests on the floor at MJCF compile time.  Per-episode start
          positions are written to qpos via `mj_resetData` in the World
          / Quadrotor reset path, not here.
        - The body references the seven Menagerie cf2 visual meshes
          (`cf2_0`…`cf2_6`) by name.  ``cf2x_assets()`` must also be in
          the scene's fragment list — call it exactly once per scene.
        - The probe geom is `contype=0 conaffinity=0` and is intended
          for mj_geomDistance queries (e.g. against a hoop_score_tube).
    """
    if with_collisions:
        collision_geoms = "\n".join(
            f'      <geom mesh="cf2_collision_{i}" contype="1" conaffinity="1" group="3"/>'
            for i in range(32)
        )
        collision_block = (
            "      <!-- Menagerie cf2 collision meshes (32 meshes, contype=1 "
            "conaffinity=1, group=3 hidden by default) -->\n"
            f"{collision_geoms}\n"
        )
    else:
        collision_block = ""

    body_xml = (
        f'<body name="{prefix}" pos="0 0 0.03">\n'
        f'      <freejoint name="{prefix}_root"/>\n'
        f'      <inertial mass="{MASS}" pos="0 0 0"\n'
        f'                diaginertia="{IXX} {IYY} {IZZ}"/>\n'
        f'      <!-- Menagerie cf2 visuals (7 mesh geoms, all non-colliding) -->\n'
        f'      <geom mesh="cf2_0" material="propeller_plastic"    contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_1" material="medium_gloss_plastic" contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_2" material="polished_gold"        contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_3" material="polished_plastic"     contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_4" material="burnished_chrome"     contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_5" material="body_frame_plastic"   contype="0" conaffinity="0"/>\n'
        f'      <geom mesh="cf2_6" material="white"                contype="0" conaffinity="0"/>\n'
        f'{collision_block}'
        f'      <!-- IMU site for sensors -->\n'
        f'      <site name="{prefix}_imu" pos="0 0 0" size="0.001"/>\n'
        f'      <!-- Position probe for hoop scoring (non-colliding;\n'
        f'           queried via mj_geomDistance against hoop_score_tube). -->\n'
        f'      <geom name="{prefix}_probe" type="sphere" size="0.012" pos="0 0 0"\n'
        f'            contype="0" conaffinity="0" rgba="1 0 0 0"/>\n'
        f'    </body>'
    )

    sensors = (
        f'<gyro        name="{prefix}_gyro"        site="{prefix}_imu"/>',
        f'<velocimeter name="{prefix}_velocimeter" site="{prefix}_imu"/>',
        f'<framequat   name="{prefix}_framequat"   objtype="body" objname="{prefix}"/>',
        f'<framepos    name="{prefix}_framepos"    objtype="body" objname="{prefix}"/>',
    )

    return SceneFragment(worldbody=(body_xml,), sensors=sensors)
