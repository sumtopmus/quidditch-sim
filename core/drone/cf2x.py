"""cf2x quadrotor — physical constants and MJCF fragment factory.

The cf2x is a small (27 g) Crazyflie-style quadrotor.  Constants taken
verbatim from PyFlyt's cf2x.urdf and cf2x.yaml so the dynamics match the
original baseline.

`cf2x_fragment(prefix)` emits a self-contained MJCF fragment for one drone:
its body (with arms, motor discs, IMU site, scoring probe) plus its four
sensors (gyro, velocimeter, framequat, framepos), all namespaced by the
given prefix.  Multi-drone scenes = multiple fragments with distinct
prefixes (e.g. "red_0", "blue_0", ...).
"""

from __future__ import annotations

import math

from core.mjcf import SceneFragment


# ── cf2x physical constants ──────────────────────────────────────────────────
MASS: float = 0.027         # kg
IXX:  float = 1.4e-5        # kg⋅m²
IYY:  float = 1.4e-5
IZZ:  float = 2.17e-5
ARM:  float = 0.028         # m (motor distance from CoM along each diagonal)
THRUST_COEF: float = 3.16e-10  # N / RPM²
TORQUE_COEF: float = 7.94e-12  # N⋅m / RPM²
# max_rpm = sqrt(total_max_thrust / (4 × kF));  total_max_thrust = 2.0 N (cf2x.yaml)
MAX_RPM: float = math.sqrt(2.0 / (4.0 * THRUST_COEF))   # ≈ 39 775 RPM


def cf2x_fragment(prefix: str = "drone") -> SceneFragment:
    """A single cf2x drone (body + sensors) as a composable MJCF fragment.

    Args:
        prefix: namespace for every named element — body, freejoint, IMU
            site, scoring probe, and the four sensors.  Default "drone"
            preserves the historical single-drone names.  For multi-drone
            scenes pass distinct prefixes (e.g. "red_0", "blue_0").

    Returns:
        SceneFragment with one <body> in worldbody and four sensors
        (gyro, velocimeter, framequat, framepos), all prefixed.

    Notes:
        - The body's `<pos>` is hardcoded to `(0, 0, 0.03)` so the drone
          rests on the floor at MJCF compile time.  Per-episode start
          positions are written to qpos via `mj_resetData` in the World
          / Quadrotor reset path, not here.
        - The probe geom is `contype=0 conaffinity=0` and is intended for
          mj_geomDistance queries (e.g. against a hoop_score_tube).
    """
    body_xml = (
        f'<body name="{prefix}" pos="0 0 0.03">\n'
        f'      <freejoint name="{prefix}_root"/>\n'
        f'      <inertial mass="{MASS}" pos="0 0 0"\n'
        f'                diaginertia="{IXX} {IYY} {IZZ}"/>\n'
        f'      <!-- central frame -->\n'
        f'      <geom type="box" size="0.026 0.026 0.009"\n'
        f'            rgba="0.15 0.15 0.85 1" contype="0" conaffinity="0"/>\n'
        f'      <!-- four diagonal arms -->\n'
        f'      <geom type="capsule" size="0.0025"\n'
        f'            fromto=" 0.028 -0.028 0   0 0 0"\n'
        f'            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>\n'
        f'      <geom type="capsule" size="0.0025"\n'
        f'            fromto="-0.028  0.028 0   0 0 0"\n'
        f'            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>\n'
        f'      <geom type="capsule" size="0.0025"\n'
        f'            fromto=" 0.028  0.028 0   0 0 0"\n'
        f'            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>\n'
        f'      <geom type="capsule" size="0.0025"\n'
        f'            fromto="-0.028 -0.028 0   0 0 0"\n'
        f'            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>\n'
        f'      <!-- motor discs: yellow = CCW (m0,m1), cyan = CW (m2,m3) -->\n'
        f'      <geom type="cylinder" size="0.013 0.003"\n'
        f'            pos=" 0.028 -0.028 0.005"\n'
        f'            rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0"/>\n'
        f'      <geom type="cylinder" size="0.013 0.003"\n'
        f'            pos="-0.028  0.028 0.005"\n'
        f'            rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0"/>\n'
        f'      <geom type="cylinder" size="0.013 0.003"\n'
        f'            pos=" 0.028  0.028 0.005"\n'
        f'            rgba="0.1 0.85 0.85 0.9" contype="0" conaffinity="0"/>\n'
        f'      <geom type="cylinder" size="0.013 0.003"\n'
        f'            pos="-0.028 -0.028 0.005"\n'
        f'            rgba="0.1 0.85 0.85 0.9" contype="0" conaffinity="0"/>\n'
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
