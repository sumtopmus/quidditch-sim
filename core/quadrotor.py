"""MuJoCo-based single-drone simulator that mirrors the PyFlyt Aviary API.

Provides the same interface used by QuidditchSimpleEnv:
    aviary = Quadrotor(start_pos, start_orn, render=..., seed=...)
    aviary.set_mode(7)
    aviary.set_setpoint(0, setpoint)
    aviary.step()
    state = aviary.state(0)   # (4, 3) array
    aviary.disconnect()
    aviary.step_period         # property

Physics: MuJoCo at 240 Hz; position-setpoint control at 120 Hz.
The full scene (drone + hoop + arena) is built as a single MJCF string at
construction time, so no external asset files are needed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import mujoco

from core.position_controller import Mode7Controller


# ── camera config ────────────────────────────────────────────────────────────
# Eye and lookat in world coords (metres).  Used for both the offscreen "fixed"
# camera (videos) and the live viewer.  Override per-construction by passing
# `camera={"eye": (...), "lookat": (...)}` to Quadrotor; otherwise we try
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
        path = Path(__file__).resolve().parent.parent / "config" / "camera.toml"
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

# ── cf2x physical constants (from cf2x.urdf + cf2x.yaml) ────────────────────
MASS: float = 0.027         # kg
IXX: float  = 1.4e-5        # kg⋅m²
IYY: float  = 1.4e-5
IZZ: float  = 2.17e-5
ARM: float  = 0.028         # m (motor distance from CoM along each diagonal)
THRUST_COEF: float = 3.16e-10  # N / RPM²
TORQUE_COEF: float = 7.94e-12  # N⋅m / RPM²
# max_rpm = sqrt(total_max_thrust / (4 × kF)),  total_max_thrust = 2.0 N (cf2x.yaml)
MAX_RPM: float = math.sqrt(2.0 / (4.0 * THRUST_COEF))   # ≈ 39 775 RPM

# ── timing ───────────────────────────────────────────────────────────────────
PHYSICS_HZ: int = 240
CONTROL_HZ: int = 120
_DT_PHYSICS: float = 1.0 / PHYSICS_HZ          # 0.004167 s
_DT_CONTROL: float = 1.0 / CONTROL_HZ          # 0.008333 s
_PHYS_PER_CTRL: int = PHYSICS_HZ // CONTROL_HZ  # 2

# ── scoring trigger volume ───────────────────────────────────────────────────
# Thin cylinder along the hoop's outward normal.  Both the tube and the
# drone's probe geom are non-colliding (contype=0 conaffinity=0); we detect
# overlap with mj_geomDistance() in Quadrotor.drone_in_hoop instead of going
# through MuJoCo's contact solver.  This avoids the residual contact force
# that solimp="0 0 ..." doesn't quite eliminate (a 27g drone gets pinned at
# the tube's -x face by ~0.1 N of leftover impedance).
HOOP_SCORE_TUBE_HALF_LEN: float = 0.1  # ± m around the hoop plane


class Quadrotor:
    """Single-drone MuJoCo quadrotor sim.  Mimics PyFlyt Aviary for mode 7.

    Args:
        start_pos: (1, 3) initial xyz [m]
        start_orn: (1, 3) initial euler angles [roll, pitch, yaw] [rad]
        render:    open an interactive viewer window
        seed:      RNG seed
    """

    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        render: bool = False,
        seed: int | None = None,
        markers: list[tuple[tuple[float, float, float], str, float]] | None = None,
        camera: dict | None = None,
        include_hoop: bool = True,
        include_arena_wall: bool = True,
    ) -> None:
        """
        Args:
            markers: optional list of (pos, rgba, radius) tuples; each is rendered
                     as a non-colliding sphere geom in the scene.  Useful for
                     waypoints, target markers, etc.  Baked into the MJCF at
                     construction time — cannot be modified after.
            camera:  optional {"eye": (x,y,z), "lookat": (x,y,z)} dict.  Drives
                     both the offscreen "fixed" camera (videos) and the live
                     viewer pose.  Defaults to load_camera_config() →
                     config/camera.toml, with a hardcoded sideline fallback.
            include_hoop: if False, omit the goal hoop + pole.  Useful for
                     non-Quidditch scenes (e.g. waypoint flights).
            include_arena_wall: if False, omit the cylindrical arena wall.
        """
        self.start_pos = np.asarray(start_pos, dtype=np.float64)  # (1,3)
        self.start_orn = np.asarray(start_orn, dtype=np.float64)  # (1,3)
        self._render = render
        self.np_random = np.random.default_rng(seed)
        self._camera = camera if camera is not None else load_camera_config()

        # Build and load the complete scene (drone + optional hoop/arena/markers)
        xml = _build_scene_xml(
            markers=markers,
            camera=self._camera,
            include_hoop=include_hoop,
            include_arena_wall=include_arena_wall,
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data  = mujoco.MjData(self._model)

        # Cache body ID and sensor address offsets for fast access
        self._drone_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        self._gyro_adr = int(self._model.sensor_adr[
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        ])
        self._vel_adr  = int(self._model.sensor_adr[
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "velocimeter")
        ])
        self._quat_adr = int(self._model.sensor_adr[
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "framequat")
        ])
        self._pos_adr  = int(self._model.sensor_adr[
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "framepos")
        ])

        # Scoring geom IDs for mj_geomDistance().  None when include_hoop=False.
        g_probe = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "drone_probe")
        g_tube  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "hoop_score_tube")
        self._score_geoms: tuple[int, int] | None = (
            (g_probe, g_tube) if (g_probe >= 0 and g_tube >= 0) else None
        )
        # Reusable output buffer for mj_geomDistance(fromto[6]).  Allocated once
        # so the per-step query is allocation-free.
        self._geom_dist_fromto = np.zeros(6, dtype=np.float64)

        # Flight controller (mode 7 only)
        self._controller = Mode7Controller(_DT_CONTROL)
        self._setpoint = np.zeros(4, dtype=np.float64)
        self._pwm = np.full(4, 0.364)  # approx hover throttle

        # Viewer (opened lazily on first call with render=True)
        self._viewer   = None
        self._renderer = None

        self.reset()

    # ── Gymnasium-adjacent API ────────────────────────────────────────────────

    def reset(
        self,
        start_pos: np.ndarray | None = None,
        start_orn: np.ndarray | None = None,
    ) -> None:
        if start_pos is not None:
            self.start_pos = np.asarray(start_pos, dtype=np.float64)
        if start_orn is not None:
            self.start_orn = np.asarray(start_orn, dtype=np.float64)

        mujoco.mj_resetData(self._model, self._data)
        self._data.xfrc_applied[:] = 0.0

        pos = self.start_pos[0]  # (3,)
        orn = self.start_orn[0]  # (3,) — [roll, pitch, yaw]

        # Set initial position (qpos[0:3] for the freejoint)
        self._data.qpos[0:3] = pos

        # Convert euler [roll, pitch, yaw] → quaternion [w, x, y, z]
        roll, pitch, yaw = float(orn[0]), float(orn[1]), float(orn[2])
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        self._data.qpos[3] = cr * cp * cy + sr * sp * sy  # w
        self._data.qpos[4] = sr * cp * cy - cr * sp * sy  # x
        self._data.qpos[5] = cr * sp * cy + sr * cp * sy  # y
        self._data.qpos[6] = cr * cp * sy - sr * sp * cy  # z

        mujoco.mj_forward(self._model, self._data)

        self._controller.reset()
        # Initialize setpoint to actual start pose: hold position, initial yaw, 10 cm up
        self._setpoint[:] = [pos[0], pos[1], orn[2], max(pos[2], 0.1)]
        self._pwm[:] = 0.364

        if self._render and self._viewer is None:
            import mujoco.viewer as _mjv
            self._viewer = _mjv.launch_passive(self._model, self._data)
            az, el, dist, lookat = _viewer_params(
                self._camera["eye"], self._camera["lookat"]
            )
            self._viewer.cam.azimuth   = az
            self._viewer.cam.elevation = el
            self._viewer.cam.distance  = dist
            self._viewer.cam.lookat[:] = lookat

    def set_mode(self, mode: int) -> None:
        if mode != 7:
            raise ValueError(f"Quadrotor supports only mode 7 (position setpoint), got {mode}")
        self._controller.reset()

    def set_setpoint(self, idx: int, setpoint: np.ndarray) -> None:
        self._setpoint[:] = setpoint

    def step(self) -> None:
        """One control step (1/120 s) = 2 physics substeps (2 × 1/240 s)."""
        state = self.state(0)
        self._pwm = self._controller.step(state, self._setpoint)

        for _ in range(_PHYS_PER_CTRL):
            self._apply_motor_forces()
            mujoco.mj_step(self._model, self._data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def state(self, idx: int) -> np.ndarray:
        """Return drone state as (4, 3) float64 array.

        Row 0: angular velocity   — body frame  [rad/s]
        Row 1: angular position   — euler ZYX   [rad]
        Row 2: linear velocity    — body frame  [m/s]
        Row 3: linear position    — world frame [m]
        """
        sd = self._data.sensordata
        ang_vel = sd[self._gyro_adr  : self._gyro_adr  + 3].copy()
        lin_vel = sd[self._vel_adr   : self._vel_adr   + 3].copy()
        quat    = sd[self._quat_adr  : self._quat_adr  + 4].copy()
        lin_pos = sd[self._pos_adr   : self._pos_adr   + 3].copy()
        ang_pos = _quat_to_euler_zyx(quat)
        return np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

    def idle(self, active: bool = False) -> None:
        """Block until the user closes the viewer window.

        Lets you keep orbiting/panning the camera after a script's flight is
        finished.  No-op when running headless.

        Args:
            active: If True, keep stepping the controller against the last
                setpoint so the drone holds position.  If False (default),
                freeze physics and hold the last frame.
        """
        if self._viewer is None or not self._viewer.is_running():
            return

        mode = "hovering" if active else "frozen"
        print(f"[idle] viewer open ({mode}) — close the window to exit.")
        if active:
            while self._viewer.is_running():
                self.step()
                time.sleep(self.step_period)
        else:
            while self._viewer.is_running():
                time.sleep(0.05)

    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    @property
    def step_period(self) -> float:
        """Duration of one env step in seconds (= 1 / CONTROL_HZ)."""
        return _DT_CONTROL

    @property
    def drone_in_hoop(self) -> bool:
        """True iff the drone's probe geom currently overlaps the hoop scoring tube.

        Uses mj_geomDistance(), a pure geometric query — no contact-solver
        involvement and no force on either geom.  Both probe and tube are
        marked contype=0/conaffinity=0, so MuJoCo never tries to resolve a
        contact between them.  Returns False when the scene has no hoop
        (include_hoop=False).
        """
        if self._score_geoms is None:
            return False
        g_probe, g_tube = self._score_geoms
        # distmax=0 enables early-exit: returns 0 if separated, else the
        # actual (negative) penetration distance.  We treat dist < 0 as
        # "inside the tube".
        dist = mujoco.mj_geomDistance(
            self._model, self._data, g_probe, g_tube, 0.0, self._geom_dist_fromto
        )
        return bool(dist < 0.0)

    # ── rendering helpers ─────────────────────────────────────────────────────

    def get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        return self._renderer

    def render_frame(self, width: int, height: int) -> np.ndarray:
        """Render to an RGB (H×W×3) uint8 array using the scene's fixed camera."""
        renderer = self.get_renderer(width, height)
        renderer.update_scene(self._data, camera="fixed")
        rgba = renderer.render()          # (H, W, 4) uint8 — mujoco 3.x returns RGBA
        return rgba[:, :, :3]

    # ── private physics ───────────────────────────────────────────────────────

    def _apply_motor_forces(self) -> None:
        """Convert PWM to body-frame wrench and apply via xfrc_applied."""
        rpm  = self._pwm * MAX_RPM
        rpm2 = rpm * rpm
        f    = THRUST_COEF * rpm2    # per-motor thrust [N], shape (4,)

        # Total thrust along body +z
        F_z = float(np.sum(f))

        # Lever-arm torques (body frame).  Motor positions (ENU: x=fwd, y=left):
        #   m0=(+L,-L)  m1=(-L,+L)  m2=(+L,+L)  m3=(-L,-L)
        # τx = Σ y_i · f_i  =  L(-f0 + f1 + f2 - f3)
        # τy = Σ -x_i · f_i = L(-f0 + f1 - f2 + f3)
        L  = ARM
        kM = TORQUE_COEF
        tau_x = L  * (-f[0] + f[1] + f[2] - f[3])
        tau_y = L  * (-f[0] + f[1] - f[2] + f[3])
        # Reaction torques from motor spin (m0,m1 CW→negative; m2,m3 CCW→positive)
        tau_z = kM * (-rpm2[0] - rpm2[1] + rpm2[2] + rpm2[3])

        # Rotate body-frame wrench to world frame
        R = self._data.xmat[self._drone_id].reshape(3, 3)
        self._data.xfrc_applied[self._drone_id, :3] = R @ np.array([0.0, 0.0, F_z])
        self._data.xfrc_applied[self._drone_id, 3:] = R @ np.array([tau_x, tau_y, tau_z])


# ── quaternion → ZYX Euler ────────────────────────────────────────────────────

def _quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] quaternion → [roll, pitch, yaw] ZYX Euler (radians)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Roll
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # Pitch (clamped to avoid NaN)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    # Yaw
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float64)


# ── MJCF scene generation ─────────────────────────────────────────────────────

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


def _hoop_mesh_asset_xml(major_r: float, tube_r: float = 0.012) -> str:
    """`<mesh>` + `<material>` block for the polished hoop ring."""
    v, n, f = _torus_mesh_data(major_r, tube_r, n_major=64, n_minor=12)
    return (
        f'<mesh name="hoop_ring" vertex="{v}" normal="{n}" face="{f}"/>\n'
        f'    <material name="hoop_metal" rgba="1.0 0.45 0.0 1" '
        f'specular="0.5" shininess="0.5" reflectance="0.15"/>'
    )


def _hoop_geom_xml(cx: float, cy: float, cz: float) -> str:
    """Single mesh geom referencing the hoop_ring mesh defined in <asset>."""
    return (
        f'<geom type="mesh" mesh="hoop_ring" '
        f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" material="hoop_metal" '
        f'contype="0" conaffinity="0"/>'
    )


def _pole_geom(cx: float, cy: float, cz: float, ring_r: float) -> str:
    """Thin vertical cylinder from ground to the bottom of the hoop."""
    base_z = cz - ring_r
    return (
        f'<geom type="cylinder" size="0.005" '
        f'fromto="{cx:.4f} {cy:.4f} 0  {cx:.4f} {cy:.4f} {base_z:.4f}" '
        f'rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>'
    )


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


def _arena_wall_mesh_asset_xml(radius: float, height: float) -> str:
    """`<mesh>` + `<material>` block for the translucent arena wall."""
    v, n, f = _arena_wall_mesh_data(radius, height, thickness=0.016, n=64)
    return (
        f'<mesh name="arena_wall" vertex="{v}" normal="{n}" face="{f}"/>\n'
        f'    <material name="arena_glass" rgba="0.6 0.85 1.0 0.35" '
        f'specular="0.3" shininess="0.5"/>'
    )


def _arena_wall_geom_xml() -> str:
    """Single mesh geom referencing the arena_wall mesh defined in <asset>."""
    return (
        '<geom type="mesh" mesh="arena_wall" material="arena_glass" '
        'contype="0" conaffinity="0"/>'
    )


def _markers_xml(markers: list[tuple] | None) -> str:
    """Render a list of (pos, rgba_str, radius) tuples as non-colliding spheres."""
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


def _build_scene_xml(
    hoop_center: tuple[float, float, float] = (2.0, 0.0, 2.0),
    hoop_radius: float = 0.25,           # ring major radius (= HOOP_DIAMETER / 2)
    arena_radius: float = 3.0,
    markers: list[tuple] | None = None,
    camera: dict | None = None,
    include_hoop: bool = True,
    include_arena_wall: bool = True,
) -> str:
    """Return the complete MuJoCo XML for the Quidditch scene."""
    hx, hy, hz = hoop_center
    hoop_xml  = (
        _hoop_geom_xml(hx, hy, hz) + "\n      " + _pole_geom(hx, hy, hz, hoop_radius)
        if include_hoop else ""
    )
    hoop_assets_xml = _hoop_mesh_asset_xml(hoop_radius) if include_hoop else ""
    # Invisible cylinder along the hoop normal (+x) — non-colliding; acts
    # purely as a geometric query target for mj_geomDistance.  rgba alpha=0
    # keeps it out of every render; bump alpha to e.g. 0.08 for debugging.
    score_tube_xml = (
        f'<geom name="hoop_score_tube" type="cylinder" '
        f'size="{hoop_radius:.4f} {HOOP_SCORE_TUBE_HALF_LEN:.4f}" '
        f'pos="{hx:.4f} {hy:.4f} {hz:.4f}" euler="0 1.5708 0" '
        f'contype="0" conaffinity="0" '
        f'rgba="0.1 1.0 0.3 0"/>'
        if include_hoop else ""
    )
    arena_xml = _arena_wall_geom_xml() if include_arena_wall else ""
    arena_assets_xml = (
        _arena_wall_mesh_asset_xml(arena_radius, 4.5) if include_arena_wall else ""
    )
    marker_xml = _markers_xml(markers)

    cam = camera if camera is not None else _FALLBACK_CAMERA
    cam_pos, cam_xyaxes = _camera_xyaxes(cam["eye"], cam["lookat"])

    return f"""
<mujoco model="quidditch">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="{_DT_PHYSICS:.8f}"/>

  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <global offwidth="1920" offheight="1080"/>
    <quality offsamples="8" shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.12"
             width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker"
             rgb1="0.28 0.32 0.36" rgb2="0.20 0.24 0.28"
             width="300" height="300" mark="edge" markrgb="0.4 0.4 0.4"/>
    <material name="grid" texture="grid" texuniform="true"
              texrepeat="5 5" reflectance="0.08"/>
    {hoop_assets_xml}
    {arena_assets_xml}
  </asset>

  <worldbody>
    <light name="sun"  pos="0 0 8"  dir="0 0 -1"  diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="-4 4 5" dir="1 -1 -1"  diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <geom name="floor" type="plane" size="6 6 0.05"
          material="grid" contype="1" conaffinity="1"/>

    <!-- ── Quadrotor (cf2x) ───────────────────────────────────────────── -->
    <body name="drone" pos="0 0 0.03">
      <freejoint name="root"/>
      <inertial mass="{MASS}" pos="0 0 0"
                diaginertia="{IXX} {IYY} {IZZ}"/>
      <!-- central frame -->
      <geom type="box" size="0.026 0.026 0.009"
            rgba="0.15 0.15 0.85 1" contype="0" conaffinity="0"/>
      <!-- four diagonal arms -->
      <geom type="capsule" size="0.0025"
            fromto=" 0.028 -0.028 0   0 0 0"
            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.0025"
            fromto="-0.028  0.028 0   0 0 0"
            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.0025"
            fromto=" 0.028  0.028 0   0 0 0"
            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.0025"
            fromto="-0.028 -0.028 0   0 0 0"
            rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
      <!-- motor discs: yellow = CCW (m0,m1), cyan = CW (m2,m3) -->
      <geom type="cylinder" size="0.013 0.003"
            pos=" 0.028 -0.028 0.005"
            rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.013 0.003"
            pos="-0.028  0.028 0.005"
            rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.013 0.003"
            pos=" 0.028  0.028 0.005"
            rgba="0.1 0.85 0.85 0.9" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.013 0.003"
            pos="-0.028 -0.028 0.005"
            rgba="0.1 0.85 0.85 0.9" contype="0" conaffinity="0"/>
      <!-- IMU site for sensors -->
      <site name="imu" pos="0 0 0" size="0.001"/>
      <!-- Position probe for hoop scoring.  Non-colliding (contype=0/0);
           mj_geomDistance(drone_probe, hoop_score_tube) detects overlap
           without involving the contact solver. -->
      <geom name="drone_probe" type="sphere" size="0.012" pos="0 0 0"
            contype="0" conaffinity="0" rgba="1 0 0 0"/>
    </body>

    <!-- ── Hoop (32-segment torus approximation + scoring trigger volume) -->
    <body name="hoop" pos="0 0 0">
      {hoop_xml}
      {score_tube_xml}
    </body>

    <!-- ── Arena boundary wall ────────────────────────────────────────── -->
    <body name="arena" pos="0 0 0">
      {arena_xml}
    </body>

    <!-- ── Optional markers (waypoints, debug spheres, …) ─────────────── -->
    {marker_xml}

    <!-- Fixed camera (config/camera.toml): eye={cam["eye"]} looking at {cam["lookat"]} -->
    <camera name="fixed" pos="{cam_pos}" xyaxes="{cam_xyaxes}"/>
  </worldbody>

  <sensor>
    <gyro        name="gyro"        site="imu"/>
    <velocimeter name="velocimeter" site="imu"/>
    <framequat   name="framequat"   objtype="body" objname="drone"/>
    <framepos    name="framepos"    objtype="body" objname="drone"/>
  </sensor>
</mujoco>
"""
