"""Per-drone view bound to a `World`.

A `Quadrotor` does not own MjModel/MjData — it borrows them from the World
and resolves its body and sensors by namespace prefix.  Multiple
`Quadrotor` views can share one `World` (multi-drone) as long as they use
distinct prefixes.

Public surface (per drone):
    Quadrotor(world, prefix="drone")
    Quadrotor.set_mode(7)
    Quadrotor.set_setpoint(setpoint)         # NO idx parameter
    Quadrotor.state() -> (4, 3) ndarray      # NO idx parameter
    Quadrotor.set_start(start_pos, start_orn)
    Quadrotor.step_period -> float

Convenience for single-drone callers (demos, env):
    Quadrotor.standalone(start_pos, start_orn, *, render=False, camera=None,
                         seed=None, markers=None, extra_fragments=()) -> Quadrotor

Façade methods (route to the bound World) so existing single-drone callers
keep working without learning about the World object:
    Quadrotor.step()             -> world.step()
    Quadrotor.disconnect()       -> world.disconnect()
    Quadrotor.idle(active=False) -> world.idle(active)
    Quadrotor.render_frame(w, h) -> world.render_frame(w, h)
    Quadrotor.get_renderer(w, h) -> world.get_renderer(w, h)

`load_camera_config` is re-exported from this module for back-compat with
demo/camera_test.py and other historical importers.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import mujoco

from core.position_controller import Mode7Controller
from core.mjcf import SceneFragment, load_camera_config
from core.mjcf.meshes import _markers_xml
from core.drone.cf2x import (
    MASS,
    IXX,
    IYY,
    IZZ,
    ARM,
    THRUST_COEF,
    TORQUE_COEF,
    MAX_RPM,
    cf2x_fragment,
)
from core.world import (
    World,
    PHYSICS_HZ,
    CONTROL_HZ,
    _DT_PHYSICS,
    _DT_CONTROL,
    _PHYS_PER_CTRL,
)


# Re-export so `from core.quadrotor import load_camera_config` keeps working.
__all__ = [
    "Quadrotor",
    "load_camera_config",
    "PHYSICS_HZ",
    "CONTROL_HZ",
]


class Quadrotor:
    """Per-drone view bound to a `World`.

    Holds the drone's setpoint, controller, PWM state, and cached references
    into ``world.model`` / ``world.data``.  Resolves every body / sensor /
    geom by ``f"{prefix}_..."``; for the historical single-drone case the
    prefix is ``"drone"`` so the body is named ``"drone"``, sensors
    ``"drone_gyro"`` etc.
    """

    def __init__(self, world: World, prefix: str = "drone") -> None:
        self._world = world
        self._prefix = prefix

        # Register with the world so World.step()/reset() iterate over us.
        world.drones.append(self)

        model = world.model

        # Cache body ID and sensor address offsets for fast access.  The body
        # itself is named just `prefix`; sensors are `f"{prefix}_<role>"`.
        self._drone_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix
        )
        if self._drone_id < 0:
            raise ValueError(
                f"Quadrotor: no body named {prefix!r} in the world MJCF "
                f"(did you forget cf2x_fragment(prefix={prefix!r})?)"
            )
        self._gyro_adr = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_gyro")
        ])
        self._vel_adr  = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_velocimeter")
        ])
        self._quat_adr = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_framequat")
        ])
        self._pos_adr  = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_framepos")
        ])

        # qpos offset for this drone's freejoint (7 dofs: xyz + quat).
        # The freejoint is named f"{prefix}_root" in cf2x_fragment.
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_root"
        )
        self._qpos_adr = int(model.jnt_qposadr[joint_id])

        # Flight controller (mode 7 only) + setpoint + PWM state.
        self._controller = Mode7Controller(_DT_CONTROL)
        self._setpoint = np.zeros(4, dtype=np.float64)
        self._pwm = np.full(4, 0.364)  # approx hover throttle

        # Initial pose — written into qpos by World.reset() via _reset_qpos().
        # Defaults to origin + identity orientation; callers set via set_start().
        self.start_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.start_orn = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: int) -> None:
        if mode != 7:
            raise ValueError(
                f"Quadrotor supports only mode 7 (position setpoint), got {mode}"
            )
        self._controller.reset()

    def set_setpoint(self, setpoint: np.ndarray) -> None:
        """Update the position setpoint (4,) = (x, y, yaw, z)."""
        self._setpoint[:] = setpoint

    def set_start(self, start_pos: np.ndarray, start_orn: np.ndarray) -> None:
        """Stash a new start pose; applied on the next World.reset()."""
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.start_orn = np.asarray(start_orn, dtype=np.float64)

    def state(self) -> np.ndarray:
        """Return drone state as (4, 3) float64 array.

        Row 0: angular velocity   — body frame  [rad/s]
        Row 1: angular position   — euler ZYX   [rad]
        Row 2: linear velocity    — body frame  [m/s]
        Row 3: linear position    — world frame [m]
        """
        sd = self._world.data.sensordata
        ang_vel = sd[self._gyro_adr  : self._gyro_adr  + 3].copy()
        lin_vel = sd[self._vel_adr   : self._vel_adr   + 3].copy()
        quat    = sd[self._quat_adr  : self._quat_adr  + 4].copy()
        lin_pos = sd[self._pos_adr   : self._pos_adr   + 3].copy()
        ang_pos = _quat_to_euler_zyx(quat)
        return np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

    @property
    def step_period(self) -> float:
        """Duration of one control step in seconds (delegates to World)."""
        return self._world.step_period

    # ── façade: route to bound World so single-drone callers don't change ────

    def step(self) -> None:
        self._world.step()

    def disconnect(self) -> None:
        self._world.disconnect()

    def idle(self, active: bool = False) -> None:
        self._world.idle(active)

    def render_frame(self, width: int, height: int) -> np.ndarray:
        return self._world.render_frame(width, height)

    def get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        return self._world.get_renderer(width, height)

    # ── classmethod: one-call standalone setup for demos and the env ─────────

    @classmethod
    def standalone(
        cls,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        *,
        render: bool = False,
        camera: dict | None = None,
        seed: int | None = None,
        markers: list[tuple] | None = None,
        extra_fragments: Iterable[SceneFragment] = (),
    ) -> "Quadrotor":
        """Build a one-drone World and return a `Quadrotor` view onto it.

        Args:
            start_pos: (1, 3) initial xyz [m]
            start_orn: (1, 3) initial euler [roll, pitch, yaw] [rad]
            render:    open the interactive viewer
            camera:    optional ``{"eye": ..., "lookat": ...}``
            seed:      RNG seed
            markers:   list of (pos, rgba_str, radius) sphere markers
            extra_fragments: additional `SceneFragment`s (hoop, arena wall, ...)

        Returns:
            A ready-to-use `Quadrotor`; the underlying `World` is reachable
            via ``quad._world`` if needed but not exposed publicly.
        """
        fragments: list[SceneFragment] = [cf2x_fragment(prefix="drone")]
        fragments.extend(extra_fragments)
        if markers:
            fragments.append(SceneFragment(worldbody=(_markers_xml(markers),)))

        world = World(fragments, camera=camera, render=render, seed=seed)
        quad = cls(world, prefix="drone")
        quad.set_start(start_pos, start_orn)
        world.reset()
        return quad

    # ── World hooks (called by World, not by user code) ──────────────────────

    def _reset_qpos(self) -> None:
        """Write start_pos + start_orn into the world's qpos for our freejoint."""
        data = self._world.data
        adr = self._qpos_adr

        pos = self.start_pos[0]  # (3,)
        orn = self.start_orn[0]  # (3,) — [roll, pitch, yaw]

        data.qpos[adr + 0 : adr + 3] = pos

        # Euler [roll, pitch, yaw] → quaternion [w, x, y, z]
        roll, pitch, yaw = float(orn[0]), float(orn[1]), float(orn[2])
        cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
        data.qpos[adr + 3] = cr * cp * cy + sr * sp * sy  # w
        data.qpos[adr + 4] = sr * cp * cy - cr * sp * sy  # x
        data.qpos[adr + 5] = cr * sp * cy + sr * cp * sy  # y
        data.qpos[adr + 6] = cr * cp * sy - sr * sp * cy  # z

    def _reset_controller(self) -> None:
        """Reset controller + initialise setpoint to start pose, post-mj_forward."""
        self._controller.reset()
        pos = self.start_pos[0]
        orn = self.start_orn[0]
        # Hold position, initial yaw, ≥10 cm up.
        self._setpoint[:] = [pos[0], pos[1], orn[2], max(pos[2], 0.1)]
        self._pwm[:] = 0.364

    def _compute_control(self) -> None:
        """Run one controller step → updated PWM (called once per control step)."""
        state = self.state()
        self._pwm = self._controller.step(state, self._setpoint)

    def _apply_control(self) -> None:
        """Convert PWM to body-frame wrench and write into xfrc_applied.

        Called every physics substep (``_PHYS_PER_CTRL`` times per control
        step) so the world-frame wrench tracks the current rotation matrix.
        """
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
        data = self._world.data
        R = data.xmat[self._drone_id].reshape(3, 3)
        data.xfrc_applied[self._drone_id, :3] = R @ np.array([0.0, 0.0, F_z])
        data.xfrc_applied[self._drone_id, 3:] = R @ np.array([tau_x, tau_y, tau_z])


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
