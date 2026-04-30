"""Cascaded PID position controller for a QuadX drone (mode 7: position setpoint).

Replicates PyFlyt's cf2x mode-7 controller exactly, using the same PID gains
from cf2x.yaml, the same 4-level cascade, and the same motor-mixing matrix.

Control hierarchy (mode 7):
    [x, y, yaw, z]  →  lin_pos_PID (xy)
                     →  world→body rotation of xy velocity
                     →  lin_vel_PID (xy)
                     →  roll/pitch swap
                     →  ang_pos_PID (roll, pitch, yaw)
                     →  ang_vel_PID (roll, pitch, yaw)
    z  →  z_pos_PID  →  z_vel_PID  →  thrust [0,1]
    [roll_torque, pitch_torque, yaw_torque, thrust]  →  motor_map  →  PWM[4]
"""

from __future__ import annotations

import math
import numpy as np


class _PID:
    """Discrete PID matching PyFlyt's numba-jitted implementation."""

    def __init__(
        self,
        kp: list | np.ndarray,
        ki: list | np.ndarray,
        kd: list | np.ndarray,
        limits: list | np.ndarray,
        period: float,
    ) -> None:
        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.limits = np.asarray(limits, dtype=np.float64)
        self.period = float(period)
        self._integral = np.zeros_like(self.kp)
        self._prev_error = np.zeros_like(self.kp)

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._prev_error[:] = 0.0

    def step(self, state: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        error = np.asarray(setpoint, dtype=np.float64) - np.asarray(state, dtype=np.float64)
        proportional = self.kp * error
        self._integral = np.clip(
            self._integral + self.ki * error * self.period,
            -self.limits, self.limits,
        )
        derivative = self.kd * (error - self._prev_error) / self.period
        self._prev_error = error
        return np.clip(proportional + self._integral + derivative, -self.limits, self.limits)


# Motor allocation matrix — maps [roll, pitch, yaw, thrust] cmd to 4 PWMs.
# From PyFlyt QuadX (PX4 QuadX layout, ENU body frame):
#   m0 front-right (CCW): (-roll, -pitch, -yaw, +thrust)
#   m1 back-left   (CCW): (+roll, +pitch, -yaw, +thrust)
#   m2 front-left  (CW):  (+roll, -pitch, +yaw, +thrust)
#   m3 back-right  (CW):  (-roll, +pitch, +yaw, +thrust)
_MOTOR_MAP = np.array([
    [-1.0, -1.0, -1.0, +1.0],
    [+1.0, +1.0, -1.0, +1.0],
    [+1.0, -1.0, +1.0, +1.0],
    [-1.0, +1.0, +1.0, +1.0],
], dtype=np.float64)


class Mode7Controller:
    """Position-setpoint (mode 7) controller for the cf2x quadrotor.

    Setpoint: [x, y, yaw, z]  (world-frame position + heading)
    State:    (4, 3) array  [ang_vel_body, ang_pos_euler, lin_vel_body, lin_pos_world]
    Output:   PWM (4,) in [0.05, 1.0]

    PID gains are taken verbatim from cf2x.yaml.
    """

    def __init__(self, control_period: float) -> None:
        dt = control_period

        # cf2x.yaml gains — do NOT change without retesting.
        self._ang_vel  = _PID([4e-2,4e-2,8e-2], [5e-7,5e-7,2.7e-4], [1e-4,1e-4,0], [1,1,1], dt)
        self._ang_pos  = _PID([2.0, 2.0, 2.0],  [0,0,0],             [0,0,0],       [3,3,3], dt)
        self._lin_vel  = _PID([0.8, 0.8],        [0.3, 0.3],          [0.5, 0.5],    [0.4, 0.4], dt)
        self._lin_pos  = _PID([1.0, 1.0],        [0, 0],              [0, 0],        [2.0, 2.0], dt)
        self._z_pos    = _PID([1.0],             [0],                 [0],           [1.0], dt)
        self._z_vel    = _PID([2.0],             [0.5],               [0.05],        [1.0], dt)

    def reset(self) -> None:
        for pid in (self._ang_vel, self._ang_pos, self._lin_vel,
                    self._lin_pos, self._z_pos, self._z_vel):
            pid.reset()

    def step(self, state: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        """Compute motor PWM commands.

        Args:
            state:    (4, 3) — [ang_vel_body, ang_pos_euler, lin_vel_body, lin_pos_world]
            setpoint: (4,)   — [x, y, yaw, z]

        Returns:
            pwm: (4,) motor throttles in [0.05, 1.0]
        """
        ang_vel = state[0]  # body-frame
        ang_pos = state[1]  # euler [roll, pitch, yaw]
        lin_vel = state[2]  # body-frame
        lin_pos = state[3]  # world-frame

        # ── xy position → desired xy velocity (world frame) ──────────────
        a_out = np.zeros(3, dtype=np.float64)
        a_out[:2] = self._lin_pos.step(lin_pos[:2], setpoint[:2])
        a_out[2] = setpoint[2]  # yaw setpoint passes through

        # Rotate velocity command to body frame (world→body = Rz(yaw).T)
        yaw = float(ang_pos[2])
        c, s = math.cos(yaw), math.sin(yaw)
        # world→body for xy: [[c, s], [-s, c]]
        a_out[:2] = np.array([c * a_out[0] + s * a_out[1],
                               -s * a_out[0] + c * a_out[1]])

        # ── xy velocity → roll/pitch command ─────────────────────────────
        a_out[:2] = self._lin_vel.step(lin_vel[:2], a_out[:2])

        # Velocity error → attitude: swap and negate (PyFlyt convention)
        a_out = np.array([-a_out[1], a_out[0], a_out[2]])

        # ── attitude → angular velocity command ──────────────────────────
        a_out = self._ang_pos.step(ang_pos, a_out)

        # ── angular velocity → normalized torque ─────────────────────────
        a_out = self._ang_vel.step(ang_vel, a_out)

        # ── z: position → velocity → thrust ─────────────────────────────
        z_out = self._z_pos.step(lin_pos[2:3], setpoint[3:4])
        z_out = self._z_vel.step(lin_vel[2:3], z_out)
        z_out = np.clip(z_out, 0.0, 1.0)

        # ── motor mixing ─────────────────────────────────────────────────
        cmd = np.array([a_out[0], a_out[1], a_out[2], z_out[0]])
        pwm = _MOTOR_MAP @ cmd

        # Saturation: preserve the ratio between motors while clamping to [0.05, 1.0]
        hi, lo = float(np.max(pwm)), float(np.min(pwm))
        if hi != lo:
            pwm_max = min(hi, 1.0)
            pwm_min = max(lo, 0.05)
            add = (pwm_min - lo) / (pwm_max - lo) * (pwm_max - pwm)
            sub = (hi - pwm_max) / (hi - pwm_min) * (pwm - pwm_min)
            pwm = pwm + add - sub

        return np.clip(pwm, 0.05, 1.0)
