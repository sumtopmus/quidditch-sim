"""QuidditchSimpleEnv — single-drone, single-hoop scoring task.

Game rules (Phase 1 milestone):
  - Circular arena, 3 m radius (6 m diameter).
  - One vertical hoop (0.5 m diameter) on a pole, at (2, 0, 2) m — i.e.
    2 m from arena center (1 m from the wall), 2 m height, hoop plane perpendicular to the ground
    and facing the arena center.
  - Drone starts at the arena center on the ground.
  - Score: drone crosses the hoop plane from the inside (arena-center side)
    to the outside while its y/z position is within the hoop aperture.
  - Episode ends on score, crash, out-of-bounds, or 2-minute timeout.

Observation (16 floats):
    [0:3]   angular velocity  — body frame, rad/s
    [3:6]   attitude          — ground frame euler, rad
    [6:9]   linear velocity   — body frame, m/s
    [9:12]  position          — ground frame, m
    [12:15] unit vector from drone to hoop center
    [15]    signed distance to hoop plane / ARENA_RADIUS  (negative = inside)

Action (4 floats, normalized to [-1, 1]):
    Mapped to a delta applied to the current position setpoint each step:
        [0] dx   ∈ [−0.2, +0.2] m
        [1] dy   ∈ [−0.2, +0.2] m
        [2] dyaw ∈ [−0.5, +0.5] rad
        [3] dz   ∈ [−0.1, +0.1] m
    Setpoint is clamped to arena bounds and safe altitude each step.

Reward (per step):
    −(dist_to_hoop / ARENA_RADIUS) × 0.01      distance-shaping term
    +10.0  on scoring through the hoop          terminal bonus
    −2.0   on crash or out-of-bounds            terminal penalty

Flight mode: PyFlyt mode 7 — position setpoint [x, y, yaw, z].
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from PyFlyt.core import Aviary


@contextlib.contextmanager
def _silence_c_stdout():
    """Redirect C-level stdout to /dev/null for the duration of the block.

    Python's sys.stdout redirection cannot suppress printf() calls from C
    extensions (e.g. PyBullet's 'argv[0]=' startup message).  We must dup2
    the real file descriptor instead.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

ARENA_RADIUS: float = 3.0  # m (6 m diameter)

# Hoop geometry: vertical ring at x=2, y=0, z=2.
# 2 m from arena center, 1 m from the wall, 2 m height.
# "Outward normal" points from arena center through the hoop (away from center).
HOOP_CENTER = np.array([2.0, 0.0, 2.0], dtype=np.float64)
HOOP_OUTWARD_NORMAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
HOOP_DIAMETER: float = 0.5  # m (50 cm)
HOOP_RADIUS: float = HOOP_DIAMETER / 2.0

# Drone initial state
DRONE_START_POS = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
DRONE_START_ORN = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

# Timing
EPISODE_SECONDS: float = 120.0  # 2-minute episodes
PHYSICS_HZ: int = 240  # PyFlyt default (do not change)

# Action scaling: each normalized action component [-1, 1] maps to these deltas
ACTION_SCALE = np.array([0.2, 0.2, 0.5, 0.1], dtype=np.float32)  # yaw (rad) not scaled

# Reward
SCORE_REWARD: float = 10.0
CRASH_PENALTY: float = -2.0
DIST_REWARD_SCALE: float = 0.01  # multiplied by −(dist/ARENA_RADIUS) per step
TAKEOFF_GRACE_STEPS: int = 30  # skip crash check while drone lifts off from ground

# ---------------------------------------------------------------------------
# Hoop visualization
# ---------------------------------------------------------------------------
HOOP_SEGS: int = 24  # sphere segments in the ring visual
HOOP_SEG_RADIUS: float = 0.004  # m — radius of each visual sphere
POLE_RADIUS: float = 0.005  # m — radius of the support pole
HOOP_RGBA = (1.0, 0.45, 0.0, 1.0)  # orange
POLE_RGBA = (0.55, 0.55, 0.55, 1.0)

# ---------------------------------------------------------------------------
# Arena boundary visualization — ring of magenta vertical pillars
# ---------------------------------------------------------------------------
ARENA_WALL_SEGS: int = 60  # number of pillars around the perimeter
ARENA_WALL_HEIGHT: float = 4.0  # m — taller than the hoop (2 m) to frame the volume
ARENA_WALL_PILLAR_R: float = 0.025  # m — radius of each pillar
ARENA_WALL_RGBA = (0.95, 0.1, 0.95, 0.45)  # semi-transparent magenta

# ---------------------------------------------------------------------------
# rgb_array camera — elevated side view covering the full start→hoop path
# ---------------------------------------------------------------------------
VIDEO_WIDTH: int = 640
VIDEO_HEIGHT: int = 480
VIDEO_CAM_EYE = (1.0, -4.0, 3.0)  # to the side and above
VIDEO_CAM_TARGET = (1.0, 0.0, 1.5)  # midpoint of arena at mid-hoop height
VIDEO_CAM_UP = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class QuidditchSimpleEnv(gym.Env):
    """Gymnasium environment: fly a single QuadX through one vertical hoop."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # 16-dim flat obs vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        # 4-dim normalized action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._aviary: Aviary | None = None
        # Current position setpoint: [x, y, yaw, z]
        self._setpoint = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0
        self._max_steps: int = 0  # set after first Aviary reset
        self._prev_signed_dist: float = 0.0
        # Drone starts on the ground; skip the crash check for the first
        # TAKEOFF_GRACE_STEPS steps so it has time to climb above the threshold.
        self._takeoff_grace: int = 0

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._aviary is None:
            # First call — create the Aviary (opens the PyBullet physics server).
            # PyBullet prints "argv[0]=..." at C level on connect; suppress it.
            with _silence_c_stdout():
                self._aviary = Aviary(
                    start_pos=DRONE_START_POS,
                    start_orn=DRONE_START_ORN,
                    render=(self.render_mode == "human"),
                    drone_type="quadx",
                    seed=seed,
                )
        else:
            # Subsequent calls — reset physics and re-spawn drone in-place
            self._aviary.reset()

        self._aviary.set_mode(7)  # position setpoint: [x, y, yaw, z]

        # Initialize setpoint at a safe hover height so the drone takes off
        self._setpoint = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32)
        self._aviary.set_setpoint(0, self._setpoint)

        # Derive max steps from the aviary's actual step period
        self._max_steps = int(EPISODE_SECONDS / self._aviary.step_period)

        self._step_count = 0
        self._takeoff_grace = TAKEOFF_GRACE_STEPS
        drone_pos = self._drone_pos()
        self._prev_signed_dist = self._signed_dist(drone_pos)

        if self.render_mode in ("human", "rgb_array"):
            self._draw_hoop()
            self._draw_arena()

        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._aviary is not None, "Call reset() before step()."

        # Apply delta, clamp to safe limits
        delta = np.asarray(action, dtype=np.float32) * ACTION_SCALE
        self._setpoint += delta
        # x / y stay inside arena
        self._setpoint[0] = np.clip(self._setpoint[0], -ARENA_RADIUS, ARENA_RADIUS)
        self._setpoint[1] = np.clip(self._setpoint[1], -ARENA_RADIUS, ARENA_RADIUS)
        # yaw wraps to (-π, π]
        self._setpoint[2] = (self._setpoint[2] + np.pi) % (2 * np.pi) - np.pi
        # altitude: stay above 0.01 m, cap at 4 m
        self._setpoint[3] = np.clip(self._setpoint[3], 0.01, 4.0)

        self._aviary.set_setpoint(0, self._setpoint)
        self._aviary.step()
        self._step_count += 1

        drone_pos = self._drone_pos()
        curr_signed_dist = self._signed_dist(drone_pos)

        # --- scoring ---
        scored = self._detect_score(drone_pos, curr_signed_dist)
        self._prev_signed_dist = curr_signed_dist

        # --- termination ---
        out_of_bounds = float(np.linalg.norm(drone_pos[:2])) > ARENA_RADIUS
        if self._takeoff_grace > 0:
            self._takeoff_grace -= 1
            crashed = False
        else:
            crashed = float(drone_pos[2]) < 0.05  # returned to ground after takeoff
        timeout = self._step_count >= self._max_steps

        terminated = scored or out_of_bounds or crashed
        truncated = timeout and not terminated

        # --- reward ---
        dist = float(np.linalg.norm(drone_pos - HOOP_CENTER))
        reward = -(dist / ARENA_RADIUS) * DIST_REWARD_SCALE
        if scored:
            reward += SCORE_REWARD
        elif out_of_bounds or crashed:
            reward += CRASH_PENALTY

        info = {
            "scored": scored,
            "dist_to_hoop": dist,
            "step": self._step_count,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array" or self._aviary is None:
            return None
        view = self._aviary.computeViewMatrix(
            VIDEO_CAM_EYE, VIDEO_CAM_TARGET, VIDEO_CAM_UP
        )
        proj = self._aviary.computeProjectionMatrixFOV(
            fov=60,
            aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
            nearVal=0.01,
            farVal=20.0,
        )
        _, _, rgba, _, _ = self._aviary.getCameraImage(
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,  # CPU renderer, works in DIRECT mode
        )
        return np.array(rgba, dtype=np.uint8)[:, :, :3]

    def close(self) -> None:
        if self._aviary is not None:
            self._aviary.disconnect()
            self._aviary = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _drone_pos(self) -> np.ndarray:
        """Return the drone's ground-frame position as a (3,) float64 array."""
        return self._aviary.state(0)[3].copy()  # state[3] = lin_pos

    def _obs(self) -> np.ndarray:
        """Build the 16-dim observation vector."""
        state = self._aviary.state(0)  # (4, 3) array
        ang_vel = state[0]  # body-frame angular velocity
        ang_pos = state[1]  # ground-frame euler attitude
        lin_vel = state[2]  # body-frame linear velocity
        lin_pos = state[3]  # ground-frame position

        vec_to_hoop = HOOP_CENTER - lin_pos
        dist = float(np.linalg.norm(vec_to_hoop))
        unit_to_hoop = vec_to_hoop / (dist + 1e-8)
        signed_dist_norm = self._signed_dist(lin_pos) / ARENA_RADIUS

        return np.concatenate(
            [ang_vel, ang_pos, lin_vel, lin_pos, unit_to_hoop, [signed_dist_norm]],
            dtype=np.float32,
        )

    @staticmethod
    def _signed_dist(pos: np.ndarray) -> float:
        """Signed distance from `pos` to the hoop plane.

        Negative  →  on the arena-center side (correct approach side).
        Positive  →  past the hoop (scoring side).
        """
        return float(np.dot(pos - HOOP_CENTER, HOOP_OUTWARD_NORMAL))

    def _detect_score(self, drone_pos: np.ndarray, curr_signed_dist: float) -> bool:
        """Return True if the drone just flew through the hoop aperture.

        Conditions:
          1. Signed distance crossed from negative (inside) to non-negative (outside).
          2. The drone's lateral position (y, z) at the moment of crossing falls
             within the hoop aperture (radial distance from hoop center ≤ hoop radius).
        """
        if not (self._prev_signed_dist < 0.0 and curr_signed_dist >= 0.0):
            return False

        # Radial distance from hoop axis in the hoop plane (y-z plane at x=4)
        radial = np.sqrt(
            (drone_pos[1] - HOOP_CENTER[1]) ** 2 + (drone_pos[2] - HOOP_CENTER[2]) ** 2
        )
        return bool(radial <= HOOP_RADIUS)

    def _draw_hoop(self) -> None:
        """Draw the hoop and pole as static visual bodies in PyBullet.

        Uses small spheres arranged in a ring (no URDF needed).
        """
        av = self._aviary

        # Ring of spheres
        for i in range(HOOP_SEGS):
            angle = 2.0 * np.pi * i / HOOP_SEGS
            pos = [
                HOOP_CENTER[0],
                HOOP_CENTER[1] + HOOP_RADIUS * np.cos(angle),
                HOOP_CENTER[2] + HOOP_RADIUS * np.sin(angle),
            ]
            vis = av.createVisualShape(
                p.GEOM_SPHERE,
                radius=HOOP_SEG_RADIUS,
                rgbaColor=list(HOOP_RGBA),
            )
            av.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

        # Support pole: cylinder from ground to hoop base
        pole_length = HOOP_CENTER[2] - HOOP_RADIUS   # ground → bottom of ring
        pole_half = pole_length / 2.0
        pole_pos = [HOOP_CENTER[0], HOOP_CENTER[1], pole_half]
        pole_vis = av.createVisualShape(
            p.GEOM_CYLINDER,
            radius=POLE_RADIUS,
            length=pole_length,
            rgbaColor=list(POLE_RGBA),
        )
        av.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pole_vis,
            basePosition=pole_pos,
        )

    def _draw_arena(self) -> None:
        """Draw the arena boundary as a ring of vertical magenta pillars.

        60 evenly-spaced cylinders at 3 m radius, 3 m tall, semi-transparent
        magenta — clearly marks the flyable volume without obstructing the view.
        """
        av = self._aviary
        half_h = ARENA_WALL_HEIGHT / 2.0
        for i in range(ARENA_WALL_SEGS):
            angle = 2.0 * np.pi * i / ARENA_WALL_SEGS
            x = ARENA_RADIUS * np.cos(angle)
            y = ARENA_RADIUS * np.sin(angle)
            vis = av.createVisualShape(
                p.GEOM_CYLINDER,
                radius=ARENA_WALL_PILLAR_R,
                length=ARENA_WALL_HEIGHT,
                rgbaColor=list(ARENA_WALL_RGBA),
            )
            av.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, half_h],
            )
