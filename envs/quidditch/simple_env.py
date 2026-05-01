"""QuidditchSimpleEnv — single-drone, single-hoop scoring task.

Game rules (Phase 1 milestone):
  - Circular arena, 3 m radius (6 m diameter).
  - One vertical hoop (0.5 m diameter) on a pole, at (2, 0, 2) m — i.e.
    2 m from arena center (1 m from the wall), 2 m height, hoop plane perpendicular to the ground
    and facing the arena center.
  - Drone starts at a random position on the ground within the arena (randomise_start=True)
    or fixed at the arena center (randomise_start=False).
  - Score: drone enters the hoop's scoring trigger volume (a thin cylinder
    along the hoop normal, defined in MJCF — see envs.quidditch.scene) from
    the arena-center side and exits on the outside side.  Detection comes
    from envs.quidditch.scoring.GeomDistanceScorer (mj_geomDistance between
    the drone's probe geom and the hoop's score tube), not Python-side
    geometry.
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
    −20.0  on crash or out-of-bounds            terminal penalty

Flight mode: mode 7 — position setpoint [x, y, yaw, z].
"""

from __future__ import annotations

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.world import World
from core.quadrotor import Quadrotor
from core.drone.cf2x import cf2x_fragment
from envs.quidditch.scene import hoop_fragment, arena_wall_fragment
from envs.quidditch.scoring import GeomDistanceScorer
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------
# Scene geometry (ARENA_RADIUS, HOOP_CENTER, HOOP_OUTWARD_NORMAL) is imported
# from envs.quidditch.constants — single source of truth shared with the MJCF
# scene builder.  The constants below are RL-specific (rewards, action scale,
# episode timing) and stay local.

# Drone initial state — used when randomise_start=False
DRONE_START_POS = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
DRONE_START_ORN = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

# Start-position randomisation
START_SAMPLE_RADIUS: float = ARENA_RADIUS - 0.1

# Timing
EPISODE_SECONDS: float = 120.0
ACTION_SCALE = np.array([0.2, 0.2, 0.5, 0.1], dtype=np.float32)

# Reward
SCORE_REWARD: float = 10.0
CRASH_PENALTY: float = -20.0
DIST_REWARD_SCALE: float = 0.01
TAKEOFF_GRACE_STEPS: int = 30

# Camera / video parameters (used by VideoRecorderCallback)
VIDEO_WIDTH: int  = 640
VIDEO_HEIGHT: int = 480


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class QuidditchSimpleEnv(gym.Env):
    """Gymnasium environment: fly a single QuadX through one vertical hoop."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        randomise_start: bool = True,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.randomise_start = randomise_start

        # 16-dim flat obs vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        # 4-dim normalized action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._world: World | None = None
        self._quad: Quadrotor | None = None
        self._scorer: GeomDistanceScorer | None = None
        self._setpoint = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0
        self._max_steps: int = 0
        self._prev_signed_dist: float = 0.0
        self._crossing_started: bool = False
        self._enter_signed_dist: float = 0.0  # signed_dist when drone entered the score volume
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

        start_pos, start_orn = (
            self._sample_start()
            if self.randomise_start
            else (DRONE_START_POS[0].copy(), DRONE_START_ORN[0].copy())
        )

        if self._world is None:
            fragments = [
                cf2x_fragment(prefix="drone"),
                arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT),
                hoop_fragment(
                    "hoop", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS,
                ),
            ]
            self._world = World(
                fragments,
                render=(self.render_mode == "human"),
                seed=seed,
            )
            self._quad = Quadrotor(self._world, prefix="drone")
            self._scorer = GeomDistanceScorer(self._world, ["drone"], ["hoop"])

        self._quad.set_start(start_pos[np.newaxis], start_orn[np.newaxis])
        self._world.reset()
        self._quad.set_mode(7)

        self._setpoint = np.array(
            [start_pos[0], start_pos[1], start_orn[2], 0.1], dtype=np.float32
        )
        self._quad.set_setpoint(self._setpoint)

        self._max_steps = int(EPISODE_SECONDS / self._quad.step_period)
        self._step_count = 0
        self._takeoff_grace = TAKEOFF_GRACE_STEPS
        self._crossing_started = False
        self._enter_signed_dist = 0.0
        drone_pos = self._drone_pos()
        self._prev_signed_dist = self._signed_dist(drone_pos)

        if self.render_mode == "human":
            time.sleep(10)  # pause so the arena can be examined before the episode starts

        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._quad is not None, "Call reset() before step()."

        delta = np.asarray(action, dtype=np.float32) * ACTION_SCALE
        self._setpoint += delta
        self._setpoint[0] = np.clip(self._setpoint[0], -ARENA_RADIUS, ARENA_RADIUS)
        self._setpoint[1] = np.clip(self._setpoint[1], -ARENA_RADIUS, ARENA_RADIUS)
        self._setpoint[2] = (self._setpoint[2] + np.pi) % (2 * np.pi) - np.pi
        self._setpoint[3] = np.clip(self._setpoint[3], 0.01, 4.0)

        self._quad.set_setpoint(self._setpoint)
        self._world.step()
        self._step_count += 1

        drone_pos = self._drone_pos()
        curr_signed_dist = self._signed_dist(drone_pos)

        scored = self._detect_score(
            bool(self._scorer.overlaps()[0, 0]), curr_signed_dist
        )
        self._prev_signed_dist = curr_signed_dist

        out_of_bounds = float(np.linalg.norm(drone_pos[:2])) > ARENA_RADIUS
        if self._takeoff_grace > 0:
            self._takeoff_grace -= 1
            crashed = False
        else:
            crashed = float(drone_pos[2]) < 0.05
        timeout = self._step_count >= self._max_steps

        terminated = scored or out_of_bounds or crashed
        truncated = timeout and not terminated

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
        if self.render_mode != "rgb_array" or self._world is None:
            return None
        return self._world.render_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

    def close(self) -> None:
        if self._world is not None:
            self._world.disconnect()
            self._world = None
            self._quad = None
            self._scorer = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @property
    def _q(self) -> Quadrotor:
        assert self._quad is not None, "Call reset() before using the simulator."
        return self._quad

    def _drone_pos(self) -> np.ndarray:
        return self._q.state()[3].copy()

    def _obs(self) -> np.ndarray:
        state = self._q.state()  # (4, 3)
        ang_vel = state[0]
        ang_pos = state[1]
        lin_vel = state[2]
        lin_pos = state[3]

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
        return float(np.dot(pos - HOOP_CENTER, HOOP_OUTWARD_NORMAL))

    def _detect_score(self, in_hoop: bool, curr_signed_dist: float) -> bool:
        """Score event = drone entered the trigger tube from the inside (-x)
        and just exited it on the outside (+x).

        The trigger tube is a thin cylinder defined in MJCF, centred on the
        hoop along its outward normal.  `in_hoop` comes from the scorer's
        per-step (drones × hoops) overlap matrix
        (envs.quidditch.scoring.GeomDistanceScorer).
        """
        if in_hoop:
            if not self._crossing_started:
                # Just entered — record which side we came from.  prev_signed_dist
                # is the position one step ago, before we entered the volume.
                self._crossing_started = True
                self._enter_signed_dist = self._prev_signed_dist
            return False

        if not self._crossing_started:
            return False

        # Just exited the trigger volume.
        self._crossing_started = False
        return self._enter_signed_dist < 0.0 and curr_signed_dist > 0.0

    def _sample_start(self) -> tuple[np.ndarray, np.ndarray]:
        r = START_SAMPLE_RADIUS * float(np.sqrt(self.np_random.uniform(0.0, 1.0)))
        theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=np.float64)
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        orn = np.array([0.0, 0.0, yaw], dtype=np.float64)
        return pos, orn
