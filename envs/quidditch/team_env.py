"""QuidditchTeamEnv — 1v1 attacker/defender PettingZoo ParallelEnv.

Both drones are mechanically identical; team roles are configuration.  The
env exposes two agents (`red_0`, `blue_0` by default) and one is configured
as `attacker` and the other as `defender` via constructor kwargs.

Phase 2 design: see docs/superpowers/specs/2026-05-06-team-play-design.md.

Observation (22 floats per agent — slots 0:16 byte-for-byte compatible
with simple_env._obs so warm_start_ppo can copy the input layer):
    [0:3]   angular velocity  — body frame, rad/s
    [3:6]   attitude euler    — ground frame, rad
    [6:9]   linear velocity   — body frame, m/s
    [9:12]  position          — ground frame, m
    [12:15] unit vector to goal target (hoop center for Red; midpoint for Blue)
    [15]    signed distance to hoop plane / ARENA_RADIUS
    [16:19] opp_pos - self_pos (world frame)
    [19:22] opp_vel - self_vel (body frame, matches simple_env vel encoding)

Action (4 floats, normalized to [-1, 1]) — same delta-setpoint scheme as
simple_env: dx, dy, dyaw, dz applied to (x, y, yaw, z) setpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from core.world import World
from core.quadrotor import Quadrotor
from core.drone.cf2x import cf2x_assets, cf2x_fragment
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment
from envs.quidditch.scoring import GeomDistanceScorer
from envs.quidditch.tagging import TagDistanceScorer
from envs.quidditch.crash import CrashDetector
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
    BLUE_START_POS,
    BLUE_START_YAW,
    TAG_RADIUS,
    TAG_COOLDOWN_SECONDS,
    CRASH_VEL_THR,
)
from envs.quidditch.rewards import (
    SCORE_REWARD,
    CRASH_PENALTY,
    DIST_REWARD_SCALE,
    HOOP_ANCHOR_SCALE,
    TAG_ENTRY_REWARD,
    TAG_DURATION_REWARD_MAX,
    CLOSING_VEL_REWARD_SCALE,
    TAKE_DOWN_REWARD,
    TAKE_DOWN_PENALTY,
    DEFAULT_MIDPOINT_ALPHA,
)


EPISODE_SECONDS_DEFAULT: float = 30.0
ACTION_SCALE = np.array([0.2, 0.2, 0.5, 0.1], dtype=np.float32)
TAKEOFF_GRACE_STEPS: int = 30
START_SAMPLE_RADIUS: float = ARENA_RADIUS - 0.1


class _TagState:
    """Per-pair tag state machine: IDLE → IN_ZONE → COOLDOWN → IDLE/IN_ZONE_QUIET."""
    IDLE = 0
    IN_ZONE = 1
    COOLDOWN = 2

    def __init__(self) -> None:
        self.state: int = _TagState.IDLE
        self.cooldown_ticks: int = 0


@dataclass
class TeamConfig:
    red_prefix: str = "red_0"
    blue_prefix: str = "blue_0"
    hoop_prefix: str = "hoop_0"
    midpoint_alpha: float = DEFAULT_MIDPOINT_ALPHA
    tag_radius: float = TAG_RADIUS
    tag_cooldown_s: float = TAG_COOLDOWN_SECONDS
    crash_vel_thr: float = CRASH_VEL_THR
    walls_collide: bool = True
    randomise_red_start: bool = True
    episode_seconds: float = EPISODE_SECONDS_DEFAULT
    # Eval-only: when > 0, a drone-drone ram does not terminate immediately.
    # Instead Red's motors are cut and the env keeps stepping for this many
    # extra seconds so the crash is observable on video.  Rewards are frozen
    # at 0 during the aftermath; all other terminal conditions are suppressed
    # until the timer expires.  Default 0 = legacy training-safe behavior.
    crash_aftermath_seconds: float = 0.0


class QuidditchTeamEnv(ParallelEnv):
    """1v1 attacker (red) / defender (blue) team env.  See module docstring."""

    metadata = {"name": "quidditch_team_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        cfg: TeamConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else TeamConfig()
        self.render_mode = render_mode

        self._red_id  = self.cfg.red_prefix
        self._blue_id = self.cfg.blue_prefix
        self.possible_agents = [self._red_id, self._blue_id]
        self.agents: list[str] = list(self.possible_agents)

        obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_spaces: dict[str, spaces.Box] = {
            self._red_id:  obs_box,
            self._blue_id: obs_box,
        }
        self.action_spaces: dict[str, spaces.Box] = {
            self._red_id:  act_box,
            self._blue_id: act_box,
        }

        self._world: World | None = None
        self._red:  Quadrotor | None = None
        self._blue: Quadrotor | None = None
        self._hoop_scorer: GeomDistanceScorer | None = None
        self._tag_scorer:  TagDistanceScorer  | None = None
        self._crash_detector: CrashDetector | None = None

        self._setpoint_red  = np.zeros(4, dtype=np.float32)
        self._setpoint_blue = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0
        self._max_steps:  int = 0
        self._cooldown_ticks: int = 0

        self._red_takeoff_grace:  int = 0
        self._blue_takeoff_grace: int = 0

        self._tag_blue_on_red = _TagState()

        self._red_crossing_started: bool = False
        self._red_enter_signed_dist: float = 0.0
        self._red_prev_signed_dist: float  = 0.0
        self._dist_b2r_prev: float = 0.0  # used by closing-velocity shaping inside the tag zone

        # Aftermath state: when > 0, a drone-drone ram already fired and the
        # env is in the post-crash observation window with Red's motors cut.
        self._aftermath_steps_left: int = 0

        self._np_random: np.random.Generator = np.random.default_rng()

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self.action_spaces[agent]

    # ── World construction (lazy on first reset) ─────────────────────────────

    def _build_world(self, *, seed: int | None) -> None:
        fragments = [
            cf2x_assets(with_collision_meshes=True),
            cf2x_fragment(prefix=self._red_id,  with_collisions=True,
                          with_tag_sphere=True, tag_sphere_rgba=(1.0, 0.0, 0.0, 0.15),
                          body_frame_rgba=(0.75, 0.10, 0.10, 1.0)),
            cf2x_fragment(prefix=self._blue_id, with_collisions=True,
                          with_tag_sphere=True, tag_sphere_rgba=(0.0, 0.0, 1.0, 0.15),
                          body_frame_rgba=(0.10, 0.20, 0.75, 1.0)),
            arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT,
                                 with_collisions=self.cfg.walls_collide),
            hoop_fragment(self.cfg.hoop_prefix, HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
        ]
        self._world = World(
            fragments,
            render=(self.render_mode == "human"),
            seed=seed,
        )
        self._red  = Quadrotor(self._world, prefix=self._red_id)
        self._blue = Quadrotor(self._world, prefix=self._blue_id)
        self._hoop_scorer = GeomDistanceScorer(
            self._world, [self._red_id], [self.cfg.hoop_prefix]
        )
        self._tag_scorer = TagDistanceScorer(
            self._world, defender_prefixes=[self._blue_id], attacker_prefixes=[self._red_id]
        )
        self._crash_detector = CrashDetector(self._world, [self._red_id, self._blue_id])

    # ── ParallelEnv API: reset / step / render / close ───────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        if self._world is None:
            self._build_world(seed=seed)

        red_pos, red_yaw = self._sample_red_start()
        blue_pos = BLUE_START_POS.copy()
        blue_yaw = BLUE_START_YAW

        self._red.set_start(red_pos[np.newaxis], np.array([[0.0, 0.0, red_yaw]]))
        self._blue.set_start(blue_pos[np.newaxis], np.array([[0.0, 0.0, blue_yaw]]))
        self._world.reset()
        self._red.set_mode(7)
        self._blue.set_mode(7)

        self._setpoint_red  = np.array([red_pos[0],  red_pos[1],  red_yaw,  0.1], dtype=np.float32)
        self._setpoint_blue = np.array([blue_pos[0], blue_pos[1], blue_yaw, blue_pos[2]], dtype=np.float32)
        self._red.set_setpoint(self._setpoint_red)
        self._blue.set_setpoint(self._setpoint_blue)

        self._max_steps = int(self.cfg.episode_seconds / self._red.step_period)
        self._cooldown_ticks = int(ceil(self.cfg.tag_cooldown_s / self._red.step_period))
        self._step_count = 0
        self._red_takeoff_grace  = TAKEOFF_GRACE_STEPS
        self._blue_takeoff_grace = 0
        self._tag_blue_on_red = _TagState()
        self._red_crossing_started  = False
        self._red_enter_signed_dist = 0.0
        self._red_prev_signed_dist  = self._signed_dist_to_hoop_plane(self._red_pos())
        self._dist_b2r_prev         = float(np.linalg.norm(self._red_pos() - self._blue_pos()))
        self._aftermath_steps_left  = 0

        if self.render_mode == "human":
            time.sleep(1)

        self.agents = list(self.possible_agents)
        obs = self._all_obs()
        return obs, {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float],
        dict[str, bool], dict[str, bool], dict[str, dict[str, Any]],
    ]:
        assert self._red is not None and self._blue is not None

        if self._aftermath_steps_left > 0:
            return self._step_aftermath(actions)

        for agent_id, action in actions.items():
            delta = np.asarray(action, dtype=np.float32) * ACTION_SCALE
            if agent_id == self._red_id:
                self._setpoint_red += delta
                self._setpoint_red[0] = np.clip(self._setpoint_red[0], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_red[1] = np.clip(self._setpoint_red[1], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_red[2] = (self._setpoint_red[2] + np.pi) % (2 * np.pi) - np.pi
                self._setpoint_red[3] = np.clip(self._setpoint_red[3], 0.01, 4.0)
                self._red.set_setpoint(self._setpoint_red)
            else:
                self._setpoint_blue += delta
                self._setpoint_blue[0] = np.clip(self._setpoint_blue[0], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_blue[1] = np.clip(self._setpoint_blue[1], -ARENA_RADIUS, ARENA_RADIUS)
                self._setpoint_blue[2] = (self._setpoint_blue[2] + np.pi) % (2 * np.pi) - np.pi
                self._setpoint_blue[3] = np.clip(self._setpoint_blue[3], 0.01, 4.0)
                self._blue.set_setpoint(self._setpoint_blue)

        self._world.step()
        self._step_count += 1

        # ── Tag state machine (single pair: blue defending vs red attacker) ─
        # tag_during reflects whether Blue is in the zone *this step*
        # (independent of state), so a step that exits doesn't pay a duration.
        # tag_entry fires only on IDLE→IN_ZONE; cooldown gates re-entry pulses.
        in_zone = bool(self._tag_scorer.in_zone()[0, 0])
        ts = self._tag_blue_on_red

        tag_entry  = False
        tag_during = in_zone

        if ts.state == _TagState.IDLE:
            if in_zone:
                tag_entry = True
                ts.state = _TagState.IN_ZONE
        elif ts.state == _TagState.IN_ZONE:
            if not in_zone:
                ts.state = _TagState.COOLDOWN
                ts.cooldown_ticks = self._cooldown_ticks
        elif ts.state == _TagState.COOLDOWN:
            ts.cooldown_ticks -= 1
            if ts.cooldown_ticks <= 0:
                ts.state = _TagState.IN_ZONE if in_zone else _TagState.IDLE

        rewards = {self._red_id: 0.0, self._blue_id: 0.0}

        # Positions used by tag shaping, distance shaping, OOB, and scoring.
        red_pos  = self._red_pos()
        blue_pos = self._blue_pos()
        dist_b2r = float(np.linalg.norm(red_pos - blue_pos))

        if tag_entry:
            rewards[self._blue_id] += TAG_ENTRY_REWARD
            rewards[self._red_id]  -= TAG_ENTRY_REWARD
        if tag_during:
            # Proximity-graded: peak at contact, 0 at zone boundary.
            prox_bonus = TAG_DURATION_REWARD_MAX * max(
                0.0, 1.0 - dist_b2r / self.cfg.tag_radius
            )
            # Closing-velocity: blue closing on red faster than red flees.
            closing = (self._dist_b2r_prev - dist_b2r) / self._red.step_period
            close_bonus = CLOSING_VEL_REWARD_SCALE * max(0.0, closing)
            tag_bonus = prox_bonus + close_bonus
            rewards[self._blue_id] += tag_bonus
            rewards[self._red_id]  -= tag_bonus
        self._dist_b2r_prev = dist_b2r

        infos: dict[str, dict[str, Any]] = {
            self._red_id:  {"tag_entry": tag_entry, "tag_during": tag_during},
            self._blue_id: {"tag_entry": tag_entry, "tag_during": tag_during},
        }

        # ── Crash detection ──────────────────────────────────────────────────
        ev = self._crash_detector.events()
        if self._red_takeoff_grace > 0:
            self._red_takeoff_grace -= 1
            ev.solo_floor[self._red_id] = False
        if self._blue_takeoff_grace > 0:
            self._blue_takeoff_grace -= 1
            ev.solo_floor[self._blue_id] = False

        red_floor   = ev.solo_floor[self._red_id]
        blue_floor  = ev.solo_floor[self._blue_id]
        red_wall_v  = ev.wall[self._red_id]
        blue_wall_v = ev.wall[self._blue_id]
        red_wall_crash  = red_wall_v  > self.cfg.crash_vel_thr
        blue_wall_crash = blue_wall_v > self.cfg.crash_vel_thr
        drone_drone_crash = (
            ev.drone_drone is not None and ev.drone_drone[2] > self.cfg.crash_vel_thr
        )

        # ── OOB ──────────────────────────────────────────────────────────────
        red_oob  = float(np.linalg.norm(red_pos[:2]))  > ARENA_RADIUS
        blue_oob = float(np.linalg.norm(blue_pos[:2])) > ARENA_RADIUS

        # ── Score detection ──────────────────────────────────────────────────
        red_in_hoop  = bool(self._hoop_scorer.overlaps()[0, 0])
        red_signed   = self._signed_dist_to_hoop_plane(red_pos)
        scored = False
        if red_in_hoop:
            if not self._red_crossing_started:
                self._red_crossing_started = True
                self._red_enter_signed_dist = self._red_prev_signed_dist
        else:
            if self._red_crossing_started:
                self._red_crossing_started = False
                if self._red_enter_signed_dist < 0.0 and red_signed > 0.0:
                    scored = True
        self._red_prev_signed_dist = red_signed

        # ── Distance shaping ─────────────────────────────────────────────────
        dist_red  = float(np.linalg.norm(red_pos - HOOP_CENTER))
        dist_blue = float(np.linalg.norm(blue_pos - self._midpoint()))
        rewards[self._red_id]  -= (dist_red  / ARENA_RADIUS) * DIST_REWARD_SCALE
        rewards[self._blue_id] -= (dist_blue / ARENA_RADIUS) * DIST_REWARD_SCALE
        # Zero-sum mirror of red's hoop-distance penalty: blue is rewarded
        # when red is far from the hoop (= blue is succeeding at defending).
        # Same magnitude as red's penalty so the two cancel when summed.
        rewards[self._blue_id] += (dist_red / ARENA_RADIUS) * DIST_REWARD_SCALE
        # Blue-only hoop anchor: penalise blue for being far from the hoop
        # regardless of red's position, so the defender doesn't follow red
        # to the arena edge and abandon the goal.
        dist_blue_to_hoop = float(np.linalg.norm(blue_pos - HOOP_CENTER))
        rewards[self._blue_id] -= (
            (dist_blue_to_hoop / ARENA_RADIUS) * HOOP_ANCHOR_SCALE
        )

        # ── Score reward (terminal, asymmetric) ──────────────────────────────
        if scored:
            rewards[self._red_id]  += SCORE_REWARD
            rewards[self._blue_id] -= SCORE_REWARD

        # ── Crash rewards ────────────────────────────────────────────────────
        if drone_drone_crash:
            rewards[self._blue_id] += TAKE_DOWN_REWARD
            rewards[self._red_id]  += TAKE_DOWN_PENALTY

        if red_floor or red_wall_crash or red_oob:
            rewards[self._red_id]  += CRASH_PENALTY
        if blue_floor or blue_wall_crash or blue_oob:
            rewards[self._blue_id] += CRASH_PENALTY

        # ── Termination ──────────────────────────────────────────────────────
        any_terminal = (
            scored
            or drone_drone_crash
            or red_floor or red_wall_crash or red_oob
            or blue_floor or blue_wall_crash or blue_oob
        )

        # Aftermath latch: drone-drone ram defers termination so the crash
        # is visible on video.  Take-down rewards still fire on this trigger
        # step; subsequent aftermath steps pay 0 reward (see _step_aftermath).
        if (
            drone_drone_crash
            and self.cfg.crash_aftermath_seconds > 0.0
            and self._aftermath_steps_left == 0
        ):
            self._enter_aftermath()
            any_terminal = False

        terminations = {self._red_id: any_terminal, self._blue_id: any_terminal}
        truncations  = {self._red_id: False, self._blue_id: False}
        if not any_terminal and self._step_count >= self._max_steps:
            truncations = {self._red_id: True, self._blue_id: True}
            self.agents = []
        elif any_terminal:
            self.agents = []

        infos[self._red_id].update({
            "scored": scored, "drone_drone_crash": drone_drone_crash,
            "red_floor": red_floor, "red_wall_crash": red_wall_crash,
            "red_oob": red_oob, "step": self._step_count,
        })
        infos[self._blue_id].update({
            "scored": scored, "drone_drone_crash": drone_drone_crash,
            "blue_floor": blue_floor, "blue_wall_crash": blue_wall_crash,
            "blue_oob": blue_oob, "step": self._step_count,
        })

        return self._all_obs(), rewards, terminations, truncations, infos

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array" or self._world is None:
            return None
        return self._world.render_frame(640, 480)

    def close(self) -> None:
        if self._world is not None:
            self._world.disconnect()
            self._world = None
            self._red = None
            self._blue = None
            self._hoop_scorer = None
            self._tag_scorer  = None
            self._crash_detector = None

    # ── Aftermath ────────────────────────────────────────────────────────────

    def _enter_aftermath(self) -> None:
        """Cut Red's motors and arm the aftermath countdown."""
        self._red.disable_motors()
        n = int(ceil(self.cfg.crash_aftermath_seconds / self._red.step_period))
        self._aftermath_steps_left = max(1, n)

    def _step_aftermath(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float],
        dict[str, bool], dict[str, bool], dict[str, dict[str, Any]],
    ]:
        """Step the world during the post-crash observation window.

        Red's motors are off (latched in `_enter_aftermath`), so Red's action
        is ignored.  Blue's policy keeps driving Blue normally.  Rewards are
        frozen at 0 and every terminal condition is suppressed until the
        timer expires, at which point the episode ends (terminated).
        """
        blue_action = actions.get(self._blue_id)
        if blue_action is not None:
            delta = np.asarray(blue_action, dtype=np.float32) * ACTION_SCALE
            self._setpoint_blue += delta
            self._setpoint_blue[0] = np.clip(self._setpoint_blue[0], -ARENA_RADIUS, ARENA_RADIUS)
            self._setpoint_blue[1] = np.clip(self._setpoint_blue[1], -ARENA_RADIUS, ARENA_RADIUS)
            self._setpoint_blue[2] = (self._setpoint_blue[2] + np.pi) % (2 * np.pi) - np.pi
            self._setpoint_blue[3] = np.clip(self._setpoint_blue[3], 0.01, 4.0)
            self._blue.set_setpoint(self._setpoint_blue)

        self._world.step()
        self._step_count += 1
        self._aftermath_steps_left -= 1

        rewards = {self._red_id: 0.0, self._blue_id: 0.0}
        done = self._aftermath_steps_left <= 0
        terminations = {self._red_id: done, self._blue_id: done}
        truncations  = {self._red_id: False, self._blue_id: False}
        if done:
            self.agents = []
        # Carry forward the originating cause flag so end-of-episode bucketing
        # still classifies this as a drone-drone crash (aftermath only fires
        # on drone_drone_crash — see _enter_aftermath caller).
        infos: dict[str, dict[str, Any]] = {
            self._red_id:  {"aftermath": True, "drone_drone_crash": True,
                            "step": self._step_count},
            self._blue_id: {"aftermath": True, "drone_drone_crash": True,
                            "step": self._step_count},
        }
        return self._all_obs(), rewards, terminations, truncations, infos

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _red_pos(self) -> np.ndarray:
        return self._red.state()[3].copy()

    def _blue_pos(self) -> np.ndarray:
        return self._blue.state()[3].copy()

    @staticmethod
    def _signed_dist_to_hoop_plane(pos: np.ndarray) -> float:
        return float(np.dot(pos - HOOP_CENTER, HOOP_OUTWARD_NORMAL))

    def _midpoint(self) -> np.ndarray:
        a = self.cfg.midpoint_alpha
        return a * self._red_pos() + (1.0 - a) * HOOP_CENTER

    def _sample_red_start(self) -> tuple[np.ndarray, float]:
        if not self.cfg.randomise_red_start:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0
        r = START_SAMPLE_RADIUS * float(np.sqrt(self._np_random.uniform(0.0, 1.0)))
        theta = float(self._np_random.uniform(0.0, 2.0 * np.pi))
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=np.float64)
        yaw = float(self._np_random.uniform(-np.pi, np.pi))
        return pos, yaw

    def _build_agent_obs(self, agent_id: str) -> np.ndarray:
        if agent_id == self._red_id:
            self_q = self._red
            opp_q  = self._blue
            goal_target = HOOP_CENTER
        else:
            self_q = self._blue
            opp_q  = self._red
            goal_target = self._midpoint()

        s = self_q.state()
        ang_vel    = s[0]
        ang_pos    = s[1]
        lin_vel_b  = s[2]
        lin_pos    = s[3]

        opp_s = opp_q.state()
        opp_pos    = opp_s[3]
        opp_lin_vel = opp_s[2]

        vec_to_goal = goal_target - lin_pos
        dist_g      = float(np.linalg.norm(vec_to_goal))
        unit_to_goal = vec_to_goal / (dist_g + 1e-8)
        signed_dist_norm = self._signed_dist_to_hoop_plane(lin_pos) / ARENA_RADIUS

        opp_pos_rel = opp_pos     - lin_pos
        opp_vel_rel = opp_lin_vel - lin_vel_b

        return np.concatenate(
            [ang_vel, ang_pos, lin_vel_b, lin_pos,
             unit_to_goal, [signed_dist_norm],
             opp_pos_rel, opp_vel_rel],
            dtype=np.float32,
        )

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._build_agent_obs(a) for a in self.possible_agents}
