"""QuidditchTeamEnv — 1v1 attacker/defender PettingZoo ParallelEnv.

Both drones are mechanically identical; team roles are configuration.  The
env exposes two agents (`red_0`, `blue_0` by default) and one is configured
as `attacker` and the other as `defender` via constructor kwargs.

Phase 2 design: see docs/superpowers/specs/2026-05-06-team-play-design.md.

Observation (22 floats per agent — slots 0:16 byte-for-byte compatible
with simple_env._obs so warm_start_ppo_by_spec can copy the input layer):
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

import mujoco
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from core.world import World
from core.quadrotor import Quadrotor
from core.drone.cf2x import cf2x_assets, cf2x_fragment
from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import ObsSpec, load_obs_yaml

# Default non-learner spec: the 22-d body-mixed legacy team obs.  Resolved at
# import time so frozen red checkpoints load without surgery.  See decisions
# 2026-05-15 (TEAM_ENV_OBS → DUEL_V1_BODY rename).
_DUEL_V1_BODY: ObsSpec = load_obs_yaml("duel_v1_body")
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
    REWARD_LOOKAHEAD_S,
    ORACLE_HORIZON_S,
    ORACLE_TIME_CAP_S,
    TAKEDOWN_CONTACT_DIST,
)
from envs.quidditch.rewards import DEFAULT_MIDPOINT_ALPHA, default_team_stack
from envs.quidditch.rewards.stack import RewardStack, StepState


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
        reward_stack: RewardStack | None = None,
        learner_id: str | None = None,
        learner_spec: ObsSpec | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else TeamConfig()
        self.render_mode = render_mode

        self._red_id  = self.cfg.red_prefix
        self._blue_id = self.cfg.blue_prefix
        self.possible_agents = [self._red_id, self._blue_id]
        self.agents: list[str] = list(self.possible_agents)

        # Per-agent obs shape: non-learner always gets DUEL_V1_BODY (so frozen
        # Red checkpoints load without surgery).  Learner gets `learner_spec`,
        # which defaults to DUEL_V1_BODY when no learner is configured (canary
        # path: both agents on DUEL_V1_BODY, byte-identical to pre-refactor).
        if learner_id is not None and learner_id not in self.possible_agents:
            raise ValueError(
                f"learner_id={learner_id!r} not in possible_agents="
                f"{self.possible_agents}"
            )
        self._learner_id: str | None = learner_id
        self._learner_spec: ObsSpec = (
            learner_spec if learner_spec is not None else _DUEL_V1_BODY
        )

        # Observation spaces: build per-agent based on its spec.
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_spaces: dict[str, spaces.Box] = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._spec_for_agent(agent).dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
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

        # Free-joint dofadr cache for world-frame velocity readback (populated
        # on first reset).  -1 sentinel = uncached.
        self._red_dofadr:  int = -1
        self._blue_dofadr: int = -1

        # Closing-rate state for the learner (formerly in OCE).
        self._prev_dist_to_opp: float = 0.0

        # CTDE: whether Blue was tagging Red this step (for the critic onehot).
        self._last_tag_during: bool = False

        # Future-red distance cache for InterceptShaping (populated each step
        # when learner_id is set).
        self._dist_def_to_future_red:      float = 0.0
        self._dist_def_to_future_red_prev: float = 0.0

        self._np_random: np.random.Generator = np.random.default_rng()

        # YAML team_v2.yaml hardcodes agent IDs as "red_0"/"blue_0".  If cfg
        # overrides those prefixes, the canonical stack would mislabel agents
        # silently — assert match here so the failure is loud.
        if reward_stack is None:
            if self._red_id != "red_0" or self._blue_id != "blue_0":
                raise ValueError(
                    f"team_env: cfg prefixes are red={self._red_id!r}, blue={self._blue_id!r} "
                    "but default_team_stack() (loaded from team_v2.yaml) hardcodes "
                    "agent IDs 'red_0'/'blue_0'.  Build a custom RewardStack with "
                    "matching IDs and pass it via `reward_stack=`."
                )
            reward_stack = default_team_stack()
        self._reward_stack = reward_stack

    def _spec_for_agent(self, agent_id: str) -> ObsSpec:
        if agent_id == self._learner_id:
            return self._learner_spec
        return _DUEL_V1_BODY

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

        # Cache free-joint dofadrs for world-frame velocity reads.
        for prefix, attr in (
            (self._red_id,  "_red_dofadr"),
            (self._blue_id, "_blue_dofadr"),
        ):
            bid = mujoco.mj_name2id(self._world.model,
                                     mujoco.mjtObj.mjOBJ_BODY, prefix)
            jnt = int(self._world.model.body_jntadr[bid])
            setattr(self, attr, int(self._world.model.jnt_dofadr[jnt]))

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
        self._last_tag_during       = False

        # Initialise closing-rate cache (formerly OCE side).
        if self._learner_id is not None:
            learner_pos = (self._blue_pos() if self._learner_id == self._blue_id
                            else self._red_pos())
            opp_pos = (self._red_pos() if self._learner_id == self._blue_id
                        else self._blue_pos())
            self._prev_dist_to_opp = float(np.linalg.norm(opp_pos - learner_pos))
            # Initialise future-red distance cache.
            red_vel_world = self._world.data.qvel[
                self._red_dofadr : self._red_dofadr + 3
            ].copy()
            future_red = self._red_pos() + REWARD_LOOKAHEAD_S * red_vel_world
            defender_pos = (self._blue_pos() if self._learner_id == self._blue_id
                             else self._red_pos())
            self._dist_def_to_future_red_prev = float(
                np.linalg.norm(defender_pos - future_red)
            )
            self._dist_def_to_future_red = self._dist_def_to_future_red_prev
        else:
            self._prev_dist_to_opp = 0.0
            self._dist_def_to_future_red      = 0.0
            self._dist_def_to_future_red_prev = 0.0

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

        # Cache the finalized tag_during for the CTDE critic onehot (obs is
        # built at the end of this step, so it reflects this step's tag state).
        self._last_tag_during = tag_during

        # Positions used by tag shaping, distance shaping, OOB, and scoring.
        red_pos  = self._red_pos()
        blue_pos = self._blue_pos()
        dist_b2r = float(np.linalg.norm(red_pos - blue_pos))

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

        # ── Reward computation via composable stack ──────────────────────────
        dist_red  = float(np.linalg.norm(red_pos - HOOP_CENTER))
        dist_blue = float(np.linalg.norm(blue_pos - self._midpoint()))
        dist_blue_to_hoop = float(np.linalg.norm(blue_pos - HOOP_CENTER))

        # ── InterceptShaping inputs (when a learner is configured) ──────────
        if self._learner_id is not None:
            red_vel_world = self._world.data.qvel[
                self._red_dofadr : self._red_dofadr + 3
            ].copy()
            future_red = red_pos + REWARD_LOOKAHEAD_S * red_vel_world
            defender_pos = (blue_pos if self._learner_id == self._blue_id
                             else red_pos)
            self._dist_def_to_future_red_prev = self._dist_def_to_future_red
            self._dist_def_to_future_red = float(
                np.linalg.norm(defender_pos - future_red)
            )

        reward_state = StepState(
            agent_ids=(self._red_id, self._blue_id),
            red_pos=red_pos, blue_pos=blue_pos,
            dist_b2r=dist_b2r, dist_b2r_prev=self._dist_b2r_prev,
            step_period=self._red.step_period,
            tag_entry=tag_entry, tag_during=tag_during,
            dist_red_to_hoop=dist_red,
            dist_blue_to_midpoint=dist_blue,
            dist_blue_to_hoop=dist_blue_to_hoop,
            scored=scored,
            red_floor=red_floor, blue_floor=blue_floor,
            red_wall_crash=red_wall_crash, blue_wall_crash=blue_wall_crash,
            red_oob=red_oob, blue_oob=blue_oob,
            drone_drone_crash=drone_drone_crash,
            arena_radius=ARENA_RADIUS,
            tag_radius=self.cfg.tag_radius,
            dist_def_to_future_red=self._dist_def_to_future_red,
            dist_def_to_future_red_prev=self._dist_def_to_future_red_prev,
            hoop_pos=HOOP_CENTER,
        )
        rewards = self._reward_stack.compute_step(reward_state)
        self._dist_b2r_prev = dist_b2r

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
        return self._pack_agent_obs(agent_id, self._spec_for_agent(agent_id))

    def _build_agent_features(self, agent_id: str) -> dict[str, np.ndarray]:
        """Compute every feature this env can supply for one agent.

        Returns a {block_name: float32-array} dict.  `obs_spec.pack(spec, ...)`
        picks the subset the active spec asks for.  Keys match canonical
        ObsBlock.name strings post-2026-05-18 rename (see obs_spec.py).
        """
        if agent_id == self._red_id:
            self_q, opp_q = self._red,  self._blue
            self_dofadr   = self._red_dofadr
            opp_dofadr    = self._blue_dofadr
            goal_target   = HOOP_CENTER
        else:
            self_q, opp_q = self._blue, self._red
            self_dofadr   = self._blue_dofadr
            opp_dofadr    = self._red_dofadr
            goal_target   = self._midpoint()

        s = self_q.state()
        ang_vel, ang_pos, lin_vel_b, lin_pos = s[0], s[1], s[2], s[3]

        opp_s = opp_q.state()
        opp_pos, opp_lin_vel = opp_s[3], opp_s[2]

        vec_to_goal_world = goal_target - lin_pos
        dist_g            = float(np.linalg.norm(vec_to_goal_world))
        unit_to_goal      = vec_to_goal_world / (dist_g + 1e-8)
        signed_dist_norm  = self._signed_dist_to_hoop_plane(lin_pos) / ARENA_RADIUS
        vec_to_hoop_world = (HOOP_CENTER - lin_pos).astype(np.float32)

        opp_pos_rel_world = opp_pos - lin_pos
        data = self._world.data
        self_vel_world    = data.qvel[self_dofadr : self_dofadr + 3].copy()
        opp_vel_world     = data.qvel[opp_dofadr  : opp_dofadr  + 3].copy()
        opp_vel_rel_world = (opp_vel_world - self_vel_world).astype(np.float32)

        R_wb = data.xmat[self_q._drone_id].reshape(3, 3)

        dist_to_opp = float(np.linalg.norm(opp_pos_rel_world))
        if agent_id == self._learner_id:
            closing_rate = (
                (self._prev_dist_to_opp - dist_to_opp) / self._red.step_period
            )
            self._prev_dist_to_opp = dist_to_opp
        else:
            closing_rate = 0.0

        return {
            "ang_vel":                ang_vel,
            "ang_pos":                ang_pos,
            "lin_vel":                lin_vel_b,
            "lin_pos":                lin_pos,
            "unit_to_goal":           unit_to_goal,
            "signed_dist_norm":       np.array([signed_dist_norm], dtype=np.float32),
            "vec_to_hoop_world":      vec_to_hoop_world,
            "vec_to_hoop_body":       obs_spec.world_to_body(vec_to_hoop_world, R_wb),
            "vec_to_goal":            obs_spec.world_to_body(vec_to_goal_world, R_wb),
            "opp_pos_rel_world":      opp_pos_rel_world.astype(np.float32),
            "opp_pos_rel_body":       obs_spec.world_to_body(opp_pos_rel_world, R_wb),
            "opp_vel_rel_body_mixed": (opp_lin_vel - lin_vel_b).astype(np.float32),
            "opp_vel_rel_world":      opp_vel_rel_world,
            "opp_vel_rel_body_ego":   obs_spec.world_to_body(opp_vel_rel_world, R_wb),
            "closing_rate":           np.array([closing_rate], dtype=np.float32),
            "lin_vel_world":          self_vel_world.astype(np.float32),
            "time_remaining":         np.array(
                [(self._max_steps - self._step_count) / max(1, self._max_steps)],
                dtype=np.float32),
        }

    def _world_vel(self, dofadr: int) -> np.ndarray:
        return self._world.data.qvel[dofadr:dofadr + 3].copy().astype(np.float32)

    def _build_critic_features(
        self, agent_id: str, opp_next_action: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Privileged (critic-only) features for the learner, normalized at source.

        opp_next_action is injected by the caller (OCE) — the action the
        opponent applies this step.  All other features are pure functions of
        physics state.  See the 2026-06-04 design spec §5.2."""
        red_pos, blue_pos = self._red_pos(), self._blue_pos()
        red_vel = self._world_vel(self._red_dofadr)
        blue_vel = self._world_vel(self._blue_dofadr)
        cool = (self._tag_blue_on_red.state == _TagState.COOLDOWN)

        learner_pos = blue_pos if agent_id == self._blue_id else red_pos
        learner_vel = blue_vel if agent_id == self._blue_id else red_vel
        opp_pos = red_pos if agent_id == self._blue_id else blue_pos
        opp_vel = red_vel if agent_id == self._blue_id else blue_vel

        k = ORACLE_HORIZON_S
        self_future_disp = (k * learner_vel) / ARENA_RADIUS
        opp_future_rel = ((opp_pos + k * opp_vel) - learner_pos) / ARENA_RADIUS

        # score_pred (attacker = red_0): will Red's current trajectory score?
        n = HOOP_OUTWARD_NORMAL.astype(np.float32)
        signed = float(np.dot(red_pos - HOOP_CENTER, n))
        v_n = float(np.dot(red_vel, n))
        Tcap = ORACLE_TIME_CAP_S
        if v_n > 1e-4 and signed < 0.0:
            t_plane = min(-signed / v_n, Tcap)
        else:
            t_plane = Tcap
        crossing = red_pos + t_plane * red_vel
        lateral = (crossing - HOOP_CENTER) - np.dot(crossing - HOOP_CENTER, n) * n
        lateral_miss = float(np.clip(np.linalg.norm(lateral) / ARENA_RADIUS, 0.0, 1.0))
        speed = float(np.linalg.norm(red_vel))
        approach_align = float(v_n / speed) if speed > 1e-6 else 0.0
        score_pred = np.array(
            [t_plane / Tcap, lateral_miss, approach_align], dtype=np.float32)

        # takedown_pred: closest point of approach between the two drones.
        r = blue_pos - red_pos
        v = blue_vel - red_vel
        vv = float(np.dot(v, v))
        t_cpa = float(np.clip(-np.dot(r, v) / vv, 0.0, Tcap)) if vv > 1e-8 else 0.0
        min_sep = float(np.linalg.norm(r + t_cpa * v))
        imminent = 1.0 if (min_sep < TAKEDOWN_CONTACT_DIST
                           and np.linalg.norm(v) > self.cfg.crash_vel_thr) else 0.0
        takedown_pred = np.array(
            [t_cpa / Tcap, float(np.clip(min_sep / ARENA_RADIUS, 0.0, 1.0)), imminent],
            dtype=np.float32)

        # terminal margins (1 = safe, 0 = at boundary), clipped.
        def _wall_margin(p):
            return float(np.clip((ARENA_RADIUS - np.linalg.norm(p[:2])) / ARENA_RADIUS, 0.0, 1.0))
        def _floor_margin(p):
            return float(np.clip(p[2] / ARENA_WALL_HEIGHT, 0.0, 1.0))

        return {
            "opp_next_action": np.asarray(opp_next_action, np.float32).reshape(4),
            "red_pos_abs":     (red_pos / ARENA_RADIUS).astype(np.float32),
            "blue_pos_abs":    (blue_pos / ARENA_RADIUS).astype(np.float32),
            "tag_state_onehot": np.array(
                [float(self._last_tag_during), float(cool)], dtype=np.float32),
            "terminal_margins": np.array(
                [_wall_margin(red_pos), _wall_margin(blue_pos),
                 _floor_margin(red_pos), _floor_margin(blue_pos)], dtype=np.float32),
            "self_future_disp": self_future_disp.astype(np.float32),
            "opp_future_rel":   opp_future_rel.astype(np.float32),
            "score_pred":       score_pred,
            "takedown_pred":    takedown_pred,
        }

    def _pack_agent_obs(self, agent_id: str, spec: ObsSpec) -> np.ndarray:
        """Pack agent's obs from the universal feature dict under the given spec."""
        return obs_spec.pack(spec, self._build_agent_features(agent_id))

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._build_agent_obs(a) for a in self.possible_agents}
