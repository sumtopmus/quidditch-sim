"""Scripted 1v1 narratives for visual review of tag/crash/score behavior.

Two scenarios run back-to-back in a single MuJoCo viewer:
  A) Tagged out — Blue tags Red twice, then rams it down.
  B) Through despite the tag — Red gets tagged once but still scores.

Run via:  make demo  -> pick "scenarios"
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

# Toggle to draw colored event markers in the 3D scene (deferred — the
# helper is a no-op until SHOW_EVENT_MARKERS=True).
SHOW_EVENT_MARKERS = False

# Type alias for a single-side policy: takes that side's obs, returns its action.
Policy = Callable[[np.ndarray], np.ndarray]


# === Shared helpers ====================================================

_HOOP_CENTER    = np.asarray(HOOP_CENTER, dtype=np.float64)
_HOOP_NORMAL    = np.asarray(HOOP_OUTWARD_NORMAL, dtype=np.float64)
_APPROACH_POINT = np.array([0.0, 0.0, _HOOP_CENTER[2]], dtype=np.float64)
_THROUGH_POINT  = _HOOP_CENTER + 0.7 * _HOOP_NORMAL

# Action-scale constants matching simple_env / team_env: dx, dy at +/-0.2,
# dz at +/-0.1 per step. Scaling a target-relative vector by these gains and
# clipping yields a saturated delta-setpoint action.
_DX_SCALE = 0.2
_DZ_SCALE = 0.1


def _delta_action(vec: np.ndarray) -> np.ndarray:
    """Convert a world-frame target-relative vector into a delta-setpoint action."""
    if float(np.linalg.norm(vec)) < 0.01:
        return np.zeros(4, dtype=np.float32)
    return np.array([
        np.clip(vec[0] / _DX_SCALE, -1.0, 1.0),
        np.clip(vec[1] / _DX_SCALE, -1.0, 1.0),
        0.0,
        np.clip(vec[2] / _DZ_SCALE, -1.0, 1.0),
    ], dtype=np.float32)


# === Scenario A: Tagged out ============================================


def _scoring_red_factory():
    """Two-phase Red: climb to hoop altitude at origin, then push past hoop
    along its outward normal. Same pattern as test_scoring_canary so Red can
    actually score when not interfered with."""
    state = {"phase2": False}

    def policy(obs: np.ndarray) -> np.ndarray:
        pos = obs[9:12].astype(np.float64)
        if (not state["phase2"]) and np.linalg.norm(pos - _APPROACH_POINT) < 0.3:
            state["phase2"] = True
        target = _THROUGH_POINT if state["phase2"] else _APPROACH_POINT
        return _delta_action(target - pos)

    return policy


def _scenario_a_blue_factory():
    """Stateful Blue for Scenario A:
      intercept_1 -> in_sphere_1 -> pullback -> intercept_2 -> ram

    Distances tuned against the env's tag-sphere radius (~0.2 m) and
    cooldown geometry. INSIDE = 0.18 m; OUTSIDE = 0.55 m so cooldown fully
    elapses between tags. RAM_GAIN scales the to-Red vector before
    `_delta_action` clipping, ensuring saturation throughout the ram so
    the contact velocity exceeds CRASH_VEL_THR.
    """
    INSIDE_THR  = 0.18
    OUTSIDE_THR = 0.55
    RAM_GAIN    = 5.0

    state = {"phase": "intercept_1"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept_1":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere_1"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere_1":
            if dist > INSIDE_THR + 0.05:
                state["phase"] = "pullback"
            return _delta_action(to_red)

        if state["phase"] == "pullback":
            if dist > OUTSIDE_THR:
                state["phase"] = "intercept_2"
            return _delta_action(-to_red)

        if state["phase"] == "intercept_2":
            if dist < INSIDE_THR:
                state["phase"] = "ram"
            return _delta_action(to_red)

        if state["phase"] == "ram":
            return _delta_action(to_red * RAM_GAIN)

        return np.zeros(4, dtype=np.float32)

    return policy


# === Scenario B: Through despite the tag ===============================


def _scenario_b_blue_factory():
    """One-tag-then-peel-off:
      intercept -> in_sphere -> peel_off

    Blue closes once, lets the tag pulse fire, then retreats along
    `PEEL_OFF_DIR` so Red is free to continue toward the hoop.
    """
    INSIDE_THR  = 0.18
    PEEL_OFF_DIR = np.array([0.0, 1.5, 0.5], dtype=np.float64)  # +y mostly, slight up

    state = {"phase": "intercept"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere":
            # Hold for one step inside the sphere — tag_entry fires —
            # then retreat.
            state["phase"] = "peel_off"
            return _delta_action(to_red)

        if state["phase"] == "peel_off":
            return _delta_action(PEEL_OFF_DIR)

        return np.zeros(4, dtype=np.float32)

    return policy


def _log_event(t: float, label: str, rew: dict[str, float],
               totals: dict[str, float]) -> None:
    """One-line terminal log for an event."""
    print(
        f"[t={t:5.2f}s] {label:<22} "
        f"red {rew['red_0']:+.3f}  blue {rew['blue_0']:+.3f}   "
        f"totals: red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _detect_label(info: dict, rew: dict[str, float]) -> str | None:
    """Return a human-readable event label for this step, or None."""
    if info["red_0"].get("scored"):
        return "SCORE"
    if info["red_0"].get("drone_drone_crash"):
        return "DRONE-DRONE CRASH"
    if info["red_0"].get("red_floor"):
        return "RED FLOOR"
    if info["red_0"].get("red_wall_crash"):
        return "RED WALL CRASH"
    if info["red_0"].get("red_oob"):
        return "RED OOB"
    if info["blue_0"].get("blue_floor"):
        return "BLUE FLOOR"
    if info["blue_0"].get("blue_wall_crash"):
        return "BLUE WALL CRASH"
    if info["blue_0"].get("blue_oob"):
        return "BLUE OOB"
    if info["red_0"].get("tag_entry"):
        return "TAG ENTRY"
    return None


def _draw_event_marker(env, kind: str) -> None:
    """No-op until SHOW_EVENT_MARKERS=True (deferred)."""
    if not SHOW_EVENT_MARKERS:
        return
    # Future: append an mjvGeom into env._world.viewer.user_scn.geoms
    # — colored sphere at the relevant body position. Kept as a stub so
    # flipping the flag elsewhere doesn't need new plumbing.
    return


def _run_scenario(name: str, env: QuidditchTeamEnv,
                   red_policy: Policy, blue_policy: Policy,
                   max_seconds: float) -> None:
    """Run one scripted scenario from current env state until termination
    or timeout; log events to terminal."""
    print(f"\n[Scenario {name}]")
    obs, _ = env.reset(seed=42)
    totals = {"red_0": 0.0, "blue_0": 0.0}
    dt_step = 1.0 / 240.0  # 240 Hz physics; adjust if env reports differently
    max_steps = int(max_seconds / dt_step)

    for step in range(max_steps):
        actions = {
            "red_0":  red_policy(obs["red_0"]),
            "blue_0": blue_policy(obs["blue_0"]),
        }
        obs, rew, term, trunc, info = env.step(actions)
        totals["red_0"]  += rew["red_0"]
        totals["blue_0"] += rew["blue_0"]

        label = _detect_label(info, rew)
        if label is not None:
            _log_event(step * dt_step, label, rew, totals)
            _draw_event_marker(env, label)

        if any(term.values()) or any(trunc.values()):
            break

    print(
        f"[Scenario {name}] FINAL  "
        f"red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _idle_pause(env: QuidditchTeamEnv, seconds: float) -> None:
    """Step the env with zero actions for `seconds`; lets the viewer breathe
    between scenarios so the user can see the final state."""
    dt_step = 1.0 / 240.0
    for _ in range(int(seconds / dt_step)):
        env.step({
            "red_0":  np.zeros(4, dtype=np.float32),
            "blue_0": np.zeros(4, dtype=np.float32),
        })


def main() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        # Scenario A — Tagged out.
        _run_scenario(
            "A: Tagged out",
            env,
            red_policy=_scoring_red_factory(),
            blue_policy=_scenario_a_blue_factory(),
            max_seconds=25.0,
        )
        _idle_pause(env, seconds=1.5)

        # Scenario B — Through despite the tag.
        _run_scenario(
            "B: Through despite the tag",
            env,
            red_policy=_scoring_red_factory(),
            blue_policy=_scenario_b_blue_factory(),
            max_seconds=15.0,
        )

        # Hold the viewer open for inspection.
        print("\nDemo complete — viewer remains open. Ctrl-C to quit.")
        while True:
            _idle_pause(env, seconds=1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
