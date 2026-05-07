"""Shared helpers for the scripted 1v1 team-env demos.

Used by `demo/takedown_demo.py` and `demo/score_through_tag_demo.py`. Holds the
event-log plumbing, the canonical "fly through the hoop" Red policy, and the
single-scenario runner that builds a `QuidditchTeamEnv` with the live MuJoCo
viewer attached and holds the window open after the run ends.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

# Toggle to draw colored event markers in the 3D scene (deferred — the
# helper is a no-op until SHOW_EVENT_MARKERS=True).
SHOW_EVENT_MARKERS = False

# Type alias for a single-side policy: takes that side's obs, returns its action.
Policy = Callable[[np.ndarray], np.ndarray]


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


def scoring_red_factory() -> Policy:
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
    """Step the env with zero actions for `seconds`; lets the viewer render
    a held final state after the scenario ends."""
    dt_step = 1.0 / 240.0
    for _ in range(int(seconds / dt_step)):
        env.step({
            "red_0":  np.zeros(4, dtype=np.float32),
            "blue_0": np.zeros(4, dtype=np.float32),
        })


def run_single_scenario_demo(*, name: str, blue_factory: Callable[[], Policy],
                              max_seconds: float,
                              episode_seconds: float = 30.0) -> None:
    """Build a `QuidditchTeamEnv` with the MuJoCo viewer attached, run one
    scripted scenario against `scoring_red_factory()`, then keep the viewer
    open until Ctrl-C."""
    env = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False, episode_seconds=episode_seconds),
        render_mode="human",
    )
    try:
        _run_scenario(
            name,
            env,
            red_policy=scoring_red_factory(),
            blue_policy=blue_factory(),
            max_seconds=max_seconds,
        )
        print("\nDemo complete — viewer remains open. Ctrl-C to quit.")
        while True:
            _idle_pause(env, seconds=1.0)
    finally:
        env.close()
