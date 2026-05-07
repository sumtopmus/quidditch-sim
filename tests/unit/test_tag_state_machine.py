"""Tag state machine: enter → duration → exit → cooldown → re-enter
(no entry pulse during cooldown) → cooldown end → exit → re-enter
(entry pulse fires).

Ported from scripts/check_team_tag.py.  Drives the FSM by manually
positioning Blue inside/outside Red's tag sphere with qpos writes,
pinning Red at origin between steps.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig


def _step_zero(env: QuidditchTeamEnv) -> tuple[dict, dict]:
    """One zero-action step for both agents; returns (rewards, info)."""
    a = {ag: np.zeros(4, dtype=np.float32) for ag in env.agents}
    _obs, rew, _term, _trunc, info = env.step(a)
    return rew, info


def _set_blue(env: QuidditchTeamEnv, x: float) -> None:
    """Place blue_0 at (x, 0, 1) and run mj_forward so derived state updates."""
    bid = mujoco.mj_name2id(env._world.model, mujoco.mjtObj.mjOBJ_BODY, "blue_0")
    jnt = int(env._world.model.body_jntadr[bid])
    qa  = int(env._world.model.jnt_qposadr[jnt])
    env._world.data.qpos[qa : qa + 3] = [x, 0.0, 1.0]
    mujoco.mj_forward(env._world.model, env._world.data)


def _pin_red(env: QuidditchTeamEnv) -> None:
    """Re-pin red_0 at origin; called after each step to undo physics drift."""
    bid = mujoco.mj_name2id(env._world.model, mujoco.mjtObj.mjOBJ_BODY, "red_0")
    jnt = int(env._world.model.body_jntadr[bid])
    qa  = int(env._world.model.jnt_qposadr[jnt])
    env._world.data.qpos[qa : qa + 3] = [0.0, 0.0, 1.0]
    mujoco.mj_forward(env._world.model, env._world.data)


def test_tag_state_machine_full_lifecycle() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        env.reset(seed=42)
        env._red_takeoff_grace = 0
        _pin_red(env)

        # Phase 1: blue at 0.5 m (outside) — IDLE.
        _set_blue(env, 0.5)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and not info["red_0"]["tag_during"], (
            f"phase1 expected IDLE, got {info}"
        )

        # Phase 2: blue jumps to 0.15 m (inside) — entry pulse fires.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase2 expected entry, got {info}"
        )
        assert rew["blue_0"] > 4.9, f"phase2 expected blue +5, got {rew['blue_0']:.3f}"

        # Phase 3: blue stays at 0.15 m for one more step — duration only.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase3 expected duration only, got {info}"
        )
        assert 0.0 < rew["blue_0"] < 0.1, (
            f"phase3 expected ~+0.02, got {rew['blue_0']:.3f}"
        )

        # Phase 4: blue jumps to 0.5 m (outside) — exit.
        _set_blue(env, 0.5)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_during"], f"phase4 expected exit, got {info}"

        # Phase 5: blue jumps back to 0.15 m DURING cooldown — duration only,
        # no entry pulse.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase5 expected duration only (cooldown), got {info}"
        )
        assert 0.0 < rew["blue_0"] < 0.1, (
            f"phase5 expected ~+0.02, got {rew['blue_0']:.3f}"
        )

        # Phase 6: leave, wait full cooldown, re-enter — fresh entry pulse.
        _set_blue(env, 0.5)
        cooldown_steps = env._cooldown_ticks + 1
        for _ in range(cooldown_steps):
            _set_blue(env, 0.5)
            _step_zero(env)
            _pin_red(env)
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert info["red_0"]["tag_entry"], (
            f"phase6 expected fresh entry pulse, got {info}"
        )
        assert rew["blue_0"] > 4.9, f"phase6 expected blue +5, got {rew['blue_0']:.3f}"
    finally:
        env.close()
