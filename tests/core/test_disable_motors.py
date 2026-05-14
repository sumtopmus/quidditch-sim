"""Quadrotor.disable_motors(): the control path writes zero into xfrc_applied.

The control path writes a body-frame wrench into `data.xfrc_applied` every
physics substep (see `core.quadrotor.Quadrotor._apply_control`).  Disabling
motors must short-circuit that write to zero, otherwise the drone keeps
flying when the env expects it to free-fall (e.g. during crash aftermath).
"""
from __future__ import annotations

import numpy as np

from core.quadrotor import Quadrotor
from tests.conftest import build_team_world


def test_disable_motors_zeros_xfrc_via_apply_control() -> None:
    """With motors disabled, _apply_control writes zeros regardless of _pwm state."""
    world, _ = build_team_world()
    try:
        red  = Quadrotor(world, prefix="red_0")
        blue = Quadrotor(world, prefix="blue_0")
        red.set_start(np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)))
        blue.set_start(np.array([[1.0, 0.0, 1.0]]), np.zeros((1, 3)))
        world.reset()

        # Pre-seed a non-trivial wrench on Red so we can prove disable_motors
        # *overwrites* with zero rather than just "doesn't add anything".
        world.data.xfrc_applied[red._drone_id, :] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        red.disable_motors()
        red._apply_control()
        assert np.allclose(world.data.xfrc_applied[red._drone_id, :], 0.0), (
            f"red xfrc should be zeroed by _apply_control under disable, "
            f"got {world.data.xfrc_applied[red._drone_id, :]}"
        )

        # Blue (still enabled) writes a non-trivial wrench from its PWM.
        # Seed _pwm directly so we don't depend on PID warm-up.
        blue._pwm[:] = 0.5
        blue._apply_control()
        assert np.linalg.norm(world.data.xfrc_applied[blue._drone_id, :3]) > 0.05, (
            f"blue should still receive thrust when enabled, "
            f"got {world.data.xfrc_applied[blue._drone_id, :3]}"
        )
    finally:
        world.disconnect()


def test_disable_motors_no_op_compute_control() -> None:
    """_compute_control is a no-op when disabled; existing _pwm is unchanged."""
    world, _ = build_team_world()
    try:
        red  = Quadrotor(world, prefix="red_0")
        Quadrotor(world, prefix="blue_0")
        red.set_start(np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)))
        world.reset()
        red.set_mode(7)

        # Manually pin PWM to a marker value before disable.
        red._pwm[:] = 0.42
        red.disable_motors()
        # After disable_motors(), _pwm is forced to zero (the helper resets it).
        assert np.allclose(red._pwm, 0.0)

        # Subsequent _compute_control must not touch _pwm.
        red._pwm[:] = 0.99  # forced marker
        red._compute_control()
        assert np.allclose(red._pwm, 0.99), (
            f"_compute_control should be no-op under disable, _pwm changed to {red._pwm}"
        )
    finally:
        world.disconnect()


def test_disable_motors_cleared_by_reset() -> None:
    """A fresh episode (world.reset()) re-enables motors."""
    world, _ = build_team_world()
    try:
        red = Quadrotor(world, prefix="red_0")
        Quadrotor(world, prefix="blue_0")
        red.set_start(np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)))
        world.reset()
        red.disable_motors()
        assert red._motors_disabled is True

        world.reset()
        assert red._motors_disabled is False
        # And _apply_control no longer forces zero.
        red._pwm[:] = 0.5
        red._apply_control()
        assert np.linalg.norm(world.data.xfrc_applied[red._drone_id, :3]) > 0.05, (
            f"red should receive thrust again after reset re-enables motors"
        )
    finally:
        world.disconnect()
