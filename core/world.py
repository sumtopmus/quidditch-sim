"""World — owns the MuJoCo MjModel/MjData/viewer/renderer and the step loop.

A `World` is constructed from a list of `SceneFragment` (one per drone, hoop,
arena wall, marker pack, etc.).  It compiles the MJCF, allocates simulation
state, and drives physics for every drone view bound to it.

The split between `World` and `Quadrotor` exists so multi-drone scenes can
share a single `MjModel` (required for shared contacts) while each drone
keeps its own controller, setpoint, and ID resolution under its own prefix.

Public surface:
    World(fragments, camera=None, render=False, seed=None)
    World.reset()                       — reset every registered drone
    World.step()                        — apply each drone's control, then
                                          PHYS_PER_CTRL × mj_step
    World.render_frame(w, h) -> ndarray — RGB frame from the "fixed" camera
    World.disconnect()                  — close viewer + release renderer
    World.idle(active=False)            — block until the viewer window closes
    World.step_period -> float          — 1 / CONTROL_HZ
    World.model, World.data             — MjModel / MjData
    World.drones                        — list of registered Quadrotor views

Drones register themselves by appending to `World.drones` from their own
constructor — the World does not know about the `Quadrotor` class.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import mujoco

from core.mjcf import SceneFragment, WorldOptions, build_mjcf, load_camera_config, merge_all
from core.mjcf.camera import _viewer_params


# ── timing ───────────────────────────────────────────────────────────────────
PHYSICS_HZ: int = 240
CONTROL_HZ: int = 120
_DT_PHYSICS: float = 1.0 / PHYSICS_HZ          # 0.004167 s
_DT_CONTROL: float = 1.0 / CONTROL_HZ          # 0.008333 s
_PHYS_PER_CTRL: int = PHYSICS_HZ // CONTROL_HZ  # 2


class World:
    """Owns the MuJoCo model + data; steps physics for all registered drones.

    Multi-drone scenes create one `World` and one `Quadrotor` view per drone,
    each with a distinct prefix.  ``World.step()`` calls each drone's
    ``_apply_control()`` (writing into ``data.xfrc_applied``), then advances
    physics ``_PHYS_PER_CTRL`` × ``mj_step`` times before syncing the viewer.
    """

    def __init__(
        self,
        fragments: Iterable[SceneFragment],
        camera: dict | None = None,
        render: bool = False,
        seed: int | None = None,
    ) -> None:
        self._render = render
        self.np_random = np.random.default_rng(seed)
        self._camera = camera if camera is not None else load_camera_config()

        opts = WorldOptions(
            name="world",
            timestep=_DT_PHYSICS,
            camera=self._camera,
        )
        merged = merge_all(fragments)
        xml = build_mjcf(opts, [merged])
        # Forward any binary mesh / texture payloads (cf2x .obj files, etc.)
        # so MuJoCo resolves <mesh file="..."> against in-memory bytes.
        asset_dict = dict(merged.asset_files)
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_string(
            xml, assets=asset_dict
        )
        self.data:  mujoco.MjData  = mujoco.MjData(self.model)

        # Drones register themselves by appending to this list (see
        # Quadrotor.__init__).  Order = registration order = control order.
        self.drones: list = []

        # Viewer / renderer are opened lazily.
        self._viewer = None
        self._renderer = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def step_period(self) -> float:
        """Duration of one control step in seconds (= 1 / CONTROL_HZ)."""
        return _DT_CONTROL

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset physics + every registered drone, then sync the viewer."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        for drone in self.drones:
            drone._reset_qpos()

        mujoco.mj_forward(self.model, self.data)

        for drone in self.drones:
            drone._reset_controller()

        if self._render and self._viewer is None:
            import mujoco.viewer as _mjv
            self._viewer = _mjv.launch_passive(self.model, self.data)
            az, el, dist, lookat = _viewer_params(
                self._camera["eye"], self._camera["lookat"]
            )
            self._viewer.cam.azimuth   = az
            self._viewer.cam.elevation = el
            self._viewer.cam.distance  = dist
            self._viewer.cam.lookat[:] = lookat

    def step(self) -> None:
        """One control step: each drone computes a wrench, then PHYS_PER_CTRL × mj_step."""
        # Compute control once per drone.  Each drone caches its PWM and writes
        # the resulting wrench into xfrc_applied during _apply_control().
        for drone in self.drones:
            drone._compute_control()

        for _ in range(_PHYS_PER_CTRL):
            for drone in self.drones:
                drone._apply_control()
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def disconnect(self) -> None:
        """Close the viewer (if any) and release the offscreen renderer."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    def idle(self, active: bool = False) -> None:
        """Block until the user closes the viewer window.

        Args:
            active: If True, keep stepping so each drone holds its last
                setpoint.  If False (default), freeze physics and hold the
                last frame.  No-op when running headless.
        """
        if self._viewer is None or not self._viewer.is_running():
            return

        mode = "hovering" if active else "frozen"
        print(f"[idle] viewer open ({mode}) — close the window to exit.")
        if active:
            while self._viewer.is_running():
                self.step()
                time.sleep(self.step_period)
        else:
            while self._viewer.is_running():
                time.sleep(0.05)

    # ── rendering ─────────────────────────────────────────────────────────────

    def get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def render_frame(self, width: int, height: int) -> np.ndarray:
        """Render to an RGB (H×W×3) uint8 array using the scene's "fixed" camera."""
        renderer = self.get_renderer(width, height)
        renderer.update_scene(self.data, camera="fixed")
        rgba = renderer.render()          # (H, W, 4) uint8 — mujoco 3.x returns RGBA
        return rgba[:, :, :3]
