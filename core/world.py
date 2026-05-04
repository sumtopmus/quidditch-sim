"""World — owns the MuJoCo MjModel/MjData/viewer/renderer and the step loop.

A `World` is constructed from a list of `SceneFragment` (one per drone, hoop,
arena wall, marker pack, etc.).  It compiles the MJCF, allocates simulation
state, and drives physics for every drone view bound to it.

The split between `World` and `Quadrotor` exists so multi-drone scenes can
share a single `MjModel` (required for shared contacts) while each drone
keeps its own controller, setpoint, and ID resolution under its own prefix.

Public surface:
    World(fragments, render=False, seed=None)
    World.reset()                       — reset every registered drone
    World.step()                        — apply each drone's control, then
                                          PHYS_PER_CTRL × mj_step
    World.render_frame(w, h) -> ndarray — RGB frame from the "Fixed" camera

Camera definitions are loaded from config/camera.toml (raises if missing —
run ``make install`` to copy templates/camera.toml).
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
from core.mjcf.camera import _viewer_params, find_camera


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
        render: bool = False,
        seed: int | None = None,
    ) -> None:
        self._render = render
        self.np_random = np.random.default_rng(seed)
        self._cameras = load_camera_config()

        opts = WorldOptions(
            cameras=self._cameras,
            name="world",
            timestep=_DT_PHYSICS,
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

        # mj_resetData restores mocap_pos/quat to the body's MJCF-declared
        # zero pose; reposition each drone's TPV chase cam now so frame 0
        # is correct (otherwise it'd snap into place on the first step).
        # mj_kinematics propagates mocap_pos/quat → xpos/xquat for the
        # mocap body itself; mj_camlight then propagates xpos/xquat →
        # cam_xpos/cam_xmat for the camera child.  Both are needed before
        # the renderer sees the new pose.
        any_tpv = False
        for drone in self.drones:
            drone._update_tpv_mocap()
            any_tpv = any_tpv or drone._tpv_mocap_id >= 0
        if any_tpv:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)

        if self._render and self._viewer is None:
            import mujoco.viewer as _mjv
            self._viewer = _mjv.launch_passive(
                self.model, self.data,
                key_callback=self._make_key_callback(),
            )
            _, fixed_eye, fixed_lookat, _ = find_camera(self._cameras, "Fixed")
            az, el, dist, lookat = _viewer_params(fixed_eye, fixed_lookat)
            self._viewer.cam.azimuth   = az
            self._viewer.cam.elevation = el
            self._viewer.cam.distance  = dist
            self._viewer.cam.lookat[:] = lookat

    def step(self) -> None:
        """One control step: each drone computes a wrench, then PHYS_PER_CTRL × mj_step.

        When a viewer is attached, paces the loop to real time — MuJoCo's
        ``launch_passive`` viewer does not auto-pace, so a tight call loop
        would replay the episode as fast as the CPU can step physics.
        """
        for drone in self.drones:
            drone._compute_control()

        for _ in range(_PHYS_PER_CTRL):
            for drone in self.drones:
                drone._apply_control()
            mujoco.mj_step(self.model, self.data)

        # Reposition TPV chase cams from each drone's fresh post-step pose
        # before the viewer (or any renderer) reads the scene this frame.
        # See reset() for why mj_kinematics + mj_camlight are both needed.
        any_tpv = False
        for drone in self.drones:
            drone._update_tpv_mocap()
            any_tpv = any_tpv or drone._tpv_mocap_id >= 0
        if any_tpv:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(self.step_period)

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
        else:
            while self._viewer.is_running():
                time.sleep(0.05)

    # ── viewer key bindings ──────────────────────────────────────────────────

    def _make_key_callback(self):
        """Build a key_callback for `mujoco.viewer.launch_passive`.

        Direct-select named cameras with digit keys; ` (grave accent) returns
        to free cam.  Tab still cycles through fixed cameras (built into the
        viewer) — these digit shortcuts are additive, not a replacement.

            ` (grave) → free cam
            1 → North   2 → East    3 → South   4 → West
            5 → Top     6 → Fixed
            7 → drone_fpv (FPV)     8 → drone_tpv (TPV)

        Cameras that don't exist in the model (e.g. drone_fpv on a custom
        scene without cf2x_fragment) are silently skipped at callback build
        time — pressing their digit is a no-op.
        """
        digit_to_cam = {
            "1": "North",
            "2": "East",
            "3": "South",
            "4": "West",
            "5": "Top",
            "6": "Fixed",
            "7": "drone_fpv",
            "8": "drone_tpv",
        }
        cam_ids: dict[str, int] = {}
        for digit, name in digit_to_cam.items():
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid >= 0:
                cam_ids[digit] = cid

        def cb(keycode: int) -> None:
            if self._viewer is None:
                return
            ch = chr(keycode) if 0 < keycode < 128 else ""
            if ch == "`":
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            elif ch in cam_ids:
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = cam_ids[ch]

        return cb

    # ── rendering ─────────────────────────────────────────────────────────────

    def get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def render_frame(self, width: int, height: int) -> np.ndarray:
        """Render to an RGB (H×W×3) uint8 array using the scene's "Fixed" camera."""
        renderer = self.get_renderer(width, height)
        renderer.update_scene(self.data, camera="Fixed")
        rgba = renderer.render()          # (H, W, 4) uint8 — mujoco 3.x returns RGBA
        return rgba[:, :, :3]

    def render_grid(
        self,
        cam_names: tuple[str, str, str, str],
        cell_width: int,
        cell_height: int,
    ) -> np.ndarray:
        """Render four cameras and stitch into a 2x2 RGB grid.

        Args:
            cam_names: row-major (top-left, top-right, bottom-left, bottom-right).
            cell_width / cell_height: per-cell pixel dimensions.

        Returns:
            (2*cell_height, 2*cell_width, 3) uint8 RGB array.

        Re-uses the World's single renderer instance — sized via cell_width
        and cell_height on first call (see ``get_renderer``).  Mixing
        ``render_frame`` and ``render_grid`` with different sizes in the same
        World will use whichever size was requested first; pick one or
        recreate the World.
        """
        renderer = self.get_renderer(cell_width, cell_height)
        cells: list[np.ndarray] = []
        for name in cam_names:
            renderer.update_scene(self.data, camera=name)
            # .copy() because subsequent render() calls may overwrite the
            # internal pixel buffer; we need all four cells alive at once.
            cells.append(renderer.render()[:, :, :3].copy())
        # np.block can't be used here: for 3-D arrays it concatenates the
        # innermost list along the LAST axis (channels), not the width axis.
        # Build explicit row-major hstack/vstack instead.
        top    = np.hstack([cells[0], cells[1]])
        bottom = np.hstack([cells[2], cells[3]])
        return np.vstack([top, bottom])
