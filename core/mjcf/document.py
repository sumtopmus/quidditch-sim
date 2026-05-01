"""build_mjcf — assemble a complete MuJoCo XML document from fragments.

Top-level structure:
    <mujoco>
      <compiler/> <option/>
      <visual>          <- defaults baked in (headlight, global, quality) + visuals
      <asset>           <- defaults baked in (skybox, grid material) + asset fragments
      <worldbody>       <- defaults baked in (lights, floor) + worldbody fragments + camera
      <sensor>          <- only emitted if any fragment contributes sensors
      <contact>         <- only emitted if any fragment contributes contacts
    </mujoco>

Defaults can be suppressed via WorldOptions flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from core.mjcf.camera import _FALLBACK_CAMERA, _camera_xyaxes
from core.mjcf.fragment import SceneFragment, merge_all


@dataclass(frozen=True)
class WorldOptions:
    """Configuration for the top-level <mujoco> document.

    Attributes:
        name: model name, used as ``<mujoco model="...">``.
        timestep: physics timestep in seconds.
        gravity: world gravity vector.
        camera: dict {"eye": (x,y,z), "lookat": (x,y,z)} for the fixed
            offscreen camera + viewer.  Defaults to the hardcoded fallback
            if None.
        offwidth/offheight: max offscreen render resolution.
        offsamples: MSAA samples for offscreen rendering.
        shadowsize: shadow map resolution.
        include_default_skybox: emit gradient skybox + grid floor texture/material.
        include_default_lights: emit sun + fill lights.
        include_default_floor: emit a 12 m × 12 m grid-textured plane at z=0.
    """

    name: str = "world"
    timestep: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    camera: Optional[dict] = None
    offwidth: int = 1920
    offheight: int = 1080
    offsamples: int = 8
    shadowsize: int = 4096
    include_default_skybox: bool = True
    include_default_lights: bool = True
    include_default_floor: bool = True


_DEFAULT_SKYBOX_AND_GRID = """\
<texture type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.12"
             width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker"
             rgb1="0.28 0.32 0.36" rgb2="0.20 0.24 0.28"
             width="300" height="300" mark="edge" markrgb="0.4 0.4 0.4"/>
    <material name="grid" texture="grid" texuniform="true"
              texrepeat="5 5" reflectance="0.08"/>"""

_DEFAULT_LIGHTS = """\
<light name="sun"  pos="0 0 8"  dir="0 0 -1"  diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="-4 4 5" dir="1 -1 -1"  diffuse="0.2 0.2 0.2" specular="0 0 0"/>"""

_DEFAULT_FLOOR = """\
<geom name="floor" type="plane" size="6 6 0.05"
          material="grid" contype="1" conaffinity="1"/>"""


def build_mjcf(opts: WorldOptions, fragments: Iterable[SceneFragment]) -> str:
    """Compose fragments under one <mujoco> root with sensible defaults.

    Default fixtures (skybox + grid material, sun/fill lights, grid floor,
    fixed camera) are included unless suppressed via WorldOptions flags.
    Fragment-supplied content is appended to each section after the defaults.
    """
    merged = merge_all(fragments)

    cam = opts.camera if opts.camera is not None else _FALLBACK_CAMERA
    cam_pos, cam_xyaxes = _camera_xyaxes(cam["eye"], cam["lookat"])

    # ── <asset> ───────────────────────────────────────────────────────────
    asset_lines: list[str] = []
    if opts.include_default_skybox:
        asset_lines.append(_DEFAULT_SKYBOX_AND_GRID)
    asset_lines.extend(merged.assets)
    asset_block = "\n    ".join(asset_lines) if asset_lines else ""

    # ── <worldbody> ───────────────────────────────────────────────────────
    body_lines: list[str] = []
    if opts.include_default_lights:
        body_lines.append(_DEFAULT_LIGHTS)
    if opts.include_default_floor:
        body_lines.append(_DEFAULT_FLOOR)
    body_lines.extend(merged.worldbody)
    body_lines.append(f'<camera name="fixed" pos="{cam_pos}" xyaxes="{cam_xyaxes}"/>')
    worldbody_block = "\n    ".join(body_lines)

    # ── <visual> ──────────────────────────────────────────────────────────
    visual_lines = [
        '<headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>',
        f'<global offwidth="{opts.offwidth}" offheight="{opts.offheight}"/>',
        f'<quality offsamples="{opts.offsamples}" shadowsize="{opts.shadowsize}"/>',
    ]
    visual_lines.extend(merged.visuals)
    visual_block = "\n    ".join(visual_lines)

    # ── <sensor> / <contact> (optional) ───────────────────────────────────
    sensor_xml = ""
    if merged.sensors:
        sensor_xml = "\n  <sensor>\n    " + "\n    ".join(merged.sensors) + "\n  </sensor>"

    contact_xml = ""
    if merged.contacts:
        contact_xml = "\n  <contact>\n    " + "\n    ".join(merged.contacts) + "\n  </contact>"

    gx, gy, gz = opts.gravity

    return f"""
<mujoco model="{opts.name}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="{gx} {gy} {gz}" timestep="{opts.timestep:.8f}"/>

  <visual>
    {visual_block}
  </visual>

  <asset>
    {asset_block}
  </asset>

  <worldbody>
    {worldbody_block}
  </worldbody>{sensor_xml}{contact_xml}
</mujoco>
"""
