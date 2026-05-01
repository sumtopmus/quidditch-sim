"""SceneFragment — composable bundle of MJCF children for top-level sections.

Each fragment carries tuples of XML strings to be inserted as children of
a corresponding section in the final document (<asset>, <worldbody>,
<sensor>, <contact>, <visual>), plus an optional bundle of binary file
payloads (`asset_files`) that the World forwards to
`MjModel.from_xml_string(..., assets=...)` so MuJoCo can resolve mesh /
texture `file="..."` references against in-memory bytes without ever
touching the filesystem at sim-init time.

Build the final MJCF by merging fragments and wrapping with build_mjcf().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SceneFragment:
    """An MJCF contribution that gets merged into top-level sections.

    Each XML field is a tuple of XML strings to be inserted as children of
    the corresponding section in the final document.  Tuples (immutable)
    so fragments can be safely shared across compositions.

    `asset_files` carries binary asset payloads as ``(filename, bytes)``
    tuples (rather than a dict, which would not be hashable and would
    break ``frozen=True``).  The World converts to a dict and passes it
    to ``mujoco.MjModel.from_xml_string(..., assets=...)`` so
    ``<mesh file="cf2_0.obj"/>`` resolves against in-memory bytes.

    Caller responsibility: fragments must not declare conflicting
    ``(name, bytes_a)`` and ``(name, bytes_b)`` for the same name.  In
    practice this is satisfied by calling each asset-bundle factory
    (e.g. ``cf2x_assets()``) exactly once per scene.
    """

    assets:      tuple[str, ...] = ()
    worldbody:   tuple[str, ...] = ()
    sensors:     tuple[str, ...] = ()
    contacts:    tuple[str, ...] = ()
    visuals:     tuple[str, ...] = ()
    asset_files: tuple[tuple[str, bytes], ...] = ()

    def merge(self, other: "SceneFragment") -> "SceneFragment":
        return SceneFragment(
            assets=self.assets + other.assets,
            worldbody=self.worldbody + other.worldbody,
            sensors=self.sensors + other.sensors,
            contacts=self.contacts + other.contacts,
            visuals=self.visuals + other.visuals,
            asset_files=self.asset_files + other.asset_files,
        )


def merge_all(fragments: Iterable[SceneFragment]) -> SceneFragment:
    """Reduce an iterable of fragments via SceneFragment.merge."""
    out = SceneFragment()
    for f in fragments:
        out = out.merge(f)
    return out
