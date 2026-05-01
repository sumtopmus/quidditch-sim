"""SceneFragment — composable bundle of MJCF children for top-level sections.

Each fragment carries tuples of XML strings to be inserted as children of
a corresponding section in the final document (<asset>, <worldbody>,
<sensor>, <contact>, <visual>).  Build the final MJCF by merging fragments
and wrapping with build_mjcf().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SceneFragment:
    """An MJCF contribution that gets merged into top-level sections.

    Each field is a tuple of XML strings to be inserted as children of the
    corresponding section in the final document.  Tuples (immutable) so
    fragments can be safely shared across compositions.
    """

    assets:    tuple[str, ...] = ()
    worldbody: tuple[str, ...] = ()
    sensors:   tuple[str, ...] = ()
    contacts:  tuple[str, ...] = ()
    visuals:   tuple[str, ...] = ()

    def merge(self, other: "SceneFragment") -> "SceneFragment":
        return SceneFragment(
            assets=self.assets + other.assets,
            worldbody=self.worldbody + other.worldbody,
            sensors=self.sensors + other.sensors,
            contacts=self.contacts + other.contacts,
            visuals=self.visuals + other.visuals,
        )


def merge_all(fragments: Iterable[SceneFragment]) -> SceneFragment:
    """Reduce an iterable of fragments via SceneFragment.merge."""
    out = SceneFragment()
    for f in fragments:
        out = out.merge(f)
    return out
