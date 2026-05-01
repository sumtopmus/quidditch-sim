"""GeomDistanceScorer — per-step (N drones × M hoops) overlap matrix.

For each (drone, hoop) pair, calls mj_geomDistance between the drone's probe
geom (f"{drone_prefix}_probe") and the hoop's score tube
(f"{hoop_prefix}_score_tube") and returns True iff the signed distance is
negative (probe is inside the tube).

Used by envs.quidditch.simple_env (and future multi-drone / multi-hoop envs)
to drive the scoring state machine — entry / exit / direction is handled
upstream, the scorer just answers "is drone i currently in hoop j?".
"""

import mujoco
import numpy as np

from core.world import World


class GeomDistanceScorer:
    def __init__(
        self,
        world: World,
        drone_prefixes: list[str],
        hoop_prefixes: list[str],
    ) -> None:
        """Cache geom IDs for every (drone, hoop) pair.

        Raises ValueError if any expected geom is missing from the model
        (e.g. you passed a drone prefix that didn't have a cf2x_fragment,
        or a hoop prefix without a hoop_fragment).
        """
        self._model = world.model
        self._data  = world.data

        # _pairs[i][j] = (probe_geom_id, tube_geom_id) for drone i, hoop j.
        self._pairs: list[list[tuple[int, int]]] = []
        for d_pref in drone_prefixes:
            row: list[tuple[int, int]] = []
            for h_pref in hoop_prefixes:
                probe = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{d_pref}_probe"
                )
                tube = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{h_pref}_score_tube"
                )
                if probe < 0:
                    raise ValueError(
                        f"GeomDistanceScorer: no geom named "
                        f"{d_pref + '_probe'!r} in the world MJCF"
                    )
                if tube < 0:
                    raise ValueError(
                        f"GeomDistanceScorer: no geom named "
                        f"{h_pref + '_score_tube'!r} in the world MJCF"
                    )
                row.append((probe, tube))
            self._pairs.append(row)

        self._fromto = np.zeros(6, dtype=np.float64)
        self._n = len(drone_prefixes)
        self._m = len(hoop_prefixes)

    def overlaps(self) -> np.ndarray:
        """Return an (N, M) bool array: True iff drone i overlaps hoop j now."""
        out = np.zeros((self._n, self._m), dtype=bool)
        for i, row in enumerate(self._pairs):
            for j, (probe, tube) in enumerate(row):
                d = mujoco.mj_geomDistance(
                    self._model, self._data, probe, tube, 0.0, self._fromto
                )
                out[i, j] = d < 0.0
        return out
