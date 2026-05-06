"""TagDistanceScorer — per-pair (attacker probe, defender tag-sphere) overlap.

For each (defender, attacker) pair, calls mj_geomDistance between the
defender's probe and the attacker's tag sphere, returning True iff the
signed distance is negative (probe is inside the sphere).

Mirrors envs.quidditch.scoring.GeomDistanceScorer but for the team env's
tag-zone events; the entry/exit/cooldown state machine itself lives in
team_env.py — this scorer only answers "is defender d currently inside
attacker a's tag sphere?".
"""
from __future__ import annotations

import mujoco
import numpy as np

from core.world import World


class TagDistanceScorer:
    def __init__(
        self,
        world: World,
        defender_prefixes: list[str],
        attacker_prefixes: list[str],
    ) -> None:
        """Cache geom IDs for every (defender, attacker) pair.

        Raises ValueError if any expected geom is missing (e.g. the drone
        was constructed without with_tag_sphere=True).
        """
        self._model = world.model
        self._data  = world.data

        self._pairs: list[list[tuple[int, int]]] = []
        for d_pref in defender_prefixes:
            row: list[tuple[int, int]] = []
            for a_pref in attacker_prefixes:
                probe = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{d_pref}_probe"
                )
                tag   = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{a_pref}_tag_sphere"
                )
                if probe < 0:
                    raise ValueError(
                        f"TagDistanceScorer: no geom named "
                        f"{d_pref + '_probe'!r} in the world MJCF"
                    )
                if tag < 0:
                    raise ValueError(
                        f"TagDistanceScorer: no geom named "
                        f"{a_pref + '_tag_sphere'!r} in the world MJCF "
                        f"(was the drone built with with_tag_sphere=True?)"
                    )
                row.append((probe, tag))
            self._pairs.append(row)

        self._fromto = np.zeros(6, dtype=np.float64)
        self._n_def = len(defender_prefixes)
        self._n_att = len(attacker_prefixes)

    def in_zone(self) -> np.ndarray:
        """(N_defenders, N_attackers) bool array: defender d's probe is
        inside attacker a's tag sphere now."""
        out = np.zeros((self._n_def, self._n_att), dtype=bool)
        for i, row in enumerate(self._pairs):
            for j, (probe, tag) in enumerate(row):
                d = mujoco.mj_geomDistance(
                    self._model, self._data, probe, tag, 0.0, self._fromto
                )
                out[i, j] = d < 0.0
        return out
