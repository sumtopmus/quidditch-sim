"""CrashDetector — classifies MuJoCo contact pairs and reports velocity-gated crash events.

Iterates `data.contact[:ncon]` each step and answers, for the team env:
  - did drone D hit the floor (after takeoff grace, gated upstream)?
  - did drone D hit a wall segment, and at what |v_rel · normal|?
  - did two drones collide, and at what |v_rel · normal|?

Velocity gating (against CRASH_VEL_THR) lives upstream — this detector
just reports the maximum |v_rel · normal| observed across all matching
contacts in the current step.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from core.world import World


_FLOOR_GEOM_NAME = "floor"
_WALL_PREFIX     = "arena_wall_seg_"


@dataclass
class CrashEvents:
    solo_floor:  dict[str, bool]
    wall:        dict[str, float]
    drone_drone: tuple[str, str, float] | None


class CrashDetector:
    def __init__(self, world: World, drone_prefixes: list[str]) -> None:
        self._model = world.model
        self._data  = world.data
        self._drone_prefixes = list(drone_prefixes)

        self._floor_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM_NAME
        )
        if self._floor_id < 0:
            raise ValueError(
                f"CrashDetector: no geom named {_FLOOR_GEOM_NAME!r} in the world."
            )

        self._owner: dict[int, str] = {self._floor_id: "floor"}

        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith(_WALL_PREFIX):
                self._owner[i] = "wall"

        for i in range(self._model.ngeom):
            if i in self._owner:
                continue
            if int(self._model.geom_contype[i]) != 1:
                continue
            body_id = int(self._model.geom_bodyid[i])
            body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for pref in self._drone_prefixes:
                if body_name == pref:
                    self._owner[i] = pref
                    break

    def events(self) -> CrashEvents:
        d = self._data
        m = self._model

        solo_floor: dict[str, bool] = {p: False for p in self._drone_prefixes}
        wall:       dict[str, float] = {p: 0.0   for p in self._drone_prefixes}
        drone_drone_max: tuple[str, str, float] | None = None

        v6_a = np.zeros(6, dtype=np.float64)
        v6_b = np.zeros(6, dtype=np.float64)

        for c in range(int(d.ncon)):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            o1 = self._owner.get(g1)
            o2 = self._owner.get(g2)
            if o1 is None or o2 is None:
                continue

            drone_side: str | None = None
            other_side: str | None = None
            if o1 in self._drone_prefixes and o2 not in self._drone_prefixes:
                drone_side, other_side = o1, o2
            elif o2 in self._drone_prefixes and o1 not in self._drone_prefixes:
                drone_side, other_side = o2, o1

            if drone_side is not None and other_side == "floor":
                solo_floor[drone_side] = True
                continue

            if drone_side is not None and other_side == "wall":
                body_id = int(m.geom_bodyid[g1 if o1 == drone_side else g2])
                mujoco.mj_objectVelocity(
                    m, d, mujoco.mjtObj.mjOBJ_BODY, body_id, v6_a, 0
                )
                normal = np.asarray(con.frame[0:3], dtype=np.float64)
                vrel = abs(float(np.dot(v6_a[3:6], normal)))
                if vrel > wall[drone_side]:
                    wall[drone_side] = vrel
                continue

            if o1 in self._drone_prefixes and o2 in self._drone_prefixes:
                a_pref, b_pref = o1, o2
                body_a = int(m.geom_bodyid[g1])
                body_b = int(m.geom_bodyid[g2])
                mujoco.mj_objectVelocity(
                    m, d, mujoco.mjtObj.mjOBJ_BODY, body_a, v6_a, 0
                )
                mujoco.mj_objectVelocity(
                    m, d, mujoco.mjtObj.mjOBJ_BODY, body_b, v6_b, 0
                )
                normal = np.asarray(con.frame[0:3], dtype=np.float64)
                v_rel_normal = abs(float(np.dot(v6_a[3:6] - v6_b[3:6], normal)))
                if drone_drone_max is None or v_rel_normal > drone_drone_max[2]:
                    drone_drone_max = (a_pref, b_pref, v_rel_normal)
                continue

        return CrashEvents(
            solo_floor=solo_floor, wall=wall, drone_drone=drone_drone_max
        )
