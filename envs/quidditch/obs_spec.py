"""Single source of truth for observation layouts.

Each obs construction site (simple_env, team_env, OpponentControlledEnv) builds
its array via the ObsSpec declared here.  The spec is also serialized into
run_info.toml's [obs] block so load-time tools can detect shape changes.

See docs/superpowers/specs/2026-05-12-obs-spec-design.md for rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ObsBlock:
    """One named segment of an observation vector.

    Equality is on the full (name, dim, frame, notes) tuple — that is what makes
    OPP_VEL_REL_BODY and OPP_VEL_REL_WORLD distinct constants even though they
    share a name.
    """
    name: str
    dim: int
    frame: str | None = None    # "world", "body", "body_mixed", or None
    notes: str | None = None    # free-text; not machine-checked by compat


@dataclass(frozen=True)
class ObsSpec:
    """Ordered sequence of ObsBlocks. Equality is structural over blocks."""
    blocks: tuple[ObsBlock, ...]

    @property
    def dim(self) -> int:
        return sum(b.dim for b in self.blocks)

    def offsets(self) -> list[tuple[ObsBlock, slice]]:
        """Return [(block, slice(start, end)), ...] in spec order."""
        out: list[tuple[ObsBlock, slice]] = []
        off = 0
        for b in self.blocks:
            out.append((b, slice(off, off + b.dim)))
            off += b.dim
        return out


def pack(spec: ObsSpec, values: dict[str, ArrayLike]) -> np.ndarray:
    """Concatenate per-block values into a flat float32 obs vector in spec order.

    `values` keys must match block names within this spec.  Block names are unique
    within a single ObsSpec, so name-keyed lookup is unambiguous here.  See
    decision 2 in the design spec for the cross-spec name collision rule.

    Raises KeyError on missing keys, ValueError on per-block dim mismatch.
    """
    arrays: list[np.ndarray] = []
    for block in spec.blocks:
        v = values[block.name]  # KeyError if absent — intentional
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (block.dim,):
            raise ValueError(
                f"pack: block {block.name!r} expects shape ({block.dim},), "
                f"got {arr.shape}"
            )
        arrays.append(arr)
    return np.concatenate(arrays, dtype=np.float32)


# ── Canonical ObsBlock constants ─────────────────────────────────────────────
# Each block's identity is (name, dim, frame).  When the meaning of a block
# changes in a way that breaks compatibility (e.g., a frame change), declare a
# new constant rather than mutating an existing one.

ANG_VEL           = ObsBlock("ang_vel",          dim=3, frame="body")
ANG_POS           = ObsBlock("ang_pos",          dim=3, frame="body")
LIN_VEL_BODY      = ObsBlock("lin_vel",          dim=3, frame="body")
LIN_POS           = ObsBlock("lin_pos",          dim=3, frame="world")
UNIT_TO_GOAL = ObsBlock(
    "unit_to_goal", dim=3, frame="world",
    notes="unit vector toward hoop (red) or midpoint (blue)",
)
SIGNED_DIST_NORM = ObsBlock(
    "signed_dist_norm", dim=1,
    notes="(pos - hoop)·hoop_normal / ARENA_RADIUS",
)
VEC_TO_HOOP = ObsBlock(
    "vec_to_hoop", dim=3, frame="world",
    notes="HOOP_CENTER - learner_pos, not normalized",
)
OPP_POS_REL = ObsBlock("opp_pos_rel", dim=3, frame="world")
OPP_VEL_REL_BODY = ObsBlock(
    "opp_vel_rel", dim=3, frame="body_mixed",
    notes="legacy: each velocity in its own body frame",
)
OPP_VEL_REL_WORLD = ObsBlock("opp_vel_rel", dim=3, frame="world")
CLOSING_RATE = ObsBlock(
    "closing_rate", dim=1,
    notes="-d‖opp - learner‖/dt",
)


# ── Composed specs, one per obs construction site ────────────────────────────

SIMPLE_ENV_OBS: ObsSpec = ObsSpec((
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM,
))

DUEL_V1_BODY: ObsSpec = ObsSpec(
    SIMPLE_ENV_OBS.blocks + (OPP_POS_REL, OPP_VEL_REL_BODY),
)

DUEL_V2_WORLD: ObsSpec = ObsSpec((
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL,
    VEC_TO_HOOP, OPP_POS_REL, OPP_VEL_REL_WORLD, CLOSING_RATE,
))


# ── Name registry — used by config-driven obs selection ──────────────────────
# Maps the string name of a canonical spec (as written in conf/obs/*.yaml's
# `name:` field) to the ObsSpec constant itself.  Adding a new composed spec
# requires adding it here as well so `cfg.obs.name` lookups can resolve it.
SPEC_BY_NAME: dict[str, ObsSpec] = {
    "SIMPLE_ENV_OBS": SIMPLE_ENV_OBS,
    "DUEL_V1_BODY":   DUEL_V1_BODY,
    "DUEL_V2_WORLD":  DUEL_V2_WORLD,
}
