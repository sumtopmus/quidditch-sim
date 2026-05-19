"""Single source of truth for observation layouts.

Each obs construction site (simple_env, team_env, OpponentControlledEnv) builds
its array via the ObsSpec declared here.  The spec is also serialized into
run_info.toml's [obs] block so load-time tools can detect shape changes.

See docs/superpowers/specs/2026-05-12-obs-spec-design.md for rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def world_to_body(vec_world: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector from world frame into a body frame defined by R_wb.

    `R_wb` is body→world (MuJoCo `data.xmat[body_id].reshape(3, 3)`
    convention — columns are body axes expressed in world coords).  The
    inverse rotation (world→body) is R_wb.T, applied here.
    """
    return (R_wb.T @ vec_world).astype(np.float32)


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
VEC_TO_GOAL_BODY = ObsBlock(
    "vec_to_goal", dim=3, frame="body",
    notes="goal point - learner_pos, rotated into learner body frame; "
          "goal = α·red_pos + (1-α)·hoop_center (α from TeamConfig.midpoint_alpha)",
)
VEC_TO_HOOP_BODY     = ObsBlock("vec_to_hoop",  dim=3, frame="body")
OPP_POS_REL_BODY     = ObsBlock("opp_pos_rel",  dim=3, frame="body")
OPP_VEL_REL_BODY_EGO = ObsBlock(
    "opp_vel_rel", dim=3, frame="body",
    notes="(opp_vel_world - learner_vel_world) rotated into learner body "
          "frame; distinct from OPP_VEL_REL_BODY (body_mixed)",
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

DUEL_V3_BODY_EGO: ObsSpec = ObsSpec((
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS,
    VEC_TO_GOAL_BODY,
    VEC_TO_HOOP_BODY,
    OPP_POS_REL_BODY,
    OPP_VEL_REL_BODY_EGO,
    CLOSING_RATE,
))


# ── Block registry — name→block for YAML-driven obs spec resolution ──────────
# Built by introspecting module-level ObsBlock attributes.  Keys are the
# Python identifier (UPPER_SNAKE_CASE), not ObsBlock.name.  Multiple blocks
# may share a `name` field (legacy collision — see _LEGACY_NAME_RENAMES);
# their Python identifiers always differ.
#
# When adding a new module-level ObsBlock constant, add it ABOVE this line.
BLOCK_BY_NAME: dict[str, "ObsBlock"] = {
    name: obj for name, obj in dict(globals()).items()
    if isinstance(obj, ObsBlock)
}


def build_spec_from_block_names(block_names: Iterable[str]) -> ObsSpec:
    """Build an ObsSpec from a sequence of canonical block identifiers.

    Raises KeyError naming the unknown block and listing known ones."""
    blocks: list[ObsBlock] = []
    for n in block_names:
        if n not in BLOCK_BY_NAME:
            raise KeyError(
                f"Unknown ObsBlock {n!r}. "
                f"Known blocks: {sorted(BLOCK_BY_NAME)}"
            )
        blocks.append(BLOCK_BY_NAME[n])
    return ObsSpec(tuple(blocks))


def load_obs_yaml(stem: str) -> ObsSpec:
    """Load conf/obs/<stem>.yaml and build its ObsSpec.

    Convenience for tests + scripts that need a known spec by config name.
    Raises FileNotFoundError if the YAML is missing, KeyError if it lacks
    a `blocks:` field or names an unknown block."""
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / "conf" / "obs" / f"{stem}.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    if "blocks" not in cfg:
        raise KeyError(
            f"conf/obs/{stem}.yaml has no `blocks:` field — "
            f"either upgrade the YAML to the new schema or use SPEC_BY_NAME[name] directly."
        )
    return build_spec_from_block_names(cfg["blocks"])


# ── Name registry — used by config-driven obs selection ──────────────────────
# Maps the string name of a canonical spec (as written in conf/obs/*.yaml's
# `name:` field) to the ObsSpec constant itself.  Adding a new composed spec
# requires adding it here as well so `cfg.obs.name` lookups can resolve it.
SPEC_BY_NAME: dict[str, ObsSpec] = {
    "SIMPLE_ENV_OBS":   SIMPLE_ENV_OBS,
    "DUEL_V1_BODY":     DUEL_V1_BODY,
    "DUEL_V2_WORLD":    DUEL_V2_WORLD,
    "DUEL_V3_BODY_EGO": DUEL_V3_BODY_EGO,
}


def describe(spec: ObsSpec, name: str | None = None) -> str:
    """Render a human-readable block-by-block layout of one ObsSpec."""
    header = f"{name} ({spec.dim}-d):" if name else f"({spec.dim}-d):"
    lines = [header]
    for block, sl in spec.offsets():
        frame = block.frame or ""
        notes = f"  {block.notes}" if block.notes else ""
        lines.append(
            f"  [{sl.start:>2}:{sl.stop:<2}]  {block.name:<18} {block.dim}-d  {frame:<11}{notes}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for n, s in SPEC_BY_NAME.items():
        print(describe(s, n))
        print()
