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


ARENA_NORM: float = 3.0       # ARENA_RADIUS; kept local to avoid a constants import cycle
_VEL_NORM: float = 3.0
_ANGVEL_NORM: float = 10.0

# Per-block divisors for the CTDE actor view.  Blocks absent from this map are
# passed through unchanged (scale 1.0) — including all critic blocks, which are
# normalized at source in team_env._build_critic_features.
NORM_BY_BLOCK: dict[str, float] = {
    "ang_vel":            _ANGVEL_NORM,
    "ang_pos":            float(np.pi),
    "lin_vel_world":      _VEL_NORM,
    "lin_pos":            ARENA_NORM,
    "vec_to_hoop_world":  ARENA_NORM,
    "opp_pos_rel_world":  ARENA_NORM,
    "opp_vel_rel_world":  _VEL_NORM,
    "closing_rate":       _VEL_NORM,
    "time_remaining":     1.0,
}


def pack_normalized(spec: ObsSpec, values: dict[str, ArrayLike],
                    norm_by_block: dict[str, float]) -> np.ndarray:
    """Like pack(), but divide each block by norm_by_block.get(name, 1.0).

    Used only on the CTDE Dict views; the flat path keeps calling pack() so it
    stays byte-identical."""
    arrays: list[np.ndarray] = []
    for block in spec.blocks:
        v = np.asarray(values[block.name], dtype=np.float32)
        if v.shape != (block.dim,):
            raise ValueError(
                f"pack_normalized: block {block.name!r} expects ({block.dim},), got {v.shape}")
        arrays.append(v / np.float32(norm_by_block.get(block.name, 1.0)))
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
    "vec_to_hoop_world", dim=3, frame="world",
    notes="HOOP_CENTER - learner_pos, not normalized",
)
OPP_POS_REL = ObsBlock("opp_pos_rel_world", dim=3, frame="world")
OPP_VEL_REL_BODY = ObsBlock(
    "opp_vel_rel_body_mixed", dim=3, frame="body_mixed",
    notes="legacy: each velocity in its own body frame",
)
OPP_VEL_REL_WORLD = ObsBlock("opp_vel_rel_world", dim=3, frame="world")
CLOSING_RATE = ObsBlock(
    "closing_rate", dim=1,
    notes="-d‖opp - learner‖/dt",
)
VEC_TO_GOAL_BODY = ObsBlock(
    "vec_to_goal", dim=3, frame="body",
    notes="goal point - learner_pos, rotated into learner body frame; "
          "goal = α·red_pos + (1-α)·hoop_center (α from TeamConfig.midpoint_alpha)",
)
VEC_TO_HOOP_BODY     = ObsBlock("vec_to_hoop_body",  dim=3, frame="body")
OPP_POS_REL_BODY     = ObsBlock("opp_pos_rel_body",  dim=3, frame="body")
OPP_VEL_REL_BODY_EGO = ObsBlock(
    "opp_vel_rel_body_ego", dim=3, frame="body",
    notes="(opp_vel_world - learner_vel_world) rotated into learner body "
          "frame; distinct from OPP_VEL_REL_BODY (body_mixed)",
)

# ── CTDE actor-view blocks (world-frame, normalized at pack-time) ────────────
LIN_VEL_WORLD = ObsBlock("lin_vel_world", dim=3, frame="world",
                         notes="world-frame linear velocity (qvel[dofadr:+3])")
TIME_REMAINING = ObsBlock("time_remaining", dim=1,
                          notes="(max_steps - step) / max_steps, in [0,1]")

# ── Critic-only (privileged) blocks — CTDE value head, world frame ───────────
OPP_NEXT_ACTION = ObsBlock("opp_next_action", dim=4,
                           notes="opponent's applied action this step (injected)")
SELF_FUTURE_DISP = ObsBlock("self_future_disp", dim=3, frame="world",
                            notes="k·v_self_world / arena")
OPP_FUTURE_REL = ObsBlock("opp_future_rel", dim=3, frame="world",
                          notes="((opp_pos + k·v_opp) - self_pos)/arena")
SCORE_PRED = ObsBlock("score_pred", dim=3,
                      notes="[time_to_plane/Tcap, lateral_miss/arena, approach_align]")
TAKEDOWN_PRED = ObsBlock("takedown_pred", dim=3,
                         notes="[time_to_cpa/Tcap, min_sep/arena, imminent_flag]")
RED_POS_ABS = ObsBlock("red_pos_abs", dim=3, frame="world", notes="red_pos/arena")
BLUE_POS_ABS = ObsBlock("blue_pos_abs", dim=3, frame="world", notes="blue_pos/arena")
TAG_STATE_ONEHOT = ObsBlock("tag_state_onehot", dim=2,
                            notes="[tag_during, tag_cooldown_active]")
TERMINAL_MARGINS = ObsBlock("terminal_margins", dim=4,
                            notes="[red_wall, blue_wall, red_floor, blue_floor] margins")


# ── Legacy persisted obs-block names ─────────────────────────────────────────
# Translates pre-2026-05-18 `[obs] slots` entries on read so old run_info.toml
# files still match the renamed canonical blocks.  Do not extend — new runs
# persist the unique names directly.  Body-frame variants of these blocks
# were not persisted before 2026-05-18, so no body-frame entries appear here.
_LEGACY_NAME_RENAMES: dict[tuple[str, str | None], str] = {
    ("opp_vel_rel", "body_mixed"): "opp_vel_rel_body_mixed",
    ("opp_vel_rel", "world"):      "opp_vel_rel_world",
    ("vec_to_hoop", "world"):      "vec_to_hoop_world",
    ("opp_pos_rel", "world"):      "opp_pos_rel_world",
}


def _apply_legacy_rename(name: str, frame: str | None) -> str:
    """Return the post-2026-05-18 unique name for a (name, frame) pair, else `name`."""
    return _LEGACY_NAME_RENAMES.get((name, frame), name)


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
            f"upgrade the YAML to the new schema (see Task 4 of the "
            f"2026-05-18-yaml-driven-obs plan)."
        )
    return build_spec_from_block_names(cfg["blocks"])


def build_ctde_specs_from_yaml(stem: str) -> tuple[ObsSpec, ObsSpec]:
    """Load conf/obs/<stem>.yaml and return (actor_spec, critic_spec).

    Requires `actor_blocks:` + `critic_blocks:` (the dual-view schema).
    Raises KeyError if either is missing."""
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo_root / "conf" / "obs" / f"{stem}.yaml").read_text())
    for key in ("actor_blocks", "critic_blocks"):
        if key not in cfg:
            raise KeyError(f"conf/obs/{stem}.yaml missing `{key}:` (CTDE dual-view schema)")
    return (build_spec_from_block_names(cfg["actor_blocks"]),
            build_spec_from_block_names(cfg["critic_blocks"]))


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
    import glob
    repo_root = Path(__file__).resolve().parents[2]
    for yaml_path in sorted(glob.glob(str(repo_root / "conf" / "obs" / "*.yaml"))):
        stem = Path(yaml_path).stem
        spec = load_obs_yaml(stem)
        print(describe(spec, stem))
        print()
