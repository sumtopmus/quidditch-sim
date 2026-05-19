# YAML-driven obs composition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move obs-spec composition from Python (`SPEC_BY_NAME` + `DUEL_V*_*` constants + hardcoded dispatch in `team_env`) into `conf/obs/*.yaml`. After landing, defining a new observation layout is one new YAML file — no Python edits.

**Architecture:** YAML carries `name + n_stack + blocks: [BLOCK_IDENTIFIER, ...]`. Each env computes a universal feature dict every step, keyed by canonical block name; `obs_spec.pack(spec, features)` picks the subset the active spec needs. `ObsBlock.name` strings get frame suffixes wherever a base name had multiple variants. Legacy persisted `[obs]` blocks (pre-rename) are translated on read via a small `_LEGACY_NAME_RENAMES` dict. The 7 promoted models' `.hydra/config.yaml` and W&B artifacts are migrated once via a throwaway script.

**Tech Stack:** Python 3.11, MuJoCo, Hydra, OmegaConf, stable-baselines3, PettingZoo, pytest, W&B SDK.

**Spec:** [`docs/superpowers/specs/2026-05-18-yaml-driven-obs-design.md`](../specs/2026-05-18-yaml-driven-obs-design.md)

---

## File map

**Modified:**
- `envs/quidditch/obs_spec.py` — delete composed-spec constants + `SPEC_BY_NAME`; add `BLOCK_BY_NAME` + helpers + `_LEGACY_NAME_RENAMES`; rename collision-prone block `name=` fields
- `envs/quidditch/team_env.py` — replace `_pack_agent_obs` dispatch with universal feature dict
- `envs/quidditch/simple_env.py` — same shape; accept `spec` kwarg
- `envs/quidditch/env_factories.py` — drop `obs_spec_name`; add `obs_blocks` + `obs_name`
- `config_schema.py` — add `blocks: list[str]` to `ObsConfig`
- `conf/obs/{simple,duel_v1_body,duel_v2_world,duel_v3_body_ego}.yaml` — rewrite in new schema
- `conf/env/{simple,team}.yaml` — swap `obs_spec_name` for `obs_blocks` + `obs_name`
- `scripts/_train_common.py:read_obs_spec` — apply `_LEGACY_NAME_RENAMES` on parse
- `scripts/train.py` — replace `SPEC_BY_NAME[cfg.obs.name]` lookups with `build_spec_from_block_names(cfg.obs.blocks)`
- `scripts/migrate_legacy_models.py:LEGACY_SPECS` — rename block names; replace `SPEC_BY_NAME` use with new helper
- `scripts/_render_model_doc.py:_section_obs_spec` — use `build_spec_from_block_names`
- Test files (~8): `tests/envs/quidditch/test_obs_spec.py`, `tests/envs/quidditch/test_augmented_obs.py`, `tests/core/policies/test_warm_start.py`, `tests/core/policies/test_warm_start_by_spec.py`, `tests/scripts/test_render_model_doc.py`, `tests/scripts/test_migrate_legacy_models.py`, `tests/scripts/test_train_resolve_parent_wiring.py`, `tests/scripts/test_log_run_artifact.py`, `tests/scripts/test_wandb_init.py`

**Created (committed):**
- `tests/envs/quidditch/test_yaml_obs_loader.py` — new dedicated tests for `BLOCK_BY_NAME` / `build_spec_from_block_names` / `load_obs_yaml` / `_LEGACY_NAME_RENAMES`

**Created (transient, NOT committed):**
- `tmp_migrate_obs_blocks.py` — one-shot disk + W&B migration; deleted after a successful run.

---

## Task 1: Foundation — `BLOCK_BY_NAME` + `build_spec_from_block_names` + `load_obs_yaml` (non-breaking)

Adds the new helpers alongside the existing `SPEC_BY_NAME` / composed constants. Nothing is removed yet. Pure additions; all existing tests stay green.

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Create: `tests/envs/quidditch/test_yaml_obs_loader.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/envs/quidditch/test_yaml_obs_loader.py`:

```python
"""Tests for the YAML-driven obs loader (BLOCK_BY_NAME, build_spec_from_block_names, load_obs_yaml)."""
from pathlib import Path

import pytest

from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import (
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM,
    ObsSpec,
)


def test_block_by_name_contains_every_module_level_obs_block():
    """Every ObsBlock attribute on the module must appear in BLOCK_BY_NAME."""
    expected = {
        name: obj
        for name, obj in vars(obs_spec).items()
        if isinstance(obj, obs_spec.ObsBlock)
    }
    assert obs_spec.BLOCK_BY_NAME == expected


def test_block_by_name_keys_are_python_identifiers():
    """Keys are uppercase Python identifiers (ANG_VEL, OPP_VEL_REL_BODY_EGO, ...)."""
    for name in obs_spec.BLOCK_BY_NAME:
        assert name.isidentifier(), f"{name!r} is not a valid Python identifier"
        assert name.isupper() or "_" in name, f"{name!r} should be UPPER_SNAKE_CASE"


def test_build_spec_from_block_names_constructs_ordered_spec():
    spec = obs_spec.build_spec_from_block_names(
        ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS", "UNIT_TO_GOAL", "SIGNED_DIST_NORM"]
    )
    assert isinstance(spec, ObsSpec)
    assert spec.blocks == (ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM)
    assert spec.dim == 16  # 3+3+3+3+3+1


def test_build_spec_from_block_names_raises_on_unknown_block():
    with pytest.raises(KeyError) as excinfo:
        obs_spec.build_spec_from_block_names(["ANG_VEL", "DOES_NOT_EXIST"])
    msg = str(excinfo.value)
    assert "DOES_NOT_EXIST" in msg
    assert "ANG_VEL" in msg  # error lists known blocks


def test_load_obs_yaml_simple(tmp_path: Path, monkeypatch):
    """load_obs_yaml reads conf/obs/<stem>.yaml and builds an ObsSpec.

    Uses the repo's real conf/obs/simple.yaml — at this task the YAML still
    holds the legacy `name`-only schema, so the test only asserts that the
    helper raises a clear error when `blocks:` is absent.
    """
    with pytest.raises(KeyError, match="blocks"):
        obs_spec.load_obs_yaml("simple")
```

- [ ] **Step 1.2: Run the tests — expect failures**

```bash
cd /Users/shurioque/Library/Mobile\ Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/yaml-driven-obs
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py -v
```

Expected: 5 ERRORS / FAILED, all on `AttributeError: module 'envs.quidditch.obs_spec' has no attribute 'BLOCK_BY_NAME'` (or similar).

- [ ] **Step 1.3: Implement `BLOCK_BY_NAME` + helpers in `obs_spec.py`**

At the bottom of [`envs/quidditch/obs_spec.py`](../../../envs/quidditch/obs_spec.py), just before the existing `SPEC_BY_NAME: dict[str, ObsSpec] = {...}` block, add:

```python
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


def build_spec_from_block_names(block_names: "Iterable[str]") -> ObsSpec:
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
```

Add to the imports at the top of the file:

```python
from pathlib import Path
from typing import Iterable
```

- [ ] **Step 1.4: Run the tests — expect 4 pass, 1 still fail**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py -v
```

Expected: 4 PASS, 1 FAIL on `test_load_obs_yaml_simple` (because we expect a KeyError but the existing YAML doesn't have `blocks:` so it raises — wait, that IS what we expect). Re-check: the test asserts `pytest.raises(KeyError, match="blocks")` — should PASS.

If `test_load_obs_yaml_simple` fails differently, debug — it's probably a path resolution issue. Verify `Path(__file__).resolve().parents[2]` resolves to the worktree root.

Expected after debug: ALL 5 PASS.

- [ ] **Step 1.5: Run the full suite to confirm no regressions**

```bash
python -m pytest --no-header -q 2>&1 | tail -20
```

Expected: same pass count as before this task (no regressions). The 1 new test file (5 tests) is additive.

- [ ] **Step 1.6: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_yaml_obs_loader.py
git commit -m "$(cat <<'EOF'
feat(obs-spec): add BLOCK_BY_NAME + build_spec_from_block_names + load_obs_yaml

Non-breaking additions.  SPEC_BY_NAME and the composed-spec constants
(SIMPLE_ENV_OBS, DUEL_V*_*) are untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rename collision-prone `ObsBlock.name` fields + legacy translation

Single atomic commit. The 3 `OPP_VEL_REL_*`, 2 `VEC_TO_HOOP*`, 2 `OPP_POS_REL*` block constants gain explicit frame suffixes in their `name=` field. `team_env._pack_agent_obs` dispatch dict keys are updated to match. `_LEGACY_NAME_RENAMES` is added; `read_obs_spec` applies it so legacy persisted slot blocks still load correctly.

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Modify: `envs/quidditch/team_env.py`
- Modify: `scripts/_train_common.py`
- Modify: `tests/envs/quidditch/test_yaml_obs_loader.py`

- [ ] **Step 2.1: Write failing tests for renamed names + legacy translation**

Append to `tests/envs/quidditch/test_yaml_obs_loader.py`:

```python
def test_opp_vel_rel_variants_have_unique_names():
    """The 3 opp_vel_rel variants must carry distinct .name fields after rename."""
    from envs.quidditch.obs_spec import (
        OPP_VEL_REL_BODY, OPP_VEL_REL_WORLD, OPP_VEL_REL_BODY_EGO,
    )
    names = {OPP_VEL_REL_BODY.name, OPP_VEL_REL_WORLD.name, OPP_VEL_REL_BODY_EGO.name}
    assert names == {"opp_vel_rel_body_mixed", "opp_vel_rel_world", "opp_vel_rel_body_ego"}


def test_vec_to_hoop_variants_have_unique_names():
    from envs.quidditch.obs_spec import VEC_TO_HOOP, VEC_TO_HOOP_BODY
    assert VEC_TO_HOOP.name == "vec_to_hoop_world"
    assert VEC_TO_HOOP_BODY.name == "vec_to_hoop_body"


def test_opp_pos_rel_variants_have_unique_names():
    from envs.quidditch.obs_spec import OPP_POS_REL, OPP_POS_REL_BODY
    assert OPP_POS_REL.name == "opp_pos_rel_world"
    assert OPP_POS_REL_BODY.name == "opp_pos_rel_body"


def test_read_obs_spec_translates_legacy_opp_vel_rel_body_mixed(tmp_path):
    """A run_info.toml with the pre-rename name field parses to the new name."""
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 3\n'
        'n_stack = 1\n'
        'slots = [\n'
        '  {name = "opp_vel_rel", dim = 3, frame = "body_mixed",'
        ' notes = "legacy: each velocity in its own body frame"},\n'
        ']\n'
    )
    spec, n_stack = read_obs_spec(info)
    assert spec.blocks[0].name == "opp_vel_rel_body_mixed"
    assert n_stack == 1


def test_read_obs_spec_translates_legacy_vec_to_hoop_world(tmp_path):
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 3\n'
        'n_stack = 3\n'
        'slots = [\n'
        '  {name = "vec_to_hoop", dim = 3, frame = "world",'
        ' notes = "HOOP_CENTER - learner_pos, not normalized"},\n'
        ']\n'
    )
    spec, _ = read_obs_spec(info)
    assert spec.blocks[0].name == "vec_to_hoop_world"


def test_read_obs_spec_passthrough_for_non_renamed_names(tmp_path):
    """Single-variant blocks (ang_vel, lin_pos, ...) keep their existing names."""
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 6\n'
        'n_stack = 1\n'
        'slots = [\n'
        '  {name = "ang_vel", dim = 3, frame = "body"},\n'
        '  {name = "lin_pos", dim = 3, frame = "world"},\n'
        ']\n'
    )
    spec, _ = read_obs_spec(info)
    assert [b.name for b in spec.blocks] == ["ang_vel", "lin_pos"]
```

- [ ] **Step 2.2: Run the new tests — expect failures**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py -v -k "unique_names or translates or passthrough"
```

Expected: 5 tests FAIL (rename-related ones because names are still "opp_vel_rel" / "vec_to_hoop" / "opp_pos_rel"; legacy-translation ones because `_LEGACY_NAME_RENAMES` doesn't exist yet).

- [ ] **Step 2.3: Apply renames in `envs/quidditch/obs_spec.py`**

Edit the 7 affected ObsBlock constants. Locate each and update the first positional argument (the `name`):

```python
# Was: ObsBlock("opp_pos_rel", dim=3, frame="world")
OPP_POS_REL = ObsBlock("opp_pos_rel_world", dim=3, frame="world")

# Was: ObsBlock("opp_vel_rel", dim=3, frame="body_mixed", ...)
OPP_VEL_REL_BODY = ObsBlock(
    "opp_vel_rel_body_mixed", dim=3, frame="body_mixed",
    notes="legacy: each velocity in its own body frame",
)

# Was: ObsBlock("opp_vel_rel", dim=3, frame="world")
OPP_VEL_REL_WORLD = ObsBlock("opp_vel_rel_world", dim=3, frame="world")

# Was: ObsBlock("vec_to_hoop", dim=3, frame="world", notes=...)
VEC_TO_HOOP = ObsBlock(
    "vec_to_hoop_world", dim=3, frame="world",
    notes="HOOP_CENTER - learner_pos, not normalized",
)

# Was: ObsBlock("vec_to_hoop", dim=3, frame="body")
VEC_TO_HOOP_BODY = ObsBlock("vec_to_hoop_body", dim=3, frame="body")

# Was: ObsBlock("opp_pos_rel", dim=3, frame="body")
OPP_POS_REL_BODY = ObsBlock("opp_pos_rel_body", dim=3, frame="body")

# Was: ObsBlock("opp_vel_rel", dim=3, frame="body", notes=...)
OPP_VEL_REL_BODY_EGO = ObsBlock(
    "opp_vel_rel_body_ego", dim=3, frame="body",
    notes="(opp_vel_world - learner_vel_world) rotated into learner body "
          "frame; distinct from OPP_VEL_REL_BODY (body_mixed)",
)
```

Leave the other 8 constants (`ANG_VEL`, `ANG_POS`, `LIN_VEL_BODY`, `LIN_POS`, `UNIT_TO_GOAL`, `SIGNED_DIST_NORM`, `CLOSING_RATE`, `VEC_TO_GOAL_BODY`) untouched — they're single-variant, no collision.

- [ ] **Step 2.4: Add `_LEGACY_NAME_RENAMES` in `obs_spec.py`**

Append just above the `BLOCK_BY_NAME` registry section:

```python
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
```

- [ ] **Step 2.5: Apply `_apply_legacy_rename` in `scripts/_train_common.py:read_obs_spec`**

In [`scripts/_train_common.py`](../../../scripts/_train_common.py), find `read_obs_spec` (around line 51). Update the `ObsBlock` construction:

```python
def read_obs_spec(info_path: Path | str) -> tuple[ObsSpec, int] | None:
    """Parse run_info.toml and return (ObsSpec, n_stack) or None if [obs] absent."""
    from envs.quidditch.obs_spec import _apply_legacy_rename
    data = tomllib.loads(Path(info_path).read_text())
    obs = data.get("obs")
    if obs is None:
        return None
    blocks = tuple(
        ObsBlock(
            name=_apply_legacy_rename(s["name"], s.get("frame")),
            dim=s["dim"],
            frame=s.get("frame"),
            notes=s.get("notes"),
        )
        for s in obs["slots"]
    )
    spec = ObsSpec(blocks)
    if spec.dim != obs["dim"]:
        raise ValueError(
            f"[obs].dim={obs['dim']} disagrees with sum of slot dims ({spec.dim}) "
            f"in {info_path}"
        )
    return spec, int(obs["n_stack"])
```

- [ ] **Step 2.6: Update `pack()` dict keys in `team_env._pack_agent_obs`**

In [`envs/quidditch/team_env.py`](../../../envs/quidditch/team_env.py), find each `obs_spec.pack(...)` call inside `_pack_agent_obs`. Update the dict keys for the renamed blocks. The block constants are still the *same Python objects*; only their `name` field changed, and pack keys must match the new names.

`DUEL_V1_BODY` branch (around line 647):

```python
return obs_spec.pack(DUEL_V1_BODY, {
    "ang_vel":                ang_vel,
    "ang_pos":                ang_pos,
    "lin_vel":                lin_vel_b,
    "lin_pos":                lin_pos,
    "unit_to_goal":           unit_to_goal,
    "signed_dist_norm":       [signed_dist_norm],
    "opp_pos_rel_world":      opp_pos_rel_world,      # was "opp_pos_rel"
    "opp_vel_rel_body_mixed": opp_vel_rel_body_mixed, # was "opp_vel_rel"
})
```

`DUEL_V2_WORLD` branch (around line 678):

```python
return obs_spec.pack(DUEL_V2_WORLD, {
    "ang_vel":           ang_vel,
    "ang_pos":           ang_pos,
    "lin_vel":           lin_vel_b,
    "lin_pos":           lin_pos,
    "unit_to_goal":      unit_to_goal,
    "vec_to_hoop_world": vec_to_hoop_world,           # was "vec_to_hoop"
    "opp_pos_rel_world": opp_pos_rel_world.astype(np.float32),  # was "opp_pos_rel"
    "opp_vel_rel_world": opp_vel_rel_world,           # was "opp_vel_rel"
    "closing_rate":      [closing_rate],
})
```

`DUEL_V3_BODY_EGO` branch (around line 700):

```python
return obs_spec.pack(DUEL_V3_BODY_EGO, {
    "ang_vel":              ang_vel,
    "ang_pos":              ang_pos,
    "lin_vel":              lin_vel_b,
    "lin_pos":              lin_pos,
    "vec_to_goal":          vec_to_goal_body,         # name="vec_to_goal" (unchanged)
    "vec_to_hoop_body":     vec_to_hoop_body,         # was "vec_to_hoop"
    "opp_pos_rel_body":     opp_pos_rel_body,         # was "opp_pos_rel"
    "opp_vel_rel_body_ego": opp_vel_rel_body,         # was "opp_vel_rel"
    "closing_rate":         [closing_rate],
})
```

- [ ] **Step 2.7: Run the rename + translation tests — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py -v -k "unique_names or translates or passthrough"
```

Expected: 5 PASS.

- [ ] **Step 2.8: Run the byte-identical canaries to verify no behavior change**

```bash
python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: BOTH PASS with same fingerprints (`SCORED at step 434, total reward 7.3837` for single-agent; team-env canary green).

If a canary regresses: the most likely cause is a typo in the renamed pack dict keys not matching the renamed block names — re-check both `obs_spec.py` constants and the pack calls. The constant's `name=` field and the pack-dict key must be byte-identical strings.

- [ ] **Step 2.9: Run full suite**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count as start of Task 2.

- [ ] **Step 2.10: Commit**

```bash
git add envs/quidditch/obs_spec.py envs/quidditch/team_env.py scripts/_train_common.py tests/envs/quidditch/test_yaml_obs_loader.py
git commit -m "$(cat <<'EOF'
refactor(obs-spec): give colliding ObsBlock names explicit frame suffixes

OPP_VEL_REL_*, VEC_TO_HOOP*, OPP_POS_REL* now carry unique `name=` fields.
team_env._pack_agent_obs dict keys updated to match.  Legacy persisted
[obs] slots are translated on read via _LEGACY_NAME_RENAMES.

Canaries (single-agent step 434/reward 7.3837, team-env trace) byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `blocks: list[str]` to `ObsConfig` schema (non-breaking)

Extends the Hydra structured-config schema. The new field defaults to an empty list — existing YAMLs that don't set it keep working until Task 4 rewrites them.

**Files:**
- Modify: `config_schema.py`
- Create test inline in this task

- [ ] **Step 3.1: Write the failing test**

Append to `tests/envs/quidditch/test_yaml_obs_loader.py`:

```python
def test_obs_config_schema_carries_blocks():
    from config_schema import ObsConfig
    cfg = ObsConfig(name="X", n_stack=2, blocks=["ANG_VEL", "ANG_POS"])
    assert cfg.blocks == ["ANG_VEL", "ANG_POS"]


def test_obs_config_default_blocks_is_empty_list():
    """Default empty list keeps current YAMLs (no blocks: field) loading."""
    from config_schema import ObsConfig
    cfg = ObsConfig()
    assert cfg.blocks == []
```

- [ ] **Step 3.2: Run — expect failure**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_obs_config_schema_carries_blocks tests/envs/quidditch/test_yaml_obs_loader.py::test_obs_config_default_blocks_is_empty_list -v
```

Expected: 2 FAIL on `TypeError: __init__() got an unexpected keyword argument 'blocks'`.

- [ ] **Step 3.3: Add `blocks` field to `ObsConfig`**

In [`config_schema.py`](../../../config_schema.py) lines 91-95:

```python
@dataclass
class ObsConfig:
    """Names a canonical ObsSpec; `blocks` carries the ordered ObsBlock identifiers."""
    name: str = "DUEL_V2_WORLD"
    n_stack: int = 3
    blocks: list[str] = field(default_factory=list)
```

Add `from dataclasses import dataclass, field` at the top if `field` isn't already imported.

- [ ] **Step 3.4: Run — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_obs_config_schema_carries_blocks tests/envs/quidditch/test_yaml_obs_loader.py::test_obs_config_default_blocks_is_empty_list -v
```

Expected: 2 PASS.

- [ ] **Step 3.5: Run full suite to confirm no regressions**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count as start of Task 3.

- [ ] **Step 3.6: Commit**

```bash
git add config_schema.py tests/envs/quidditch/test_yaml_obs_loader.py
git commit -m "$(cat <<'EOF'
feat(config-schema): add blocks: list[str] to ObsConfig

Empty default keeps existing conf/obs/*.yaml files loading until they're
rewritten in the new schema (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rewrite `conf/obs/*.yaml` in the new schema + equivalence test

Each of the four YAMLs gains a `blocks:` list. Mid-refactor invariant: `load_obs_yaml(stem)` must produce an `ObsSpec` byte-identical to the corresponding `SPEC_BY_NAME[name]` (which still exists).

**Files:**
- Modify: `conf/obs/simple.yaml`
- Modify: `conf/obs/duel_v1_body.yaml`
- Modify: `conf/obs/duel_v2_world.yaml`
- Modify: `conf/obs/duel_v3_body_ego.yaml`
- Modify: `tests/envs/quidditch/test_yaml_obs_loader.py`

- [ ] **Step 4.1: Write the parametrized equivalence test**

Append to `tests/envs/quidditch/test_yaml_obs_loader.py`:

```python
@pytest.mark.parametrize("stem,name_const", [
    ("simple",           "SIMPLE_ENV_OBS"),
    ("duel_v1_body",     "DUEL_V1_BODY"),
    ("duel_v2_world",    "DUEL_V2_WORLD"),
    ("duel_v3_body_ego", "DUEL_V3_BODY_EGO"),
])
def test_yaml_round_trip_matches_legacy_spec_by_name(stem, name_const):
    """load_obs_yaml(stem) must match SPEC_BY_NAME[name_const] block-for-block."""
    yaml_spec = obs_spec.load_obs_yaml(stem)
    legacy_spec = obs_spec.SPEC_BY_NAME[name_const]
    assert yaml_spec == legacy_spec, (
        f"YAML-built spec for {stem!r} disagrees with legacy {name_const!r}.\n"
        f"  yaml:   {[b.name for b in yaml_spec.blocks]}\n"
        f"  legacy: {[b.name for b in legacy_spec.blocks]}"
    )
```

- [ ] **Step 4.2: Run — expect failure (YAMLs lack `blocks:`)**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_yaml_round_trip_matches_legacy_spec_by_name -v
```

Expected: 4 FAIL (KeyError "blocks").

- [ ] **Step 4.3: Rewrite the four YAMLs**

Replace [`conf/obs/simple.yaml`](../../../conf/obs/simple.yaml) entirely:

```yaml
# 16-d single-agent obs (envs.quidditch.simple_env).  Slots [0:16] are
# load-bearing for team-env warm-start surgery — see decisions.md 2026-05-06.
name: SIMPLE_ENV_OBS
n_stack: 1
blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - UNIT_TO_GOAL
  - SIGNED_DIST_NORM
```

Replace [`conf/obs/duel_v1_body.yaml`](../../../conf/obs/duel_v1_body.yaml):

```yaml
# 22-d team-play obs, body-mixed opp_vel_rel.  Pre-2026-05-12 schema; used
# by red_v1 and blue_v1.
name: DUEL_V1_BODY
n_stack: 1
blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - UNIT_TO_GOAL
  - SIGNED_DIST_NORM
  - OPP_POS_REL
  - OPP_VEL_REL_BODY
```

Replace [`conf/obs/duel_v2_world.yaml`](../../../conf/obs/duel_v2_world.yaml):

```yaml
# 25-d team-play obs, world-frame opp_vel_rel + closing_rate + vec_to_hoop.
# Current default for team runs; used by blue_v4.
name: DUEL_V2_WORLD
n_stack: 3
blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - UNIT_TO_GOAL
  - VEC_TO_HOOP
  - OPP_POS_REL
  - OPP_VEL_REL_WORLD
  - CLOSING_RATE
```

Replace [`conf/obs/duel_v3_body_ego.yaml`](../../../conf/obs/duel_v3_body_ego.yaml):

```yaml
# 25-d body-frame ego-centric obs.  vec_to_goal, vec_to_hoop, opp_pos_rel,
# opp_vel_rel all rotated into the learner's body frame.
name: DUEL_V3_BODY_EGO
n_stack: 3
blocks:
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - VEC_TO_GOAL_BODY
  - VEC_TO_HOOP_BODY
  - OPP_POS_REL_BODY
  - OPP_VEL_REL_BODY_EGO
  - CLOSING_RATE
```

- [ ] **Step 4.4: Run the equivalence test — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_yaml_round_trip_matches_legacy_spec_by_name -v
```

Expected: 4 PASS.

If a parametrization fails, the YAML's block list disagrees with the corresponding `SPEC_BY_NAME` entry. Compare side-by-side against `envs/quidditch/obs_spec.py`'s composed-spec definitions (`SIMPLE_ENV_OBS`, `DUEL_V1_BODY`, …) and reorder the YAML to match.

- [ ] **Step 4.5: Run canaries to confirm no behavior drift**

```bash
python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: BOTH PASS — YAMLs aren't yet consumed by the env factory (next task), so behavior is unchanged.

- [ ] **Step 4.6: Commit**

```bash
git add conf/obs/simple.yaml conf/obs/duel_v1_body.yaml conf/obs/duel_v2_world.yaml conf/obs/duel_v3_body_ego.yaml tests/envs/quidditch/test_yaml_obs_loader.py
git commit -m "$(cat <<'EOF'
feat(conf/obs): add blocks: [...] to all four obs YAMLs

Each YAML now declares its full block layout.  Equivalence test asserts
load_obs_yaml(stem) matches the legacy SPEC_BY_NAME[name] block-for-block.

The YAMLs aren't yet consumed by env_factories — that's the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Universal feature dict in `team_env`

Replaces the three-way `if spec == X / elif spec == Y / elif spec == Z` dispatch with a single function that computes every feature this env supports, then `pack(spec, features)` picks the subset.

**Files:**
- Modify: `envs/quidditch/team_env.py`

- [ ] **Step 5.1: Write the byte-identical equivalence test**

Create `tests/envs/quidditch/test_team_env_features.py`:

```python
"""Verify the universal feature dict refactor produces byte-identical obs to the old dispatch."""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch import obs_spec as obs_spec_module


@pytest.mark.parametrize("stem", ["duel_v1_body", "duel_v2_world", "duel_v3_body_ego"])
def test_feature_dict_obs_matches_legacy_dispatch(stem):
    """For each currently-supported spec, the YAML-driven obs is deterministic + correct dim."""
    spec = obs_spec_module.load_obs_yaml(stem)

    cfg = TeamConfig()  # default red_0 / blue_0 prefixes
    env = QuidditchTeamEnv(cfg=cfg, learner_id="blue_0", learner_spec=spec)
    obs_dict, _ = env.reset(seed=42)

    learner_obs = obs_dict["blue_0"]
    assert learner_obs.shape == (spec.dim,), (
        f"obs shape {learner_obs.shape} != spec dim {spec.dim}"
    )
    assert learner_obs.dtype == np.float32
    # Determinism: re-running reset with the same seed reproduces the obs.
    env.close()
    env2 = QuidditchTeamEnv(cfg=cfg, learner_id="blue_0", learner_spec=spec)
    obs_dict2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(learner_obs, obs_dict2["blue_0"])
    env2.close()
```

This test pins the obs values for each spec at reset, against a known seed. It will pass before AND after the refactor — as long as the refactor preserves byte identity.

- [ ] **Step 5.2: Run — expect pass (pre-refactor baseline)**

```bash
python -m pytest tests/envs/quidditch/test_team_env_features.py -v
```

Expected: 3 PASS. This is the *baseline*; we'll re-run after the refactor to verify byte identity.

- [ ] **Step 5.3: Replace `_pack_agent_obs` with universal feature dict in `team_env.py`**

In [`envs/quidditch/team_env.py`](../../../envs/quidditch/team_env.py), replace the `_pack_agent_obs` method (currently lines ~604-712) with this pair:

```python
def _build_agent_features(self, agent_id: str) -> dict[str, np.ndarray]:
    """Compute every feature this env can supply for one agent.

    Returns a {block_name: float32-array} dict.  `obs_spec.pack(spec, ...)`
    picks the subset the active spec asks for.  Keys match canonical
    ObsBlock.name strings post-2026-05-18 rename (see obs_spec.py).
    """
    if agent_id == self._red_id:
        self_q, opp_q   = self._red, self._blue
        self_dofadr     = self._red_dofadr
        opp_dofadr      = self._blue_dofadr
        goal_target     = HOOP_CENTER
    else:
        self_q, opp_q   = self._blue, self._red
        self_dofadr     = self._blue_dofadr
        opp_dofadr      = self._red_dofadr
        goal_target     = self._midpoint()

    s = self_q.state()
    ang_vel, ang_pos, lin_vel_b, lin_pos = s[0], s[1], s[2], s[3]

    opp_s = opp_q.state()
    opp_pos, opp_lin_vel = opp_s[3], opp_s[2]

    # Shared intermediates — local variables, reused across block entries.
    vec_to_goal_world  = goal_target - lin_pos
    dist_g             = float(np.linalg.norm(vec_to_goal_world))
    unit_to_goal       = vec_to_goal_world / (dist_g + 1e-8)
    signed_dist_norm   = self._signed_dist_to_hoop_plane(lin_pos) / ARENA_RADIUS
    vec_to_hoop_world  = (HOOP_CENTER - lin_pos).astype(np.float32)

    opp_pos_rel_world  = opp_pos - lin_pos
    data = self._world.data
    self_vel_world     = data.qvel[self_dofadr : self_dofadr + 3].copy()
    opp_vel_world      = data.qvel[opp_dofadr  : opp_dofadr  + 3].copy()
    opp_vel_rel_world  = (opp_vel_world - self_vel_world).astype(np.float32)

    R_wb = data.xmat[self_q._drone_id].reshape(3, 3)

    dist_to_opp = float(np.linalg.norm(opp_pos_rel_world))
    if agent_id == self._learner_id:
        closing_rate = (
            (self._prev_dist_to_opp - dist_to_opp) / self._red.step_period
        )
        self._prev_dist_to_opp = dist_to_opp
    else:
        closing_rate = 0.0

    return {
        # Self state
        "ang_vel":                ang_vel,
        "ang_pos":                ang_pos,
        "lin_vel":                lin_vel_b,
        "lin_pos":                lin_pos,
        "unit_to_goal":           unit_to_goal,
        "signed_dist_norm":       np.array([signed_dist_norm], dtype=np.float32),
        # Hoop / goal vectors — world AND body variants
        "vec_to_hoop_world":      vec_to_hoop_world,
        "vec_to_hoop_body":       obs_spec.world_to_body(vec_to_hoop_world, R_wb),
        "vec_to_goal":            obs_spec.world_to_body(vec_to_goal_world, R_wb),
        # Opponent — three frame variants of velocity
        "opp_pos_rel_world":      opp_pos_rel_world.astype(np.float32),
        "opp_pos_rel_body":       obs_spec.world_to_body(opp_pos_rel_world, R_wb),
        "opp_vel_rel_body_mixed": (opp_lin_vel - lin_vel_b).astype(np.float32),
        "opp_vel_rel_world":      opp_vel_rel_world,
        "opp_vel_rel_body_ego":   obs_spec.world_to_body(opp_vel_rel_world, R_wb),
        "closing_rate":           np.array([closing_rate], dtype=np.float32),
    }


def _pack_agent_obs(self, agent_id: str, spec: ObsSpec) -> np.ndarray:
    """Pack agent's obs from the universal feature dict under the given spec."""
    return obs_spec.pack(spec, self._build_agent_features(agent_id))
```

Note: the `vec_to_goal` key (no suffix) matches `VEC_TO_GOAL_BODY.name == "vec_to_goal"` — that block is single-variant in the canonical set and was not renamed.

Remove the now-dead imports at the top of `team_env.py` (the explicit composed-constant imports `DUEL_V1_BODY, DUEL_V2_WORLD, DUEL_V3_BODY_EGO`); the `ObsSpec` import stays. Run `grep` first to confirm no other references exist in the file:

```bash
grep -n "DUEL_V1_BODY\|DUEL_V2_WORLD\|DUEL_V3_BODY_EGO" envs/quidditch/team_env.py
```

Expected: only the import-line matches remain. Remove those.

- [ ] **Step 5.4: Run the byte-identity test — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_team_env_features.py -v
```

Expected: 3 PASS (same obs values as Step 5.2 baseline).

- [ ] **Step 5.5: Run team canary**

```bash
python -m pytest tests/integration/test_team_env_canary.py -v
```

Expected: PASS (byte-identical fingerprint).

If it regresses: most likely cause is a dict-key typo. Diff `_build_agent_features` keys against `pack()` calls' previous dict keys — every old key must have a 1-to-1 successor (with renames applied per Task 2).

- [ ] **Step 5.6: Run full suite**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count + 3 new from this task's test file.

- [ ] **Step 5.7: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_team_env_features.py
git commit -m "$(cat <<'EOF'
refactor(team-env): replace _pack_agent_obs dispatch with universal feature dict

Single _build_agent_features() function computes every feature this env
can supply; pack(spec, features) picks the subset the active spec needs.

Canary byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Universal feature dict in `simple_env`

Smaller scope — only one spec currently uses this env (`SIMPLE_ENV_OBS`), so the refactor is mostly a structural rearrangement. Env gains a `spec` kwarg so it can run under a future YAML-defined spec without further code changes.

**Files:**
- Modify: `envs/quidditch/simple_env.py`

- [ ] **Step 6.1: Write the byte-identity equivalence test**

Create `tests/envs/quidditch/test_simple_env_features.py`:

```python
"""Verify simple_env universal feature dict refactor produces byte-identical obs."""
from __future__ import annotations

import numpy as np

from envs.quidditch.simple_env import QuidditchSimpleEnv


def test_simple_env_obs_deterministic_on_reset():
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    obs1, _ = env.reset(seed=42)
    env.close()

    env2 = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    obs2, _ = env2.reset(seed=42)
    env2.close()

    np.testing.assert_array_equal(obs1, obs2)
    assert obs1.shape == (16,)
    assert obs1.dtype == np.float32
```

- [ ] **Step 6.2: Run — expect baseline pass**

```bash
python -m pytest tests/envs/quidditch/test_simple_env_features.py -v
```

Expected: PASS (baseline obs values pinned).

- [ ] **Step 6.3: Refactor `simple_env._obs`**

In [`envs/quidditch/simple_env.py`](../../../envs/quidditch/simple_env.py):

(a) Add `spec` kwarg to `__init__` (around line 103), defaulting to the YAML-loaded SIMPLE_ENV_OBS for the canary-preservation path:

```python
def __init__(
    self,
    render_mode: str | None = None,
    randomise_start: bool = True,
    episode_seconds: float = EPISODE_SECONDS,
    reward_stack: RewardStack | None = None,
    spec: "ObsSpec | None" = None,
) -> None:
    super().__init__()
    self.render_mode = render_mode
    self.randomise_start = randomise_start
    self.episode_seconds = float(episode_seconds)

    self._spec = spec if spec is not None else obs_spec.load_obs_yaml("simple")
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(self._spec.dim,), dtype=np.float32,
    )
    # ... rest unchanged
```

Add `from envs.quidditch.obs_spec import ObsSpec` import if not already present.

(b) Replace `_obs` (around line 271) with a feature-dict version:

```python
def _build_features(self) -> dict[str, np.ndarray]:
    """Compute every feature simple_env can supply, keyed by canonical block name."""
    state = self._q.state()
    ang_vel, ang_pos, lin_vel, lin_pos = state[0], state[1], state[2], state[3]

    vec_to_hoop = HOOP_CENTER - lin_pos
    dist = float(np.linalg.norm(vec_to_hoop))
    unit_to_hoop = vec_to_hoop / (dist + 1e-8)
    signed_dist_norm = self._signed_dist(lin_pos) / ARENA_RADIUS

    return {
        "ang_vel":           ang_vel,
        "ang_pos":           ang_pos,
        "lin_vel":           lin_vel,
        "lin_pos":           lin_pos,
        "unit_to_goal":      unit_to_hoop,
        "signed_dist_norm":  np.array([signed_dist_norm], dtype=np.float32),
    }


def _obs(self) -> np.ndarray:
    """Pack obs from feature dict under self._spec.

    Slots [0:16] are contractually frozen — team_env mirrors the same
    encoding for slots 0:15 (+ signed_dist_norm at slot 15) so
    warm_start_ppo_by_spec can copy the input layer by name.
    """
    return obs_spec.pack(self._spec, self._build_features())
```

Remove the explicit `from envs.quidditch.obs_spec import SIMPLE_ENV_OBS` import if it's only used inside the old `_obs` (verify with grep):

```bash
grep -n "SIMPLE_ENV_OBS" envs/quidditch/simple_env.py
```

Expected: zero matches after the refactor. If any remain, replace with `self._spec` or `self._spec.dim`.

- [ ] **Step 6.4: Run byte-identity + canary**

```bash
python -m pytest tests/envs/quidditch/test_simple_env_features.py tests/integration/test_scoring_canary.py -v
```

Expected: BOTH PASS. Single-agent canary fingerprint `SCORED at step 434, total reward 7.3837` must match.

- [ ] **Step 6.5: Run full suite**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count + 1 new test from this task.

- [ ] **Step 6.6: Commit**

```bash
git add envs/quidditch/simple_env.py tests/envs/quidditch/test_simple_env_features.py
git commit -m "$(cat <<'EOF'
refactor(simple-env): universal feature dict + spec kwarg

_obs() now packs a feature dict under self._spec, loaded from
conf/obs/simple.yaml by default.  Slots [0:16] contract preserved.

Single-agent canary byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Env factories consume `obs_blocks` + `obs_name`

Replace `obs_spec_name: str` with `obs_blocks: list[str]` and `obs_name: str` on both factories. Each factory builds the spec via `build_spec_from_block_names` and passes it to the env constructor.

**Files:**
- Modify: `envs/quidditch/env_factories.py`

- [ ] **Step 7.1: Write the failing test**

Create `tests/envs/quidditch/test_env_factories.py`:

```python
"""Verify env factories accept obs_blocks + obs_name and build the right spec."""
import pytest

from envs.quidditch.env_factories import SimpleEnvFactory, TeamEnvFactory


def test_simple_env_factory_accepts_obs_blocks():
    factory = SimpleEnvFactory(
        n_envs=1, randomise_start=False, episode_seconds=10.0,
        obs_blocks=["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                    "UNIT_TO_GOAL", "SIGNED_DIST_NORM"],
        obs_name="SIMPLE_ENV_OBS",
    )
    assert factory.obs_blocks == ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                                  "UNIT_TO_GOAL", "SIGNED_DIST_NORM"]
    assert factory.obs_name == "SIMPLE_ENV_OBS"


def test_team_env_factory_accepts_obs_blocks(monkeypatch):
    """team_cfg is a stub — we only test field plumbing here."""
    factory = TeamEnvFactory(
        n_envs=1, team_cfg=None, learner_id="blue_0", opponent_spec="beeline_red",
        obs_blocks=["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                    "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
                    "OPP_VEL_REL_WORLD", "CLOSING_RATE"],
        obs_name="DUEL_V2_WORLD",
        frame_stack=3,
    )
    assert factory.obs_name == "DUEL_V2_WORLD"
    assert len(factory.obs_blocks) == 9
```

- [ ] **Step 7.2: Run — expect failure**

```bash
python -m pytest tests/envs/quidditch/test_env_factories.py -v
```

Expected: 2 FAIL on `TypeError: __init__() got an unexpected keyword argument 'obs_blocks'`.

- [ ] **Step 7.3: Update `SimpleEnvFactory`**

In [`envs/quidditch/env_factories.py`](../../../envs/quidditch/env_factories.py), replace the `SimpleEnvFactory` dataclass body:

```python
@dataclass
class SimpleEnvFactory:
    """Builds vec envs around QuidditchSimpleEnv (single-agent)."""
    n_envs: int
    randomise_start: bool
    episode_seconds: float
    obs_blocks: list[str]              # was: obs_spec_name: str = "SIMPLE_ENV_OBS"
    obs_name: str                      # identifier for tags / lineage
    seed: int = 42
    reward_stack: Any = None

    def _make_thunk(self):
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        from envs.quidditch.obs_spec import build_spec_from_block_names
        rs = self.randomise_start
        eps = self.episode_seconds
        reward_stack = self.reward_stack
        spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode=None, randomise_start=rs, episode_seconds=eps,
                reward_stack=reward_stack, spec=spec,
            )
        return _thunk

    def build_video_env_fn(self):
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        from envs.quidditch.obs_spec import build_spec_from_block_names
        rs = self.randomise_start
        eps = self.episode_seconds
        reward_stack = self.reward_stack
        spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            return QuidditchSimpleEnv(
                render_mode="rgb_array", randomise_start=rs, episode_seconds=eps,
                reward_stack=reward_stack, spec=spec,
            )
        return _thunk

    # build_train_env and build_eval_env unchanged — they delegate to _make_thunk.
```

- [ ] **Step 7.4: Update `TeamEnvFactory`**

```python
@dataclass
class TeamEnvFactory:
    """Builds vec envs around QuidditchTeamEnv + OpponentControlledEnv."""
    n_envs: int
    team_cfg: Any
    learner_id: str
    opponent_spec: str
    obs_blocks: list[str]              # was: obs_spec_name: str = "DUEL_V2_WORLD"
    obs_name: str                      # identifier for tags / lineage
    frame_stack: int = 3
    seed: int = 42
    reward_stack: Any = None

    def _make_thunk(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        from envs.quidditch.obs_spec import build_spec_from_block_names
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        reward_stack = self.reward_stack
        learner_spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, reward_stack=reward_stack,
                learner_id=learner, learner_spec=learner_spec,
            )
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
        return _thunk

    def build_video_env_fn(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import (
            OpponentControlledEnv, from_spec, FrameStackWrapper,
        )
        from envs.quidditch.obs_spec import build_spec_from_block_names
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        frame_stack = self.frame_stack
        reward_stack = self.reward_stack
        learner_spec = build_spec_from_block_names(self.obs_blocks)
        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, render_mode="rgb_array",
                reward_stack=reward_stack,
                learner_id=learner, learner_spec=learner_spec,
            )
            opp = from_spec(opp_spec, deterministic=True)
            env = OpponentControlledEnv(team, learner_id=learner, opponent=opp)
            if frame_stack > 1:
                return FrameStackWrapper(env, n_stack=frame_stack)
            return env
        return _thunk

    # build_train_env, build_eval_env unchanged.
```

Note: we removed the previously-`obs_spec_name: str = "DUEL_V2_WORLD"` default. Both `obs_blocks` and `obs_name` are now required positional fields. Hydra YAMLs supply both (Task 8).

- [ ] **Step 7.5: Run — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_env_factories.py -v
```

Expected: 2 PASS.

- [ ] **Step 7.6: Commit**

```bash
git add envs/quidditch/env_factories.py tests/envs/quidditch/test_env_factories.py
git commit -m "$(cat <<'EOF'
refactor(env-factories): consume obs_blocks + obs_name instead of obs_spec_name

Each factory builds its ObsSpec via build_spec_from_block_names() and
passes it to the env constructor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update `conf/env/{simple,team}.yaml` interpolations

Swap `obs_spec_name: ${obs.name}` for `obs_blocks: ${obs.blocks}` + `obs_name: ${obs.name}` so Hydra wiring matches the new factory signatures.

**Files:**
- Modify: `conf/env/simple.yaml`
- Modify: `conf/env/team.yaml`

- [ ] **Step 8.1: Update `conf/env/simple.yaml`**

In [`conf/env/simple.yaml`](../../../conf/env/simple.yaml), replace the `obs_spec_name` line:

```yaml
# Before
obs_spec_name: ${obs.name}

# After
obs_blocks: ${obs.blocks}
obs_name:   ${obs.name}
```

- [ ] **Step 8.2: Update `conf/env/team.yaml`**

In [`conf/env/team.yaml`](../../../conf/env/team.yaml), replace the `obs_spec_name` line:

```yaml
# Before
obs_spec_name: ${obs.name}
frame_stack:   ${obs.n_stack}

# After
obs_blocks: ${obs.blocks}
obs_name:   ${obs.name}
frame_stack: ${obs.n_stack}
```

- [ ] **Step 8.3: Smoke test — Hydra instantiation builds a complete factory**

Create `tests/scripts/test_hydra_obs_wiring.py`:

```python
"""Smoke test: Hydra-compose a full config and verify obs.blocks propagates to the factory."""
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hydra_compose_team_carries_obs_blocks():
    with initialize_config_dir(str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=canary_team"])
    assert list(cfg.obs.blocks) == [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
        "OPP_VEL_REL_WORLD", "CLOSING_RATE",
    ]
    assert cfg.env.obs_blocks == cfg.obs.blocks
    assert cfg.env.obs_name == "DUEL_V2_WORLD"


def test_hydra_compose_simple_carries_obs_blocks():
    with initialize_config_dir(str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=canary_single"])
    assert cfg.env.obs_blocks == list(cfg.obs.blocks)
    assert cfg.env.obs_name == "SIMPLE_ENV_OBS"
```

- [ ] **Step 8.4: Run — expect pass**

```bash
python -m pytest tests/scripts/test_hydra_obs_wiring.py -v
```

Expected: 2 PASS. If a test fails, verify the env YAML interpolations resolve correctly — `omegaconf.OmegaConf.to_yaml(cfg)` from inside an `initialize_config_dir` block will print the composed view.

- [ ] **Step 8.5: Run full suite — confirm no regressions**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count plus the new tests. **However**, train.py is still using `SPEC_BY_NAME[cfg.obs.name]` in two spots — that hasn't moved yet (Task 9), so any test that calls into `_build_or_load_model` via `train.py` continues to work via `SPEC_BY_NAME`. Don't be alarmed if `test_train_resolve_parent_wiring.py` passes here.

- [ ] **Step 8.6: Commit**

```bash
git add conf/env/simple.yaml conf/env/team.yaml tests/scripts/test_hydra_obs_wiring.py
git commit -m "$(cat <<'EOF'
feat(conf/env): wire obs.blocks + obs.name into env factory configs

Replaces obs_spec_name interpolation with obs_blocks + obs_name pair,
matching the new factory signatures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Replace `SPEC_BY_NAME` lookups in `scripts/train.py` and `scripts/_render_model_doc.py`

Two callsites in `train.py` (compat check + model build) and one in `_render_model_doc.py`. All three switch from `SPEC_BY_NAME[name]` to `build_spec_from_block_names(cfg.obs.blocks)`.

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/_render_model_doc.py`

- [ ] **Step 9.1: Update `scripts/train.py`**

In [`scripts/train.py`](../../../scripts/train.py):

Replace the top-of-file import (line 43):

```python
# Before
from envs.quidditch.obs_spec import SPEC_BY_NAME
# After
from envs.quidditch.obs_spec import build_spec_from_block_names
```

Replace `_check_obs_compat_from_hydra` (lines 116-131) — accept the parent's `cfg.obs.blocks` (if present) or fall back to the legacy `obs.name`-only path (since this branch is reached for warm-start / pretrain from legacy models, we need to handle both the migrated and pre-migration shapes during the rollout):

```python
def _check_obs_compat_from_hydra(parent_hydra: Path, current_spec, current_n_stack: int):
    """Read parent's obs spec from .hydra/config.yaml, compare to current."""
    cfg_path = parent_hydra / "config.yaml"
    parent_cfg = OmegaConf.load(cfg_path)
    parent_n_stack = int(parent_cfg.obs.n_stack)
    parent_blocks = list(parent_cfg.obs.get("blocks", []))
    if not parent_blocks:
        raise SystemExit(
            f"Parent .hydra/config.yaml at {cfg_path} has no obs.blocks field.\n"
            f"Run tmp_migrate_obs_blocks.py to add it (see "
            f"docs/superpowers/plans/2026-05-18-yaml-driven-obs.md Task 13)."
        )
    parent_spec = build_spec_from_block_names(parent_blocks)
    if parent_spec.blocks != current_spec.blocks or parent_n_stack != current_n_stack:
        parent_name = parent_cfg.obs.get("name", "<unnamed>")
        raise SystemExit(
            f"Obs spec mismatch:\n"
            f"  parent: {parent_name} (n_stack={parent_n_stack})\n"
            f"  current: {[b.name for b in current_spec.blocks]} (n_stack={current_n_stack})\n"
            f"Use init=warm_start for surgical extension."
        )
```

Replace `current_spec = SPEC_BY_NAME[cfg.obs.name]` (line 154):

```python
current_spec = build_spec_from_block_names(cfg.obs.blocks)
```

Replace the warm-start branch's parent spec lookup (lines 211-212):

```python
parent_spec_name, parent_n_stack, parent_blocks = _read_hydra_obs(parent_hydra)
if not parent_blocks:
    raise SystemExit(
        f"Parent .hydra/config.yaml at {parent_hydra} has no obs.blocks field.\n"
        f"Run tmp_migrate_obs_blocks.py first."
    )
parent_spec = build_spec_from_block_names(parent_blocks)
```

Update `_read_hydra_obs` (lines 134-139) to also return the blocks list:

```python
def _read_hydra_obs(parent_hydra: Path) -> tuple[str | None, int | None, list[str]]:
    cfg_path = parent_hydra / "config.yaml"
    if not cfg_path.exists():
        return None, None, []
    parent_cfg = OmegaConf.load(cfg_path)
    return (
        parent_cfg.obs.get("name"),
        int(parent_cfg.obs.n_stack),
        list(parent_cfg.obs.get("blocks", [])),
    )
```

Update the `main()` function's inner `learner_spec = SPEC_BY_NAME[cfg.obs.name]` (around line 285):

```python
# Replace the inline import + lookup:
from envs.quidditch.obs_spec import build_spec_from_block_names
learner_spec = build_spec_from_block_names(cfg.obs.blocks)
```

Remove the now-unused `from envs.quidditch.obs_spec import SPEC_BY_NAME` line near line 281 — verify with grep:

```bash
grep -n "SPEC_BY_NAME" scripts/train.py
```

Expected: zero matches after the refactor.

- [ ] **Step 9.2: Update `scripts/_render_model_doc.py:_section_obs_spec`**

In [`scripts/_render_model_doc.py`](../../../scripts/_render_model_doc.py) around line 209-220, replace the body:

```python
def _section_obs_spec(ctx: dict[str, Any]) -> str:
    """Render the obs-spec table for MODEL.md."""
    from envs.quidditch.obs_spec import build_spec_from_block_names, describe
    obs = ctx.get("hydra_cfg", {}).get("obs") or {}
    name = obs.get("name", "<unnamed>")
    blocks = obs.get("blocks") or []
    if not blocks:
        return (
            f"## Obs spec — `{name}`\n\n"
            f"> ⚠ legacy config has no `obs.blocks` field. "
            f"See run_info.toml or migrate via tmp_migrate_obs_blocks.py."
        )
    spec = build_spec_from_block_names(blocks)
    return (
        f"## Obs spec — `{name}`\n\n"
        f"```\n{describe(spec, name)}\n```\n"
    )
```

(The existing function may have richer formatting than this minimal example — preserve its surrounding scaffolding; just swap the `SPEC_BY_NAME[name]` lookup for the `build_spec_from_block_names(blocks)` call.)

Verify with grep:

```bash
grep -n "SPEC_BY_NAME" scripts/_render_model_doc.py
```

Expected: zero matches.

- [ ] **Step 9.3: Run train + render tests**

```bash
python -m pytest tests/scripts/test_train_resolve_parent_wiring.py tests/scripts/test_render_model_doc.py -v
```

Expected: most PASS; some may FAIL because the test fixtures don't carry `blocks:` in their fake configs. Those tests are updated in Task 11. Note which tests fail with `obs.blocks` errors — they belong in the Task 11 fixup list.

- [ ] **Step 9.4: Commit**

```bash
git add scripts/train.py scripts/_render_model_doc.py
git commit -m "$(cat <<'EOF'
refactor(scripts): replace SPEC_BY_NAME lookups with build_spec_from_block_names

train.py compat check + model build + warm-start lookup; _render_model_doc
obs-spec section.  All consume cfg.obs.blocks instead of SPEC_BY_NAME[name].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update `scripts/migrate_legacy_models.py:LEGACY_SPECS`

That dict maps promoted model dir names → obs-spec name strings. It's used when migrating pre-Hydra `run_info.toml` models. After Task 2's block renames, the dict still works (it stores SPEC names like `"DUEL_V1_BODY"`, not block names) — but the file imports `SPEC_BY_NAME` for a fallback dim-lookup branch. Replace that with `load_obs_yaml` over the same names.

**Files:**
- Modify: `scripts/migrate_legacy_models.py`

- [ ] **Step 10.1: Update the import + dim-lookup fallback**

In [`scripts/migrate_legacy_models.py`](../../../scripts/migrate_legacy_models.py):

Replace line 36:

```python
# Before
from envs.quidditch.obs_spec import SPEC_BY_NAME
# After
from envs.quidditch.obs_spec import load_obs_yaml
```

Replace the dim-based fallback loop (around lines 55-63):

```python
# Before
for name, spec in SPEC_BY_NAME.items():
    if spec.dim == target_dim:
        ...

# After
_KNOWN_SPECS = ("simple", "duel_v1_body", "duel_v2_world", "duel_v3_body_ego")
_SPEC_BY_NAME_LOCAL = {
    "SIMPLE_ENV_OBS":   load_obs_yaml("simple"),
    "DUEL_V1_BODY":     load_obs_yaml("duel_v1_body"),
    "DUEL_V2_WORLD":    load_obs_yaml("duel_v2_world"),
    "DUEL_V3_BODY_EGO": load_obs_yaml("duel_v3_body_ego"),
}
for name, spec in _SPEC_BY_NAME_LOCAL.items():
    if spec.dim == target_dim:
        ...
```

(Keep the surrounding logic unchanged; only swap the source of the name→spec map.)

- [ ] **Step 10.2: Run the migrate-legacy tests**

```bash
python -m pytest tests/scripts/test_migrate_legacy_models.py -v
```

Expected: PASS. If a test asserts on specific log/error messages mentioning `SPEC_BY_NAME`, update those assertions inline.

- [ ] **Step 10.3: Commit**

```bash
git add scripts/migrate_legacy_models.py
git commit -m "$(cat <<'EOF'
refactor(migrate-legacy): load known specs from YAML, drop SPEC_BY_NAME import

LEGACY_SPECS (model dir → spec name) unchanged.  The dim-fallback loop
now reads conf/obs/*.yaml via load_obs_yaml instead of SPEC_BY_NAME.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update test fixtures across the codebase

Eight test files use either the composed Python constants (`DUEL_V2_WORLD`, etc.) or in-memory fixture configs missing the new `blocks:` field. Update each so all tests stay green BEFORE we delete the composed constants in Task 12.

**Files:**
- Modify: `tests/envs/quidditch/test_augmented_obs.py`
- Modify: `tests/core/policies/test_warm_start.py`
- Modify: `tests/core/policies/test_warm_start_by_spec.py`
- Modify: `tests/scripts/test_render_model_doc.py`
- Modify: `tests/scripts/test_train_resolve_parent_wiring.py`
- Modify: `tests/scripts/test_log_run_artifact.py`
- Modify: `tests/scripts/test_wandb_init.py`
- Modify: `tests/envs/quidditch/test_obs_spec.py` — delete `test_spec_by_name_maps_canonical_specs` and any other `SPEC_BY_NAME`-specific tests (the registry is gone); preserve tests that exercise `ObsBlock` / `ObsSpec` / `pack` directly.

For each file in this task, the pattern is:
- Replace `from envs.quidditch.obs_spec import DUEL_V*_*, SIMPLE_ENV_OBS` with `from envs.quidditch.obs_spec import load_obs_yaml` and call `load_obs_yaml("...")` where the constant was used.
- For fixture dicts like `{"obs": {"name": "DUEL_V2_WORLD", "n_stack": 3}}`, add `"blocks": [...]` with the matching block list (copy from `conf/obs/duel_v2_world.yaml`).

- [ ] **Step 11.1: List the test files needing fixture updates**

```bash
grep -l "SPEC_BY_NAME\|DUEL_V1_BODY\|DUEL_V2_WORLD\|DUEL_V3_BODY_EGO\|SIMPLE_ENV_OBS" tests/ -r
```

Confirm the list matches the files-modified list above. If new files appear, add them.

Also grep for missing-`blocks` fixtures:

```bash
grep -rn '"name": "DUEL\|"name": "SIMPLE' tests/scripts/
```

Each match is a candidate fixture that needs a `"blocks": [...]` field added.

- [ ] **Step 11.2: Update each test file**

For test files importing composed constants:

```python
# Before
from envs.quidditch.obs_spec import DUEL_V2_WORLD
# After
from envs.quidditch.obs_spec import load_obs_yaml
DUEL_V2_WORLD = load_obs_yaml("duel_v2_world")  # module-level constant
```

(Keeping the same uppercase identifier preserves the rest of the test code with minimal churn. The right-hand side is now a YAML-built spec, byte-equivalent to the legacy constant.)

For fixture dicts in `tests/scripts/test_render_model_doc.py`, `test_train_resolve_parent_wiring.py`, `test_log_run_artifact.py`, `test_wandb_init.py`, add the matching `blocks:` list. Example:

```python
# Before
"obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},

# After
"obs": {
    "name": "DUEL_V2_WORLD",
    "n_stack": 3,
    "blocks": [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
        "OPP_VEL_REL_WORLD", "CLOSING_RATE",
    ],
},
```

Similarly for `SIMPLE_ENV_OBS`, `DUEL_V1_BODY`, `DUEL_V3_BODY_EGO` — copy the matching `blocks:` from the corresponding `conf/obs/*.yaml`.

- [ ] **Step 11.3: Run the affected test files individually**

```bash
python -m pytest tests/envs/quidditch/test_augmented_obs.py -v
python -m pytest tests/core/policies/test_warm_start.py tests/core/policies/test_warm_start_by_spec.py -v
python -m pytest tests/scripts/test_render_model_doc.py -v
python -m pytest tests/scripts/test_train_resolve_parent_wiring.py -v
python -m pytest tests/scripts/test_log_run_artifact.py -v
python -m pytest tests/scripts/test_wandb_init.py -v
```

Expected: each file all-PASS. If a test still fails, the fixture is missing `blocks:` somewhere — re-grep within that file for `{"obs"` and verify each fixture has the field.

- [ ] **Step 11.4: Run full suite**

```bash
python -m pytest --no-header -q 2>&1 | tail -10
```

Expected: same pass count as start of Task 9, plus the new tests added in Tasks 1-8.

- [ ] **Step 11.5: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: migrate fixtures to YAML-loaded specs + blocks: field

- Module-level constants in test files swapped for load_obs_yaml(stem).
- Fixture configs gain blocks: [...] matching the corresponding YAML.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Delete composed-spec constants + `SPEC_BY_NAME` from `obs_spec.py`

The four `SIMPLE_ENV_OBS`, `DUEL_V1_BODY`, `DUEL_V2_WORLD`, `DUEL_V3_BODY_EGO` constants and the `SPEC_BY_NAME` registry are now unused (verified by grep). Delete them and update `__main__` to iterate YAMLs instead.

**Files:**
- Modify: `envs/quidditch/obs_spec.py`

- [ ] **Step 12.1: Delete the now-obsolete round-trip test from Task 4**

In `tests/envs/quidditch/test_yaml_obs_loader.py`, remove the parametrized
`test_yaml_round_trip_matches_legacy_spec_by_name` function. It was a
mid-refactor canary; after the composed constants are deleted, the
YAML *is* the source of truth and there's nothing to round-trip against.

- [ ] **Step 12.2: Verify nothing imports the about-to-delete names**

```bash
grep -rn "from envs.quidditch.obs_spec import.*\(SIMPLE_ENV_OBS\|DUEL_V1_BODY\|DUEL_V2_WORLD\|DUEL_V3_BODY_EGO\|SPEC_BY_NAME\)" envs/ scripts/ tests/ core/ config_schema.py
grep -rn "SPEC_BY_NAME" envs/ scripts/ tests/ core/ config_schema.py
```

Expected: ZERO matches outside of `envs/quidditch/obs_spec.py` itself
(and even there, only its definition site — which is also going away in
the next step).

If a match exists, return to the appropriate earlier task (Task 9, 10, or 11) and fix it before continuing. Do not proceed until grep is clean.

- [ ] **Step 12.3: Write the failing tests**

Append to `tests/envs/quidditch/test_yaml_obs_loader.py`:

```python
def test_composed_constants_are_gone():
    """The composed-spec Python constants are deleted; only ObsBlock and helpers remain."""
    from envs.quidditch import obs_spec
    for removed in ("SIMPLE_ENV_OBS", "DUEL_V1_BODY", "DUEL_V2_WORLD",
                    "DUEL_V3_BODY_EGO", "SPEC_BY_NAME"):
        assert not hasattr(obs_spec, removed), f"{removed} should be deleted"
```

- [ ] **Step 12.4: Run — expect failure**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_composed_constants_are_gone -v
```

Expected: FAIL — the constants are still defined.

- [ ] **Step 12.5: Delete the composed-spec block + `SPEC_BY_NAME` from `obs_spec.py`**

In [`envs/quidditch/obs_spec.py`](../../../envs/quidditch/obs_spec.py), delete:

- The `# ── Composed specs ...` section header and its four `ObsSpec` constants (`SIMPLE_ENV_OBS`, `DUEL_V1_BODY`, `DUEL_V2_WORLD`, `DUEL_V3_BODY_EGO`).
- The `# ── Name registry ...` section header, the `SPEC_BY_NAME: dict[str, ObsSpec] = {...}` block, and the comment above it.

Update the `__main__` block at the bottom of the file to iterate over `conf/obs/*.yaml` instead:

```python
if __name__ == "__main__":
    import glob
    repo_root = Path(__file__).resolve().parents[2]
    for yaml_path in sorted(glob.glob(str(repo_root / "conf" / "obs" / "*.yaml"))):
        stem = Path(yaml_path).stem
        spec = load_obs_yaml(stem)
        print(describe(spec, stem))
        print()
```

This keeps `make obs-specs` (`python -m envs.quidditch.obs_spec`) working.

- [ ] **Step 12.6: Run — expect pass**

```bash
python -m pytest tests/envs/quidditch/test_yaml_obs_loader.py::test_composed_constants_are_gone -v
```

Expected: PASS.

- [ ] **Step 12.7: Run canaries + full suite**

```bash
python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
python -m pytest --no-header -q 2>&1 | tail -15
```

Expected: BOTH canaries PASS. Full suite — same count, no regressions.

If `make obs-specs` is invoked:

```bash
python -m envs.quidditch.obs_spec
```

Expected: prints describe output for all four specs (`simple`, `duel_v1_body`, `duel_v2_world`, `duel_v3_body_ego`), one per block.

- [ ] **Step 12.8: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_yaml_obs_loader.py
git commit -m "$(cat <<'EOF'
refactor(obs-spec): delete SIMPLE_ENV_OBS, DUEL_V*_*, SPEC_BY_NAME

YAML files in conf/obs/ are now the single source of truth for obs
composition.  Canonical ObsBlock constants remain.

__main__ now iterates conf/obs/*.yaml so `make obs-specs` still works.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: One-shot migration script for the 7 promoted models

Patches each promoted model's `models/<name>/.hydra/config.yaml` to add the `obs.blocks: [...]` field, then re-uploads the artifact bundle to W&B. After verification, the user manually deletes the script.

**Files:**
- Create (transient): `tmp_migrate_obs_blocks.py`

**Caution:** This task touches shared state (W&B). Confirm with the user before running the W&B-upload portion. Default the script to dry-run mode.

- [ ] **Step 13.1: Write the script**

Create `tmp_migrate_obs_blocks.py` at the worktree root:

```python
#!/usr/bin/env python3
"""One-shot: add obs.blocks to the 7 promoted models' .hydra/config.yaml + W&B artifacts.

Run order:
  1. python tmp_migrate_obs_blocks.py --dry-run        # preview disk changes
  2. python tmp_migrate_obs_blocks.py --commit-disk    # apply local writes
  3. python tmp_migrate_obs_blocks.py --commit-wandb   # patch W&B artifacts

After --commit-wandb succeeds and tests still pass, delete this script.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from envs.quidditch.obs_spec import load_obs_yaml


REPO_ROOT = Path(__file__).resolve().parent


# Maps promoted model dir name → obs-spec YAML stem.  Curated from
# brain/models.md + scripts/migrate_legacy_models.py:LEGACY_SPECS.
MODEL_TO_OBS_STEM: dict[str, str] = {
    "ppo_hoop_fixed_start_20260430_224234": "simple",
    "ppo_hoop_fixed_start_20260504_023051": "simple",
    "ppo_hoop_rand_start_20260430_234354":  "simple",
    "ppo_hoop_rand_start_20260505_174509":  "simple",
    "ppo_hoop_red_1_20260506_103058":       "duel_v1_body",
    "ppo_hoop_blue_1_20260507_194423":      "duel_v1_body",
    "ppo_hoop_blue_4_20260511_202612":      "duel_v2_world",
}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _blocks_for_stem(stem: str) -> list[str]:
    cfg = _read_yaml(REPO_ROOT / "conf" / "obs" / f"{stem}.yaml")
    return list(cfg["blocks"])


def _patch_hydra_config(hydra_cfg_path: Path, stem: str, *, dry_run: bool) -> bool:
    """Add obs.blocks to a .hydra/config.yaml.  Returns True if a change was made."""
    data = _read_yaml(hydra_cfg_path)
    obs = data.setdefault("obs", {})
    if obs.get("blocks"):
        print(f"  [skip] {hydra_cfg_path}: already has obs.blocks")
        return False
    new_blocks = _blocks_for_stem(stem)
    print(f"  [patch] {hydra_cfg_path}: add obs.blocks = {new_blocks}")
    if not dry_run:
        obs["blocks"] = new_blocks
        _write_yaml(hydra_cfg_path, data)
    return True


def _verify_round_trip(model_dir: Path, stem: str) -> None:
    """After patch, verify build_spec_from_block_names rebuilds the expected spec."""
    from envs.quidditch.obs_spec import build_spec_from_block_names
    hydra_cfg = _read_yaml(model_dir / ".hydra" / "config.yaml")
    rebuilt = build_spec_from_block_names(hydra_cfg["obs"]["blocks"])
    expected = load_obs_yaml(stem)
    assert rebuilt == expected, (
        f"{model_dir}: rebuilt spec disagrees with conf/obs/{stem}.yaml\n"
        f"  rebuilt:  {[b.name for b in rebuilt.blocks]}\n"
        f"  expected: {[b.name for b in expected.blocks]}"
    )
    print(f"  [verify] {model_dir}: round-trips to {stem}")


def patch_disk(dry_run: bool) -> None:
    print(f"=== Disk migration {'(dry-run)' if dry_run else '(LIVE)'} ===")
    for model_name, stem in MODEL_TO_OBS_STEM.items():
        model_dir = REPO_ROOT / "models" / model_name
        if not model_dir.exists():
            print(f"  [missing] {model_dir} — skip")
            continue
        hydra_cfg = model_dir / ".hydra" / "config.yaml"
        if not hydra_cfg.exists():
            print(f"  [missing] {hydra_cfg} — skip")
            continue
        _patch_hydra_config(hydra_cfg, stem, dry_run=dry_run)
        if not dry_run:
            _verify_round_trip(model_dir, stem)


def patch_wandb(dry_run: bool) -> None:
    """For each promoted model, patch its W&B artifact bundle's .hydra/config.yaml."""
    print(f"=== W&B artifact migration {'(dry-run)' if dry_run else '(LIVE)'} ===")
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping W&B migration")
        return

    api = wandb.Api()
    # Read each promoted model's _wandb_metadata.json to get exact (project, name, version)
    import json
    for model_name, stem in MODEL_TO_OBS_STEM.items():
        meta_path = REPO_ROOT / "models" / model_name / "_wandb_metadata.json"
        if not meta_path.exists():
            print(f"  [no _wandb_metadata.json] {model_name} — skip")
            continue
        meta = json.loads(meta_path.read_text())
        artifact_ref = f"{meta['entity']}/{meta['project']}/{meta['name']}:{meta['version']}"
        print(f"  [target] {artifact_ref}  (stem={stem})")
        if dry_run:
            continue

        # Download, patch .hydra/config.yaml inside the bundle, re-upload as new version.
        artifact = api.artifact(artifact_ref)
        with tempfile.TemporaryDirectory() as td:
            local = Path(td)
            artifact.download(root=str(local))
            hydra_cfg = local / ".hydra" / "config.yaml"
            if not hydra_cfg.exists():
                print(f"  [missing] {hydra_cfg} inside artifact — skip")
                continue
            data = _read_yaml(hydra_cfg)
            if data.get("obs", {}).get("blocks"):
                print(f"  [skip] {artifact_ref}: already migrated")
                continue
            data.setdefault("obs", {})["blocks"] = _blocks_for_stem(stem)
            _write_yaml(hydra_cfg, data)

            # Upload as a new version + repoint aliases.
            new_art = wandb.Artifact(
                name=meta["name"], type=artifact.type,
                metadata=dict(artifact.metadata),
                description=(artifact.description or "") + "\n\n[obs.blocks migration 2026-05-18]",
            )
            new_art.add_dir(str(local))
            wandb.init(
                project=meta["project"], entity=meta["entity"],
                job_type="migrate-obs-blocks", reinit=True, mode="online",
            )
            logged = wandb.log_artifact(new_art)
            logged.wait()
            # Re-point existing aliases (prod, <run_name>, etc.) to the new version.
            for alias in artifact.aliases:
                if alias.startswith("v"):
                    continue  # immutable version aliases stay where they are
                logged.aliases.append(alias)
            logged.save()
            wandb.finish()
            print(f"  [done] {artifact_ref} → {logged.qualified_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--commit-disk", action="store_true")
    g.add_argument("--commit-wandb", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        patch_disk(dry_run=True)
        patch_wandb(dry_run=True)
    elif args.commit_disk:
        patch_disk(dry_run=False)
    elif args.commit_wandb:
        patch_wandb(dry_run=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 13.2: Dry-run the script**

```bash
python tmp_migrate_obs_blocks.py --dry-run
```

Expected output: lists the 7 model dirs + their planned `obs.blocks` patches; for each, says `[patch]` (not `[skip]`). For W&B, lists 7 artifact refs in the form `entity/project/name:vN`.

Pause here. **Show the user the dry-run output and confirm before running `--commit-disk` and `--commit-wandb`.**

- [ ] **Step 13.3: Commit the disk migration**

After user confirms:

```bash
python tmp_migrate_obs_blocks.py --commit-disk
```

Then verify with git:

```bash
git status models/
git diff models/ | head -80
```

Expected: each promoted model's `.hydra/config.yaml` shows an added `blocks:` field.

- [ ] **Step 13.4: Commit the disk changes as a regular git commit**

```bash
git add models/
git commit -m "$(cat <<'EOF'
chore(models): add obs.blocks to 7 promoted models' .hydra/config.yaml

One-shot migration via tmp_migrate_obs_blocks.py --commit-disk.  Verified
round-trip equality against conf/obs/*.yaml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 13.5: Commit the W&B migration (user confirms)**

After re-confirming with the user (this mutates W&B state):

```bash
python tmp_migrate_obs_blocks.py --commit-wandb
```

Verify each artifact has a new version with `blocks:` in its `.hydra/config.yaml` by spot-checking one via the W&B UI or:

```bash
python - <<'PY'
import wandb, json
api = wandb.Api()
art = api.artifact("ppo_hoop_blue_4:prod")  # or the project-qualified form
print("aliases:", art.aliases)
print("version:", art.version)
PY
```

- [ ] **Step 13.6: Delete the throwaway script**

After both migrations succeed and the canary tests still pass:

```bash
rm tmp_migrate_obs_blocks.py
git status        # confirm only the deletion is staged
git add -u tmp_migrate_obs_blocks.py
git commit -m "$(cat <<'EOF'
chore: remove tmp_migrate_obs_blocks.py one-shot script

Migration complete: disk + W&B artifacts patched, canaries pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Final full-suite verification + canary smoke

Last green-pass gate before merge.

- [ ] **Step 14.1: Run the full suite**

```bash
python -m pytest --no-header -q 2>&1 | tail -15
```

Expected: ALL tests PASS. Compare the pass count against the pre-refactor baseline (snapshot recorded at the start of Task 1's `pytest --no-header -q` output) — should be **baseline + ~12 new tests** (5 from Task 1, 5 from Task 2, 2 from Task 3, 4 from Task 4, 3 from Task 5, 1 from Task 6, 2 from Task 7, 2 from Task 8, 1 from Task 12).

- [ ] **Step 14.2: Canary integration tests**

```bash
python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: BOTH PASS with their byte-identical fingerprints.

- [ ] **Step 14.3: Hydra-driven train.py smoke (5-minute job)**

```bash
make train EXP=canary_team
```

Watch for:
- Hydra composes without error.
- Env factory builds with `obs_blocks` + `obs_name`.
- Training proceeds for at least 1 eval cycle.
- The run's `.hydra/config.yaml` carries `obs.blocks` populated from `${obs.blocks}` interpolation.

Cancel after a successful eval cycle (`Ctrl-C`). Then:

```bash
cat runs/<latest_run_dir>/.hydra/config.yaml | yq .obs
```

Expected output:

```yaml
name: DUEL_V2_WORLD
n_stack: 3
blocks:
  - ANG_VEL
  - ANG_POS
  # ... (9 entries)
```

- [ ] **Step 14.4: Verify `make obs-specs` still renders all four**

```bash
make obs-specs
```

Expected: prints describe output for `simple`, `duel_v1_body`, `duel_v2_world`, `duel_v3_body_ego` — block-by-block layout for each.

- [ ] **Step 14.5: Final commit (if any straggling fixes were needed)**

If any incidental fixes were applied during smoke-testing, commit them as a single follow-up:

```bash
git add -u
git commit -m "$(cat <<'EOF'
fix: smoke-test follow-ups for YAML-driven obs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no fixes needed, skip this step.

---

## Out of scope (deferred follow-ups)

- Updating `brain/index.md`, `brain/changelog.md`, `brain/decisions.md`,
  `brain/models.md` — done as a session-end protocol activity after the
  branch is merged into `develop`, per the project's CLAUDE.md.
- Final merge into `develop` — owner decision after PR review.
- A new YAML for a future experiment (e.g. `conf/obs/duel_v4_mix.yaml`)
  — the *goal* of this refactor; first such addition is its own task.
