# YAML-driven obs composition

**Status:** draft (2026-05-18)
**Branch:** `feature/yaml-driven-obs`
**Supersedes:** the `SPEC_BY_NAME` registry + named composed `ObsSpec` constants
introduced in [2026-05-12-obs-spec-design.md](2026-05-12-obs-spec-design.md).

## Motivation

Today, adding a new observation layout for an experiment requires editing
Python in two places:

1. Declaring a new composed constant (`DUEL_V4_MIX = ObsSpec((...))`) in
   [`envs/quidditch/obs_spec.py`](../../../envs/quidditch/obs_spec.py).
2. Adding a `name → ObsSpec` entry to `SPEC_BY_NAME` in the same file.
3. Adding a matching dispatch branch (`if spec == DUEL_V4_MIX:`) in
   [`envs/quidditch/team_env.py::_pack_agent_obs`](../../../envs/quidditch/team_env.py)
   that knows how to compute the new layout from simulator state.

The Hydra config layer already has `conf/obs/*.yaml` — but those YAMLs are
just pointer files (`name: DUEL_V2_WORLD`) that resolve back through
`SPEC_BY_NAME`. They carry no structural information.

Goal of this change: a new obs layout for an experiment is **one new YAML
file**, no Python edits.

## Goals / non-goals

**Goals**

- Composing canonical `ObsBlock`s into a new `ObsSpec` is a pure config
  change.
- Each env (simple_env, team_env) computes the same set of features it
  already supports; YAML chooses *which subset, in what order*.
- Lineage / W&B tags / MODEL.md surfaces (which key off `cfg.obs.name`)
  keep working unchanged.
- The four currently-used composed specs (`SIMPLE_ENV_OBS`,
  `DUEL_V1_BODY`, `DUEL_V2_WORLD`, `DUEL_V3_BODY_EGO`) become YAML files
  that produce byte-identical `ObsSpec`s.
- Canaries (`tests/integration/test_scoring_canary.py` step 434 / reward
  7.3837; team-env canary) remain byte-identical after the refactor.

**Non-goals**

- Introducing brand-new block *types* purely in YAML. New block types
  (e.g. a "distance to nearest wall" feature) still require a one-line
  addition to the env's feature-dict function in Python. The YAML
  surface is composition-of-existing-blocks only.
- Changing how the persisted `[obs]` block in `run_info.toml` /
  `.hydra/config.yaml` is read for compatibility checks — the on-disk
  schema is preserved (plus a small additive `blocks` field on
  `.hydra/config.yaml` going forward).
- Touching `simple_env._obs`'s slots `[0:16]` contract that team-env
  warm-start surgery depends on.

## Design

### 1. YAML schema

`conf/obs/<stem>.yaml` carries the full structural description:

```yaml
name: DUEL_V4_MIX            # identifier — appears in W&B tags, lineage rows, MODEL.md
n_stack: 3                   # frame-stack depth (unchanged)
blocks:                      # ordered list of canonical ObsBlock identifiers
  - ANG_VEL
  - ANG_POS
  - LIN_VEL_BODY
  - LIN_POS
  - VEC_TO_GOAL_BODY
  - VEC_TO_HOOP_WORLD
  - OPP_POS_REL_BODY
  - OPP_VEL_REL_BODY_EGO
  - CLOSING_RATE
```

Each `blocks` entry is the literal Python identifier of an `ObsBlock`
constant in `obs_spec.py`. Unknown identifiers raise at env construction
with the full list of known blocks.

The four existing YAMLs (`simple.yaml`, `duel_v1_body.yaml`,
`duel_v2_world.yaml`, `duel_v3_body_ego.yaml`) get rewritten in this
format. Each must produce a byte-identical `ObsSpec` to the current
Python composed-constant of the same name.

### 2. `obs_spec.py` changes

**Removed:**

- `SIMPLE_ENV_OBS`, `DUEL_V1_BODY`, `DUEL_V2_WORLD`, `DUEL_V3_BODY_EGO`
- `SPEC_BY_NAME`

**Kept (with `name` field renames — see §4):**

- All canonical `ObsBlock` constants (`ANG_VEL`, `ANG_POS`, …,
  `OPP_VEL_REL_BODY_EGO`, …).

**Added:**

```python
BLOCK_BY_NAME: dict[str, ObsBlock] = {
    name: obj for name, obj in globals().items()
    if isinstance(obj, ObsBlock)
}

def build_spec_from_block_names(block_names: Iterable[str]) -> ObsSpec:
    """Build an ObsSpec from a sequence of canonical block identifiers.
    Raises KeyError with sorted list of known names on unknown input."""

def load_obs_yaml(stem: str) -> ObsSpec:
    """Load conf/obs/<stem>.yaml and build its ObsSpec.
    Convenience for tests that need a known spec by name."""
```

`__main__` of `obs_spec.py` (the `python -m envs.quidditch.obs_spec`
describer, surfaced via `make obs-specs`) iterates `conf/obs/*.yaml`
instead of `SPEC_BY_NAME`.

### 3. Universal feature dict per env

The `if spec == X / elif spec == Y` dispatch tree in
`team_env._pack_agent_obs` is replaced by a single function that
computes every feature this env can supply, keyed by canonical block
name (post-rename — see §4):

```python
def _build_agent_features(self, agent_id: str) -> dict[str, np.ndarray]:
    # Compute all shared intermediates as local variables.
    # vec_to_goal_world, unit_to_goal, opp_pos_rel_world,
    # opp_vel_rel_world, R_wb, closing_rate, …
    return {
        "ang_vel":                ang_vel,
        "ang_pos":                ang_pos,
        "lin_vel":                lin_vel_b,
        "lin_pos":                lin_pos,
        "unit_to_goal":           unit_to_goal,
        "signed_dist_norm":       np.array([signed_dist_norm], dtype=np.float32),
        "vec_to_hoop_world":      vec_to_hoop_world,
        "vec_to_hoop_body":       world_to_body(vec_to_hoop_world, R_wb),
        "vec_to_goal_body":       world_to_body(vec_to_goal_world, R_wb),
        "opp_pos_rel_world":      opp_pos_rel_world.astype(np.float32),
        "opp_pos_rel_body":       world_to_body(opp_pos_rel_world, R_wb),
        "opp_vel_rel_body_mixed": (opp_lin_vel - lin_vel_b).astype(np.float32),
        "opp_vel_rel_world":      opp_vel_rel_world,
        "opp_vel_rel_body_ego":   world_to_body(opp_vel_rel_world, R_wb),
        "closing_rate":           np.array([closing_rate], dtype=np.float32),
    }

def _pack_agent_obs(self, agent_id: str, spec: ObsSpec) -> np.ndarray:
    return obs_spec.pack(spec, self._build_agent_features(agent_id))
```

`simple_env._obs` receives the same treatment: a `_build_features()`
helper that returns the full dict for this env's supported blocks, and
`pack(self._spec, features)` consumes it. `self._spec` is set at
construction time from the resolved `cfg.obs.blocks` (passed in via env
kwargs, parallel to `learner_spec` in team_env).

**Why this shape and not per-block producer lambdas.** Profile cost of
computing the full feature set is single-digit µs per agent per step —
3–4 orders of magnitude below `mj_step` (~50–200 µs) and PPO forward
(~ms). Shared intermediates (e.g. `vec_to_goal_world` feeding both
the raw and body-rotated outputs) become local variables for free.
Readability and shared-state cost win against the lazy-evaluation gain.

If a future block ever has a genuinely expensive producer (e.g. a
long-range raycast), gate that single block:
`if "raycast_grid" in spec_names: raycast_grid = ...`. Apply
surgically per heavy block, not as the default pattern.

### 4. Unique block `name` fields

Today three `ObsBlock` base names collide across frame variants:

| Identifier | Before (`name=`) | After (`name=`) |
|---|---|---|
| `OPP_VEL_REL_BODY` | `opp_vel_rel` | `opp_vel_rel_body_mixed` |
| `OPP_VEL_REL_WORLD` | `opp_vel_rel` | `opp_vel_rel_world` |
| `OPP_VEL_REL_BODY_EGO` | `opp_vel_rel` | `opp_vel_rel_body_ego` |
| `VEC_TO_HOOP` | `vec_to_hoop` | `vec_to_hoop_world` |
| `VEC_TO_HOOP_BODY` | `vec_to_hoop` | `vec_to_hoop_body` |
| `OPP_POS_REL` | `opp_pos_rel` | `opp_pos_rel_world` |
| `OPP_POS_REL_BODY` | `opp_pos_rel` | `opp_pos_rel_body` |

All other blocks keep their current `name`. Rule: when a base name has
2+ variants, all variants gain explicit suffixes. When adding a future
second variant of a currently-single-variant block, rename both
together and add a `_LEGACY_NAME_RENAMES` entry.

This is necessary because the universal feature dict is keyed by
`ObsBlock.name`. Multiple blocks sharing the same `name` would collide
in the dict. The alternative (extending `ObsBlock` with a separate
`key` field) was considered and rejected — the rename keeps `pack()`
simple and exposes the discriminating frame in persisted `[obs]`
slot dumps.

### 5. Env factory + config schema

**`config_schema.py`:**

```python
@dataclass
class ObsConfig:
    name: str = "DUEL_V2_WORLD"
    n_stack: int = 3
    blocks: list[str] = field(default_factory=list)
```

**`env_factories.py`:** `SimpleEnvFactory` and `TeamEnvFactory` drop
`obs_spec_name: str` and gain `obs_blocks: list[str]` and `obs_name:
str`. Each calls `build_spec_from_block_names(self.obs_blocks)` once
inside `_make_thunk` and passes the resolved `ObsSpec` to the env
constructor.

**`conf/env/{simple,team}.yaml`:**

```yaml
# Before
obs_spec_name: ${obs.name}
frame_stack:   ${obs.n_stack}

# After
obs_blocks:    ${obs.blocks}
obs_name:      ${obs.name}
frame_stack:   ${obs.n_stack}
```

## Migration

### Legacy `run_info.toml` (pre-Hydra-Part-1 models)

These persist a full `[obs]` slots block with the old (collision-prone)
names. Read via [`_train_common.read_obs_spec`](../../../scripts/_train_common.py).

Migration: **rename-on-read.** A small dict in `obs_spec.py`:

```python
# Legacy persisted obs-block names — used to translate pre-2026-05-18
# `[obs] slots` entries on read.  Do not extend; new runs persist
# unique names directly.
_LEGACY_NAME_RENAMES: dict[tuple[str, str | None], str] = {
    ("opp_vel_rel", "body_mixed"): "opp_vel_rel_body_mixed",
    ("opp_vel_rel", "world"):      "opp_vel_rel_world",
    ("vec_to_hoop", "world"):      "vec_to_hoop_world",
    ("opp_pos_rel", "world"):      "opp_pos_rel_world",
    # body-frame variants of these blocks weren't persisted pre-2026-05-18.
}
```

`read_obs_spec` applies the rename when constructing each `ObsBlock`.
Pure read-path translation; no disk writes.

### Hydra-era `.hydra/config.yaml` (7 promoted models)

These carry `obs.name + obs.n_stack` only; no `blocks` list. Today
the env factory uses `SPEC_BY_NAME[name]` to reconstruct.

Migration: **one-shot disk migration** (precedent: 2026-05-15
`TEAM_ENV_OBS → DUEL_V1_BODY` rename). A throwaway script,
`tmp_migrate_obs_blocks.py`, that:

1. For each promoted model under `models/<name>/`, reads
   `.hydra/config.yaml`, looks up `obs.name` in a local `NAME_TO_BLOCKS`
   dict, injects `obs.blocks: [...]`, writes back.
2. For each affected W&B artifact (one per promoted model), uses
   `wandb.Api()` to download, patch `.hydra/config.yaml` inside the
   artifact bundle, re-upload. Targets are version-precise (e.g.
   `red_v1:v2`, not `:prod`) — same care as the 2026-05-15 script
   to avoid touching `:v0` siblings in the same collection.
3. Verifies each migrated config resolves to a byte-identical `ObsSpec`
   to the corresponding pre-refactor Python constant (using a snapshot
   of those constants taken before the deletion).
4. Is manually deleted by the user after `git status` confirms the
   expected disk changes and the canary tests still pass.

No permanent `_LEGACY_NAME_TO_BLOCKS` fallback in `obs_spec.py` —
after the script runs, all on-disk configs carry `obs.blocks`.

### `scripts/migrate_legacy_models.py:LEGACY_SPECS`

This hand-written audit map of pre-Hydra models retains the old block
names (`opp_vel_rel` everywhere). Mechanically rename to match the
unique names. Pure search-and-replace within that one dict.

## Affected files

**Created:**
- `tmp_migrate_obs_blocks.py` (throwaway; deleted after run)

**Modified — code:**
- `envs/quidditch/obs_spec.py` — delete composed-spec constants +
  `SPEC_BY_NAME`; add `BLOCK_BY_NAME`, helpers; rename block names per §4.
- `envs/quidditch/team_env.py` — replace `_pack_agent_obs` dispatch with
  universal feature dict; drop direct imports of composed constants.
- `envs/quidditch/simple_env.py` — replace `_obs` body with a feature
  dict and a `pack(self._spec, features)` call; accept `spec` kwarg.
- `envs/quidditch/env_factories.py` — drop `obs_spec_name`; add
  `obs_blocks: list[str]` and `obs_name: str`.
- `config_schema.py` — add `blocks: list[str]` to `ObsConfig`.
- `scripts/_train_common.py:read_obs_spec` — apply
  `_LEGACY_NAME_RENAMES` on read.
- `scripts/train.py` — replace `SPEC_BY_NAME[cfg.obs.name]` lookups
  with `build_spec_from_block_names(cfg.obs.blocks)`.
- `scripts/migrate_legacy_models.py:LEGACY_SPECS` — rename block names.
- `scripts/_render_model_doc.py:_section_obs_spec` — swap
  `SPEC_BY_NAME[name]` for `build_spec_from_block_names(cfg.obs.blocks)`.

**Modified — configs:**
- `conf/obs/{simple,duel_v1_body,duel_v2_world,duel_v3_body_ego}.yaml` —
  rewrite in new schema; each must produce byte-identical `ObsSpec`.
- `conf/env/{simple,team}.yaml` — swap `obs_spec_name` for
  `obs_blocks` + `obs_name`.

**Modified — tests (~8 files):**
- `tests/core/policies/test_warm_start.py`,
  `test_warm_start_by_spec.py`,
  `tests/envs/quidditch/test_augmented_obs.py` — replace direct
  composed-constant imports with `load_obs_yaml("...")`.
- `tests/scripts/test_render_model_doc.py`,
  `test_migrate_legacy_models.py`,
  `test_train_resolve_parent_wiring.py`,
  `test_log_run_artifact.py`,
  `test_wandb_init.py` — add `blocks: [...]` to in-memory fixture
  config dicts.

## Testing strategy

**Equality canary mid-refactor.** Before deleting the composed Python
constants, add a one-off test that snapshots them and asserts
`load_obs_yaml("simple") == _snapshot_simple_env_obs` etc. for all four.
The snapshots use the *new* (renamed) block names — the test passes
once the YAMLs and renames are in lockstep. Delete the snapshots once
the deletion lands.

**Behaviour canaries.** Both byte-identical fingerprints must survive:
- `tests/integration/test_scoring_canary.py`: SCORED at step 434, total
  reward 7.3837.
- `tests/integration/test_team_env_canary.py`: team-env step trace.

**Compat-check canary.** Add a test that round-trips: write an `[obs]`
slots block for `DUEL_V1_BODY` using pre-refactor (`name="opp_vel_rel"`,
…) wire format → parse via `read_obs_spec` → assert it equals
`load_obs_yaml("duel_v1_body")`. This guards the `_LEGACY_NAME_RENAMES`
translation.

**Pretrain canary.** A smoke test (with `WANDB_MODE=disabled`) that
loads `models/ppo_hoop_blue_4_*/best_model.zip` after the disk
migration and verifies obs-compat against
`load_obs_yaml("duel_v2_world")` — guards the §migration disk patch.

## Risks

- **W&B artifact migration is destructive.** Re-uploading patched
  `.hydra/` to W&B creates a new artifact version; the prior `:vN`
  remains immutable but unreferenced from the migrated `.hydra/`.
  Aliases (`:prod`, `:<run_name>`) get re-pointed to the new version.
  If the script aborts mid-flight, some collections may have the new
  version live while others don't. Mitigation: dry-run mode first (`-n`)
  that prints the planned mutations; explicit `--commit` to actually
  upload; idempotent re-runs (skip if the artifact already has `blocks`
  in its `.hydra/config.yaml`).
- **Lazy `BLOCK_BY_NAME` initialization timing.** Built by introspecting
  `globals()` at module import. Adding a new block constant *below*
  the `BLOCK_BY_NAME` assignment line would silently exclude it.
  Mitigation: build it at the bottom of the module after all `ObsBlock`
  constants are declared; add a unit test that asserts every
  module-level `ObsBlock` attribute appears in `BLOCK_BY_NAME`.
- **Block-name suffix policy isn't future-proof.** If we later add the
  first body-frame variant of a currently-single-variant block (e.g.
  `LIN_POS_BODY`), the existing block needs renaming and another
  `_LEGACY_NAME_RENAMES` entry. Documented in §4; not auto-enforced.

## Out of scope (deferred)

- Declarative producers in YAML (see Non-goals).
- Removing `obs_name` from the env factory (and deriving it from the
  YAML's `name:` field implicitly). The current explicit propagation is
  cleaner because env factories don't read YAMLs directly — Hydra does.
- Migrating `scripts/migrate_legacy_models.py` itself off the
  hand-written `LEGACY_SPECS` audit table. That table is unrelated to
  this refactor's surface (it covers pre-2026-05-13 run_info.toml
  schema) and the entries are correct in spirit; only the block names
  inside it change.
