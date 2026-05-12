# Obs-spec named slots — Design

**Goal.** Make the structure of every observation vector in the project an explicit, named, serialized artifact — declared once in `envs/quidditch/obs_spec.py`, consumed by all three obs builders, written into `run_info.toml` under a new `[obs]` block, and checked at load time on `--pretrain` / `--warm-start` / `--resume`. Generalize the hardcoded 16→22 input-layer surgery in `core/policies/warm_start.py` into a named-block surgery that copies whatever blocks match `(name, dim, frame)` and small-inits the rest.

**Why now.** Phase 2b had three obs revisions in one week (16-d simple → 22-d team → 25-d augmented in `OpponentControlledEnv`, plus `frame_stack=3` on top). Old Blue checkpoints (`blue_v1` and any pre-2026-05-11 checkpoint) are no longer loadable under the current env shape because `PPO.load` rejects the obs-shape mismatch with no actionable diagnostic — and the silent body-mixed → world-frame change on `opp_vel_rel` would not have been caught even if shapes had stayed the same. Both failure modes are in scope for the spec.

**Non-goals.**
- Eval-time / frozen-opponent runtime spec checks (e.g. when `blue_v6` trains against `frozen:models/red_v1/best_model`, the team env builds a 22-d opp obs and the frozen Red consumes it — that's a runtime contract, not a load-time surgery question, and needs its own design).
- GUI eval (`make eval-team GUI=1`) obs-spec checks.
- Auto-inferring obs specs for legacy runs. The set of promoted models is closed (seven) and hand-writing the mapping is safer than building inference for a closed set.
- `run_info.toml` schema versioning. We add a new optional block; old readers ignore it. If `[obs]` itself ever needs to evolve, a `[obs].version` key gets added then.

## Design Decisions

### 1. The spec module — `envs/quidditch/obs_spec.py`

Two frozen dataclasses plus a registry of canonical `ObsBlock` constants and composed `ObsSpec` constants for each entry point.

```python
@dataclass(frozen=True)
class ObsBlock:
    name: str
    dim: int
    frame: str | None = None   # "world", "body", "body_mixed", or None
    notes: str | None = None   # free-text, not machine-checked

@dataclass(frozen=True)
class ObsSpec:
    blocks: tuple[ObsBlock, ...]

    @property
    def dim(self) -> int: return sum(b.dim for b in self.blocks)
    def offsets(self) -> list[tuple[ObsBlock, slice]]: ...
```

Equality on `ObsBlock` uses all four fields. Equality on `ObsSpec` is structural over `blocks`. The `name` field is *not* unique across all defined blocks — see decision 2.

**Block declarations:**

```python
ANG_VEL           = ObsBlock("ang_vel",          dim=3, frame="body")
ANG_POS           = ObsBlock("ang_pos",          dim=3, frame="body")
LIN_VEL_BODY      = ObsBlock("lin_vel",          dim=3, frame="body")
LIN_POS           = ObsBlock("lin_pos",          dim=3, frame="world")
UNIT_TO_GOAL      = ObsBlock("unit_to_goal",     dim=3, frame="world", notes="unit vector toward hoop (red) or midpoint (blue)")
SIGNED_DIST_NORM  = ObsBlock("signed_dist_norm", dim=1, notes="(pos - hoop)·hoop_normal / ARENA_RADIUS")
VEC_TO_HOOP       = ObsBlock("vec_to_hoop",      dim=3, frame="world", notes="HOOP_CENTER - learner_pos, not normalized")
OPP_POS_REL       = ObsBlock("opp_pos_rel",      dim=3, frame="world")
OPP_VEL_REL_BODY  = ObsBlock("opp_vel_rel",      dim=3, frame="body_mixed", notes="legacy: each velocity in its own body frame")
OPP_VEL_REL_WORLD = ObsBlock("opp_vel_rel",      dim=3, frame="world")
CLOSING_RATE      = ObsBlock("closing_rate",     dim=1, notes="-d‖opp - learner‖/dt")
```

**Composed specs:**

```python
SIMPLE_ENV_OBS = ObsSpec((ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM))
TEAM_ENV_OBS   = ObsSpec(SIMPLE_ENV_OBS.blocks + (OPP_POS_REL, OPP_VEL_REL_BODY))
AUGMENTED_OBS  = ObsSpec((ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL,
                         VEC_TO_HOOP, OPP_POS_REL, OPP_VEL_REL_WORLD, CLOSING_RATE))
```

### 2. Same `name` across blocks is allowed when `frame` differs

The `opp_vel_rel` block exists in two variants — `OPP_VEL_REL_BODY` (the legacy body-mixed pre-2026-05-11 team obs) and `OPP_VEL_REL_WORLD` (the post-2026-05-11 augmented obs). They share `name="opp_vel_rel"` and `dim=3` but differ in `frame`. Compatibility is checked on the full `(name, dim, frame)` tuple, so these collide intentionally — a parent run that used the body-mixed variant cannot transparently load into a current env using the world-frame variant; the diff calls this out under decision 5.

**Decision:** the dictionary key when filling a spec (decision 3) must be unique within a single spec; collision across specs is allowed and meaningful.

### 3. Obs builders consume the spec — no more ad-hoc concatenate

A small helper `obs_spec.pack(spec, values)` resolves each block in `spec.blocks` order by looking up `values[block.name]`, asserts the per-block dim matches, and returns the concatenated `np.float32` array.

```python
# simple_env._obs()
return obs_spec.pack(SIMPLE_ENV_OBS, {
    "ang_vel": ang_vel, "ang_pos": ang_pos,
    "lin_vel": lin_vel, "lin_pos": lin_pos,
    "unit_to_goal": unit_to_hoop,
    "signed_dist_norm": [signed_dist_norm],
})
```

Same pattern at `team_env._build_agent_obs()` (fills `TEAM_ENV_OBS`) and `OpponentControlledEnv._augment_learner_obs()` (fills `AUGMENTED_OBS`). The `team_env` builder keeps the body-mixed `opp_vel_rel` value because `OPP_VEL_REL_BODY.frame = "body_mixed"` is what the legacy spec records — the existing 22-d obs stays byte-identical and the canary `tests/integration/test_scoring_canary.py` (`step 434 / reward 7.3837`) does not shift.

`observation_space` for each env becomes `spaces.Box(..., shape=(SPEC.dim,), ...)` — derived from the spec, removing the hazard of having to update three places when a literal changes.

### 4. `[obs]` block in `run_info.toml`

`scripts/_train_common.write_run_info` gains `obs_spec: ObsSpec | None = None` and `n_stack: int = 1`. When `obs_spec` is provided, the function emits a new `[obs]` block in inline-array-of-tables form:

```toml
[obs]
dim     = 25
n_stack = 3
# slots is an inline array-of-tables; each entry is one ObsBlock.
slots = [
  {name = "ang_vel",          dim = 3, frame = "body"},
  {name = "ang_pos",          dim = 3, frame = "body"},
  {name = "lin_vel",          dim = 3, frame = "body"},
  {name = "lin_pos",          dim = 3, frame = "world"},
  {name = "unit_to_goal",     dim = 3, frame = "world"},
  {name = "vec_to_hoop",      dim = 3, frame = "world"},
  {name = "opp_pos_rel",      dim = 3, frame = "world"},
  {name = "opp_vel_rel",      dim = 3, frame = "world"},
  {name = "closing_rate",     dim = 1},
]
```

Rules:
- `dim` is redundant with `sum(slots[*].dim)` but emitted explicitly as a fast sanity check on read.
- `frame` and `notes` are emitted only when set; absent entries → `None` on read.
- `n_stack = 1` is the default and *is still written* — explicit-is-better-than-implicit, and the back-fill (decision 7) needs to write it for old models.

Both `train_team_ppo` and `train_ppo` pass `obs_spec=` and `n_stack=` from their env-build sites; `n_stack` comes from `config["training"].get("frame_stack", 1)`.

`scripts/_train_common.read_obs_spec(info_path) -> tuple[ObsSpec, int] | None` parses the block and returns `(ObsSpec, n_stack)` — or `None` if absent.

### 5. Loader integration — strict refusal default, opt-in surgery

A new helper `check_obs_compat(parent_info, current, current_n_stack, *, surgery: bool) -> tuple[ObsSpec, int] | None` compares the parent's `[obs]` block against the current env's spec. On exact match it returns the parent's `(spec, n_stack)`. On any mismatch it prints a column-by-column diff and exits unless `surgery=True`.

**Status encoding in the diff:**
- ✅ — exact match on `(name, dim, frame)` (notes may differ silently and are surfaced separately).
- ⚠️ — same `name` and `dim`, different `frame`. Block remains aligned offset-wise; surgery cannot copy it safely (frame semantics differ), so the columns get small-init.
- ❌ — block added or removed (presence mismatch), or `n_stack` differs. Same-name-different-`dim` is also rendered as `❌ removed` + `❌ added` because offsets diverge for everything downstream — it's the honest representation of the column drift.

**Offset column semantics in the diff:** the renderer walks both specs in parallel order (not by offset alignment) and emits one row per spec position. `✅` and `⚠️` rows have coinciding offsets. `❌ removed` rows show the parent offset; `❌ added` rows show the current offset.

```
Obs spec mismatch between parent and current env.
  parent: models/ppo_hoop_blue_1_20260507_194423/run_info.toml
  current: TEAM_ENV_OBS  (n_stack=3)

  offset  parent                          current
  ------  ------------------------------  ------------------------------
   0:3    ang_vel       dim=3 body        ang_vel       dim=3 body       ✅
   3:6    ang_pos       dim=3 body        ang_pos       dim=3 body       ✅
   6:9    lin_vel       dim=3 body        lin_vel       dim=3 body       ✅
   9:12   lin_pos       dim=3 world       lin_pos       dim=3 world      ✅
  12:15   unit_to_goal  dim=3 world       unit_to_goal  dim=3 world      ✅
  15:16   signed_dist_norm dim=1          —                              ❌  removed
  15:18   —                               vec_to_hoop   dim=3 world      ❌  added
  16:19   opp_pos_rel   dim=3 world       opp_pos_rel   dim=3 world      ✅
  19:22   opp_vel_rel   dim=3 body_mixed  opp_vel_rel   dim=3 world      ⚠️  frame changed
   —      —                               closing_rate  dim=1            ❌  added
  n_stack 1                               3                              ❌

  Run with --obs-surgery to copy matching blocks and small-init the rest.
```

Notes diffs (informational only, do not affect the load decision):

```
  ⚠️  Notes changed on matching blocks (informational, load proceeds):
       signed_dist_norm:
         parent:  (pos - hoop)·hoop_normal / ARENA_RADIUS
         current: signed distance to hoop plane, normalized by ARENA_RADIUS
```

**Wiring per load path:**
- `train_ppo.py --pretrain` and `train_team_ppo.py --pretrain`: call `check_obs_compat` before `PPO.load`. Strict match → load normally. `--obs-surgery` and mismatch → route through generalized surgery (decision 6).
- `train_team_ppo.py --warm-start`: same. `--obs-surgery` is the only path to the surgery codepath after the legacy hardcoded 16→22 helper is retired.
- `train_team_ppo.py --resume` and `train_ppo.py --resume`: stricter — `--obs-surgery` is **rejected** for resume. Resume means "same run, same model, same env"; an obs mismatch means the code drifted under the run, which is a stop-and-investigate signal, not something to paper over. Error message names the parent run.
- Parent without `[obs]` block (older run, pre-this-feature): treat the same as a mismatch — refuse by default, allow surgery via `--obs-surgery`. Back-fill (decision 7) covers the seven promoted models so this path mostly triggers only for archived runs in `runs/`.

### 6. Generalized surgery — `warm_start_ppo_by_spec`

`core/policies/warm_start.py` gets a new entry point that supersedes the hardcoded 16→22 one:

```python
def warm_start_ppo_by_spec(
    *,
    old_checkpoint: str | Path,
    new_env: Any,                  # VecEnv with current spec applied
    parent_spec: ObsSpec, parent_n_stack: int,
    current_spec: ObsSpec, current_n_stack: int,
    new_dim_init_scale: float = 0.01,
    **ppo_kwargs: Any,
) -> PPO: ...
```

**Algorithm — input layer surgery, per-frame-stack-slice:**

1. Walk `current_spec.blocks` in order. For each current block, find the parent block with matching `(name, dim, frame)`; record `(parent_offset, current_offset, dim)`. Blocks without a match are "new" and will be small-init.
2. Load old policy; create new policy with input dim `current_spec.dim * current_n_stack`.
3. For each `(hidden_dim, current_spec.dim * current_n_stack)` input weight matrix in the new policy's `state_dict`, partition columns into `current_n_stack` slices. For each slice:
   - For each matched `(parent_off, current_off, dim)` triple: copy `old_weight[:, parent_off : parent_off+dim]` into `new_weight[:, slice_base + current_off : slice_base + current_off + dim]`.
   - σ=0.01 Gaussian init for unmatched columns within the slice.
4. If `parent_n_stack != current_n_stack`:
   - `parent_n_stack < current_n_stack` (e.g. 1 → 3): parent's slice(s) are repeated to fill the current slices. With `parent_n_stack=1` and `current_n_stack=3`, the single parent slice is written into each of the three current slices, so initial behavior is "act on the current frame, ignore history."
   - `parent_n_stack > current_n_stack` (e.g. 3 → 1): copy the most-recent parent slice (highest temporal index) and discard the older slices. Older slices are not behaviorally addressable in the new policy.
   - In both cases the surgery logs the decision explicitly.
5. Non-input layers: copy when shape matches (current behavior in `warm_start_ppo`), skip otherwise.

The existing `warm_start_ppo(old_input_dim=16, new_input_dim=22)` function is **deleted**. Its sole call site in `scripts/train_team_ppo.py:209-218` is rewritten to route through `warm_start_ppo_by_spec` with `SIMPLE_ENV_OBS` as parent and `TEAM_ENV_OBS` as current. The existing `tests/integration/test_warm_start.py` becomes the canary that the new function reproduces the old 16→22 behavior.

`--warm-start` remains as an alias of `--pretrain --obs-surgery` for backward compatibility with `make` targets and existing muscle memory; it does not gain new behavior of its own.

### 7. Back-fill of existing promoted models

The seven runs in `models/` predate this feature, so their `run_info.toml` files lack `[obs]`. Without back-fill, every load from one of them would force `--obs-surgery` even when shapes match exactly. The fix is a one-off script with a hand-written mapping.

`scripts/backfill_obs_spec.py`:

```python
LEGACY_SPECS: dict[str, tuple[str, int]] = {
    # run_name prefix                                 → (spec name, n_stack)
    "ppo_hoop_fixed_start_20260430_224234":             ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_fixed_start_20260504_023051":             ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_rand_start_20260430_234354":              ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_rand_start_20260505_174509":              ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_red_1_20260506_103058":                   ("TEAM_ENV_OBS",   1),
    "ppo_hoop_blue_1_20260507_194423":                  ("TEAM_ENV_OBS",   1),
    "ppo_hoop_blue_4_20260511_202612":                  ("AUGMENTED_OBS",  3),
}
```

The script:
1. Walks `models/*/run_info.toml`.
2. Looks up the run-name prefix against `LEGACY_SPECS`.
3. Refuses to overwrite if `[obs]` already exists (idempotent).
4. Appends the `[obs]` block in canonical format (inline array-of-tables) with a `# back-filled by scripts/backfill_obs_spec.py on YYYY-MM-DD` comment marker.
5. Prints a summary table.

**Why hand-written and not inferred.** `blue_1` and `red_1` both used `TEAM_ENV_OBS` (body-mixed `opp_vel_rel`), but `blue_4` used `AUGMENTED_OBS` (world-frame plus `vec_to_hoop` and `closing_rate`). The only difference at runtime was that `blue_4`'s training script went through the new `OpponentControlledEnv` augmentation path; there is no flag in `config_snapshot.toml` we could key off. The spec was determined by *what code ran*, which we know by date — hand-writing the seven mappings is faster and safer than building an inference function for a closed set.

**Runs in `runs/` are not back-filled.** Only promoted models in `models/` matter (they're the only `--pretrain` sources). Failed/in-progress runs that someone might warm-start from will surface "no `[obs]` block, refusing unless `--obs-surgery`" — the correct behavior.

The script is committed and kept (not deleted) for audit trail.

### 8. Tests

**New tests** (added to the existing pytest suite):
- `tests/unit/test_obs_spec.py` — `ObsBlock`/`ObsSpec` basics: `dim` sums correctly; `offsets()` returns correct slices; two blocks with same name but different frame don't collide on `(name, dim, frame)` equality; `pack()` raises on missing key.
- `tests/unit/test_obs_spec_toml.py` — round-trip: `write_run_info(obs_spec=SPEC, n_stack=N)` then `read_obs_spec(path)` returns the same spec. Covers all three concrete specs and `n_stack ∈ {1, 3}`.
- `tests/unit/test_obs_compat.py` — `check_obs_compat`: identical specs pass; added/removed blocks raise `SystemExit` with `❌` in the diff; same-name-different-frame raises with `⚠️` for that line; notes-only diff passes (informational footer only); `--obs-surgery=True` allows all three mismatch types but `n_stack` differing still rejects on `--resume` path.
- `tests/unit/test_warm_start_by_spec.py` — replaces / augments the existing 16→22 test: `SIMPLE_ENV_OBS` → `TEAM_ENV_OBS` warm-start copies columns 0:16 exactly and σ=0.01-inits columns 16:22; `TEAM_ENV_OBS` → `AUGMENTED_OBS` warm-start correctly handles the offset shift (slot 15 differs, slots 16:19 still match by name, slots 19:22 are a frame change and get small-init).
- `tests/integration/test_warm_start.py` — kept; now exercises the spec-routed path. Same `step 434 / reward 7.3837` canary still holds since `SIMPLE_ENV_OBS` ⊂ `TEAM_ENV_OBS`.

**Regression coverage that must not break:**
- Single-agent scoring canary: `step 434 / reward 7.3837`.
- `test_augmented_obs.py` — the 5 cases for the 25-d augmented obs continue to pass against `AUGMENTED_OBS`-built arrays.
- `test_team_pretrain_args.py`, `test_write_run_info_pretrain.py`, `test_read_parent_chain_total.py` — `--pretrain` plumbing.

## Future work

- **Eval-time / frozen-opponent spec checks.** When `blue_v6` trains against `frozen:models/red_v1/best_model`, cross-check the env's opp-obs shape against the frozen policy's recorded spec. Runtime contract, different design.
- **GUI eval (`make eval-team GUI=1`) spec checks.** Same shape of problem.
- **Schema versioning on `[obs]`.** Add a `version` key only when the block's own shape needs to evolve.
