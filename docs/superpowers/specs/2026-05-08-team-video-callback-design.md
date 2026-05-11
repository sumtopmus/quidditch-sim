# Team-env eval video callback — design

**Status.** Approved (brainstorming complete) — ready for implementation plan.
**Branch.** `feature/team-video-callback`.
**Closes.** Known issue at `brain/index.md` line 36 ("Team-env training has no video callback yet").

## Problem

`make train-team-red`, `train-team-red-warm`, and `train-team-blue` produce no eval videos in `runs/<name>/<trial>/videos/` and emit nothing to TensorBoard under `eval/video/*`. The single-agent path (`make train`) does both via `VideoRecorderCallback` in `scripts/callbacks.py`, but the shared callback factory `_train_common.build_callbacks` deliberately omits video — its docstring (`scripts/_train_common.py:119-121`) flags this as coupled to simple-env render plumbing and punts to the entry script.

Two underlying coupling points:

1. `VideoRecorderCallback._capture_cells` reaches into `env._quad.render_cells(...)` (`scripts/callbacks.py:146`). `_quad` exists on `QuidditchSimpleEnv` only; `QuidditchTeamEnv` has `_red`/`_blue` plus a shared `_world`.
2. The callback's `grid_cams` config defaults assume a single drone with prefix-less chase-cam names (e.g. plain `tpv`). The team env carries two drones with prefixed cam names (`red_0_tpv`, `blue_0_tpv`).

There is also a latent bug in single-agent: `config/training.toml` declares `cells = ["south", "east", "top", "tpv"]`, but the actual MuJoCo cam name is `drone_tpv` (per-prefix in `core/drone/cf2x.py:298-303`). MuJoCo's `update_scene(camera="tpv")` falls back to free-cam when the name doesn't resolve, so the bottom-right cell of every existing single-agent grid video is a free-cam shot, not a chase view. Folded into this work because we touch the same call-site.

## Goal

Team training writes eval videos to `runs/<name>/<trial>/videos/step_<N>.mp4` and embeds them in TensorBoard at `eval/video/<cam>` on the same eval cadence as single-agent. One generic callback class works for both training modes; one shared config block expresses cam/grid/freq settings.

## Non-goals

- Per-drone FPV/TPV defaults that auto-substitute the learner prefix (option B from brainstorming — rejected; arena-centric default chosen instead).
- Recording the live opponent's policy explicitly — `OpponentControlledEnv` already drives the opponent at every `env.step`, so any cam pointed at the opponent's body works without special wiring.
- Team-specific frequency settings — `[training.callbacks].video_every_n_evals` and `video_fps` stay shared with single-agent.

## Approach

### Three coordinated changes

**1. Rename single-agent drone prefix `drone` → `red_0`.**

In `envs/quidditch/simple_env.py:166`, the single drone is constructed as `Quadrotor(self._world, prefix="drone")`. Renaming to `red_0` aligns the simple-env body/sensor/cam namespace with the team-env namespace:

- Same chase-cam name `red_0_tpv` resolves in both training modes.
- The single-agent learner's role (fly through the hoop) maps semantically onto the team env's Red (attacker, scores). Phase 2a warm-start (`models/ppo_hoop_rand_start_20260505_174509 → red_0`) reads cleanly with no role flip.
- Phase 2b ("train Blue against frozen Red") loads a frozen Red checkpoint that was *also* trained as Red — naming consistency throughout the lineage.

**Compatibility:**
- SB3 model files store `observation_space` and `action_space` shapes, not body/cam names. Existing `models/ppo_hoop_*` checkpoints continue to load against the renamed env.
- MuJoCo body/dof/geom ordering is XML-declaration order, not name order, so the rename does not reorder simulation state. The scoring canary value (`SCORED at step 434 / total reward 7.3837` in `tests/integration/test_scoring_canary.py`) is expected to hold post-rename. **Verifying this is the canary check; if it shifts, investigate before merging.**
- The `cf2_fragment` and `arena_wall_fragment` factories already accept `prefix` as an argument — no factory changes needed.
- Things to grep for and update: any literal `"drone"` string in `envs/quidditch/simple_env.py`, `envs/quidditch/rewards.py` (if it indexes by prefix), `core/policies/warm_start.py` (input-layer surgery may reference prefix), `tests/`, `demo/`, and `Makefile`/help strings.

**2. Make `VideoRecorderCallback` env-agnostic.**

Replace `env._quad.render_cells(...)` (`scripts/callbacks.py:146`) with `env._world.render_cells(...)`. `_world` exists directly on `QuidditchSimpleEnv` (`envs/quidditch/simple_env.py:123`) and `QuidditchTeamEnv` (`envs/quidditch/team_env.py:127`).

For team training, the callback's `env_fn` returns the **single-agent** `OpponentControlledEnv` shim (it has to — the rollout loop calls `env.reset()` / `env.step(action)` with single-agent gym shape). `OpponentControlledEnv` does not expose `_world` today. Add a one-line passthrough property on `OpponentControlledEnv` (`envs/quidditch/opponents.py`):

```python
@property
def _world(self) -> "World":
    return self.team_env._world
```

This keeps the callback's call-site uniform across env types (`env._world.render_cells(...)`) without leaking team-env internals. Update the callback docstring to drop simple-env-specific language.

**3. Move video callback construction into `_train_common.build_callbacks`.**

Add an optional `video_env_fn: Callable | None = None` parameter to `build_callbacks`. When provided, build a `VideoRecorderCallback` from the shared `[training.callbacks.video]` config and append to the returned callback list. When `None`, behave exactly as today (no video).

`scripts/train_team_ppo.py` calls `build_callbacks(..., video_env_fn=lambda: OpponentControlledEnv(QuidditchTeamEnv(render_mode="rgb_array", **cfg), ...))`. The video env mirrors the eval env construction — same opponent spec, same env config — but with `render_mode="rgb_array"` instead of `None`.

`scripts/train_ppo.py` is intentionally **not** migrated to use `build_callbacks` for video. The single-agent entry script keeps its inline `VideoRecorderCallback` construction (per `_train_common`'s docstring: "train_ppo.py keeps its own copies of these helpers (untouched) so the single-agent canary commit stays bit-identical"). Future consolidation is a separate refactor.

### Config

One shared `[training.callbacks.video]` block in `config/training.toml` and `templates/training.toml`. Default cells with the prefix rename in place:

```toml
[training.callbacks.video]
grid        = true
cells       = ["south", "east", "top", "fixed"]
cell_width  = 960
cell_height = 540
```

**Default rationale (option C from brainstorming, "arena-centric"):** all four cams are scene-fixed (no per-drone names), so the same default works for `train`, `train-team-red`, `train-team-red-warm`, and `train-team-blue` without per-mode tuning. Users can override to per-drone cams (`red_0_tpv`, `blue_0_tpv`, etc.) via the same key.

This default also fixes the latent `tpv` resolution bug — the broken `"tpv"` entry is replaced with `"fixed"`, so every cell renders a real cam. (If a user wants a chase view, they swap in `red_0_tpv` knowing it now resolves.)

Frequency settings stay in `[training.callbacks]`:
```toml
video_every_n_evals = 4
video_fps           = 20
```

These are read by both training modes via the shared `_train_common.build_callbacks` path.

### Eval episode mechanic for team

`OpponentControlledEnv` is already a single-agent `gym.Env` (`reset()` → learner obs, `step(action)` → learner transition, opponent driven internally each step via the wrapped scripted/frozen Opponent). The existing `VideoRecorderCallback._on_step` rollout loop works unchanged. The opponent's behaviour is captured in the video without special wiring because it physically moves in the shared world.

### TB tags

Same scheme as single-agent: `eval/video/<cam>` per cell, lowercase. With the arena-centric default the tags are `eval/video/south`, `eval/video/east`, `eval/video/top`, `eval/video/fixed`. With per-drone overrides (`red_0_tpv`), the cam name is already lowercase — no special handling.

## Files touched

| File | Change |
|------|--------|
| `envs/quidditch/simple_env.py` | `prefix="drone"` → `prefix="red_0"`. Update any other internal references. |
| `envs/quidditch/rewards.py` | Audit for `"drone"` literals; rename if any. |
| `core/policies/warm_start.py` | Audit for prefix references in input-layer surgery; rename if any. |
| `scripts/callbacks.py` | `env._quad.render_cells` → `env._world.render_cells`; drop simple-env-only docstring language. |
| `envs/quidditch/opponents.py` | Add `_world` passthrough property to `OpponentControlledEnv`. |
| `scripts/_train_common.py` | Add optional `video_env_fn` param to `build_callbacks`; build + append `VideoRecorderCallback` when provided. |
| `scripts/train_team_ppo.py` | Pass `video_env_fn=...` to `build_callbacks`; the lambda constructs `OpponentControlledEnv(QuidditchTeamEnv(render_mode="rgb_array"), ...)`. |
| `config/training.toml` | `cells = ["south", "east", "top", "fixed"]` (was `[..., "tpv"]`). |
| `templates/training.toml` | Mirror config change. |
| `tests/integration/test_scoring_canary.py` | Verify `step 434 / 7.3837` still asserts after prefix rename. |
| `tests/` (new) | Smoke test: build team env in rgb_array mode, instantiate callback, run `_capture_cells` once, assert cells are non-empty `(cell_h, cell_w, 3)` arrays. |
| `Makefile` | Audit help strings for `drone` prefix references. |
| `demo/` | Audit for `prefix="drone"` references; update if any. |

## Test plan

**Pre-merge canaries (must pass):**
1. `make test-fast` — full unit suite (post-rename MJCF builds, tag FSM, crash detector, render smoke).
2. `make test` — including `tests/integration/test_scoring_canary.py` asserting `step 434 / total reward 7.3837` for the renamed single-agent env.
3. `make test-warm MODEL=ppo_hoop_rand_start_20260505_174509` — verifies warm-start surgery still works with the renamed prefix.

**New tests:**
4. New unit test (`tests/unit/test_team_video_callback.py` or similar): construct `OpponentControlledEnv(QuidditchTeamEnv(render_mode="rgb_array"), learner_id="red_0", opponent=BeelineBlueOpponent())`, instantiate `VideoRecorderCallback`, manually invoke `_capture_cells(env)`, assert the result is a list of 4 `(cell_h, cell_w, 3)` `uint8` arrays. No SB3 model needed.

**Smoke (manual, optional):**
5. `make train-team-red TIMESTEPS=200000` (short run that hits at least one video trigger). Confirm `runs/team_red/<trial>/videos/step_*.mp4` exists, is playable, and shows a 2x2 grid with all four cells rendering real scene content. Confirm TB shows `eval/video/{south,east,top,fixed}` clips at the same step.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Prefix rename shifts the scoring canary value | Run canary first; if it shifts, the prefix change is somehow affecting state ordering — investigate before merging. Most likely explanation would be alphabetic ordering inside `merge_all` or fragment composition, both of which currently preserve insertion order. |
| Existing `models/ppo_hoop_*` checkpoints fail to load | SB3 model files don't reference body/cam names — load only depends on obs/action space shapes (16-d and 4-d). Verified by `make test-warm`. |
| Adding `_world` passthrough leaks team-env internals onto the wrapper | The leak is intentional and minimal: the callback explicitly contracts on `_world` as the rendering entry point. If a future refactor introduces a more principled `Renderable` protocol, the property becomes the implementation point. |
| Single-agent video output bytes change | Acceptable. The canary assertion is on reward/step counts, not video bytes. The `tpv` → `fixed` cell change is the visible delta and is intentional (was rendering free-cam fallback before). |
| Future single-agent regress when someone forgets to update both `train_ppo.py` and `_train_common.py` | Out of scope. The two-callback-builder split is pre-existing and called out in `_train_common`'s own docstring. |

## Open questions

None remaining as of writing.
