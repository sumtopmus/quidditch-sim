# Design — Migrate `scripts/check_*.py` to pytest, plus scripted GUI scenarios

**Date:** 2026-05-06
**Status:** Design approved; ready for implementation plan.

## Goal

Replace the seven hand-rolled check scripts under `repo/scripts/` with a proper `tests/` suite driven by pytest (unit + integration), and replace the single visual check (`make check-gui`) with two scripted 1v1 narrative demos that play back-to-back inside one MuJoCo viewer.

Two motivations:

1. **Asserted, fast, filterable.** Today's checks are loose `if not ok: print(...)` scripts — adding a new check means adding a Makefile target and a `sys.path` shim. With pytest, an asserted check is one `def test_*` function; markers cleanly separate fast unit logic from slow integration canaries.
2. **Watchable storytelling for the 1v1 game.** `make check-gui` is currently just "fly through the hoop and hover for 60s." With the team env online, a more useful visual check is a deterministic narrative: Red attacks → Blue tags ×2 → takedown; then Red attacks → tagged once → still scores. Both scenarios scripted, both deterministic, viewer + terminal reward log.

## Inventory of what's being replaced

| Old script | Type | New home |
|---|---|---|
| `scripts/check_env.py` (SB3 + zero-action parts) | mixed contract + smoke | `tests/integration/test_simple_env_contract.py` |
| `scripts/check_env.py` (scripted-fly-to-hoop) | deterministic regression canary | `tests/integration/test_scoring_canary.py` |
| `scripts/check_env.py --viewer` | manual visual | **deleted** — replaced by `demo/scenarios.py` |
| `scripts/check_team_env.py` | deterministic regression canary | `tests/integration/test_team_env_canary.py` |
| `scripts/check_team_mjcf.py` | unit asserts on MJCF model | `tests/unit/test_team_mjcf.py` |
| `scripts/check_team_crash.py` | unit asserts on `CrashDetector` | `tests/unit/test_crash_detector.py` |
| `scripts/check_team_tag.py` | unit asserts on tag FSM | `tests/unit/test_tag_state_machine.py` |
| `scripts/check_team_warm.py` | integration; needs trained model | `tests/integration/test_warm_start.py` |
| (none — new) | render smoke | `tests/unit/test_render_smoke.py` |

All seven `check_*.py` files are deleted after their tests pass. `scripts/train_*.py`, `scripts/eval_*.py`, `scripts/lineage.py`, `scripts/_train_common.py`, `scripts/callbacks.py` are untouched.

## Directory layout

```
repo/
├── tests/                       # NEW — pytest root
│   ├── conftest.py              # shared helpers (build_team_world, qpos_addr, set_body_state)
│   ├── unit/
│   │   ├── test_team_mjcf.py
│   │   ├── test_crash_detector.py
│   │   ├── test_tag_state_machine.py
│   │   └── test_render_smoke.py
│   └── integration/
│       ├── test_simple_env_contract.py
│       ├── test_scoring_canary.py
│       ├── test_team_env_canary.py
│       └── test_warm_start.py
├── demo/
│   ├── menu.py                  # gain a new entry: ("scenarios", ...)
│   └── scenarios.py             # NEW — scripted 1v1 narratives A + B
├── pyproject.toml               # NEW — minimal, pytest config only
├── requirements.txt             # +pytest
├── Makefile                     # see "Makefile changes" below
└── scripts/                     # all seven check_*.py DELETED
```

`tests/` (plural) follows pytest's auto-discovery convention. Two subdirs because the unit / integration split also corresponds to the slow-test split — `tests/unit/` runs in milliseconds with no episode loops, `tests/integration/` runs real env steps and is marked `slow`.

## pytest configuration

`pyproject.toml` (new file, no project metadata, only `[tool.pytest.ini_options]`):

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: integration tests with real episode loops (full canaries, warm-start)",
]
```

- `pythonpath = ["."]` removes the per-file `sys.path.insert(0, str(Path(__file__).parent.parent))` shim that every current `check_*.py` carries. Tests use plain `from envs.quidditch.team_env import ...`.
- `testpaths = ["tests"]` scopes auto-discovery so a stray `def test_*` in `scripts/` or `demo/` doesn't get collected.
- `--strict-markers` makes typos in marker names errors rather than silent skips.
- `--ra` shows summary lines for skipped tests so the warm-start skip is visible.

`requirements.txt`: append `pytest`. No version pin.

## Unit tests

Each is a direct port of one `check_*` script. Conversion rules:

- Every existing `assert` stays; manual `raise AssertionError("...")` becomes `assert ..., "..."`.
- `print` statements that confirmed success are deleted (silent success is the pytest convention).
- `print` statements that document state mid-test are deleted unless they appear in an assertion message.
- The `def case_X(): ...` + `if __name__: case_X()` pattern becomes `def test_X():`.
- Each test owns its own world; no module-level state.

### `tests/unit/test_team_mjcf.py`

One test: `test_team_env_mjcf_geoms_present_with_correct_collision_bits()`.

- Builds the world from the same fragment list as `check_team_mjcf.py` (cf2x assets with collision meshes; red_0 + blue_0 with `with_tag_sphere=True`; arena wall at `ARENA_RADIUS, ARENA_WALL_HEIGHT`; one hoop at `HOOP_CENTER`).
- Resolves the five geom names: `red_0_probe`, `blue_0_probe`, `red_0_tag_sphere`, `blue_0_tag_sphere`, `hoop_0_score_tube`. Asserts each has `geom_contype == 0 and geom_conaffinity == 0`.
- Iterates `arena_wall_seg_00` through `arena_wall_seg_15`; asserts each has `geom_contype == 1 and geom_conaffinity == 1`.
- Calls `world.disconnect()` in a `finally` block.

### `tests/unit/test_crash_detector.py`

Three tests, one per case in `check_team_crash.py`:

- `test_drone_into_wall_above_threshold` — pin blue at origin, set red at `(ARENA_RADIUS - 0.1, 0, 1.0)` with `vel=(2.0, 0, 0)`. Step up to 20 times. Assert `events.wall["red_0"] > 0.5` once contact lands. Fail if no contact in 20 steps.
- `test_drone_drone_below_threshold_no_crash` — set red at `(-0.03, 0, 1.0) vel=(0.25, 0, 0)`, blue at `(0.03, 0, 1.0) vel=(-0.25, 0, 0)`. Step 30 times. Assert at least one drone-drone contact occurred AND no contact ever exceeded `CRASH_VEL_THR`.
- `test_drone_drone_above_threshold_crash` — set red at `(-0.5, 0, 1.0) vel=(2.0, 0, 0)`, blue at `(0.5, 0, 1.0) vel=(-2.0, 0, 0)`. Step up to 120 times. Assert at least one drone-drone contact with `vrel > CRASH_VEL_THR` occurred.

The `_build_world`, `_qa`, `_set_state` helpers move into `tests/conftest.py` as plain functions: `build_team_world() -> tuple[World, CrashDetector]`, `qpos_addr(world, body_name) -> int`, `set_body_state(world, body, pos, vel)`.

### `tests/unit/test_tag_state_machine.py`

**One test** exercising all 6 phases (matches the script's structure today). Splitting into 6 tests would force re-running phase 1–5 setup before phase 6, multiplying runtime for no gain.

Test name: `test_tag_state_machine_full_lifecycle`.

Reproduces all 6 phases from `check_team_tag.py`:
1. Phase 1 (blue at 0.5 m, outside) → IDLE expected.
2. Phase 2 (blue at 0.15 m, inside) → entry pulse, `blue_0` reward > 4.9.
3. Phase 3 (blue stays at 0.15 m) → duration only, reward in (0, 0.1).
4. Phase 4 (blue jumps to 0.5 m) → exit.
5. Phase 5 (blue jumps back during cooldown) → duration only (no entry pulse), reward in (0, 0.1).
6. Phase 6 (leave, wait `cooldown_ticks + 1`, re-enter) → fresh entry pulse, reward > 4.9.

Helpers `_pin_red`, `_set_blue` come from `tests/conftest.py` (along with the body-name addressing). The `env._red_takeoff_grace = 0` override at start stays.

### `tests/unit/test_render_smoke.py`

New, didn't exist before. One test: `test_offscreen_render_one_frame_no_crash()`.

- Builds a minimal world (single drone, hoop, arena wall — same fragment list as the simple env's MJCF).
- Constructs `mujoco.Renderer(model, height=240, width=320)`.
- Calls `mujoco.mj_forward(model, data)`, then `renderer.update_scene(data)`, then `renderer.render()`.
- Asserts the returned array has shape `(240, 320, 3)` and dtype `np.uint8`.

Catches the bug class where a mesh path breaks, an asset is missing, or a camera/material parses fail at render time — none of which the headless episode loops would catch.

### Shared helpers — `tests/conftest.py`

Plain helper functions, not pytest fixtures (they're called imperatively):

```python
def build_team_world() -> tuple[World, CrashDetector]: ...
def qpos_addr(world, body_name: str) -> int: ...
def set_body_state(world, body: str, pos, vel=(0, 0, 0)) -> None: ...
```

Fixtures only get added if some test wants automatic teardown — current tests handle `world.disconnect()` themselves so no fixtures needed in v1.

## Integration tests

All four carry `@pytest.mark.slow`. Default `make test` includes them; `make test-fast` skips them via path filter (runs `tests/unit/` only).

### `tests/integration/test_simple_env_contract.py`

Two tests, ported from `check_env.py` parts 1 & 2:

- `test_sb3_check_env_passes()` — calls `stable_baselines3.common.env_checker.check_env(QuidditchSimpleEnv(randomise_start=False), warn=True)`. No episode loop.
- `test_zero_action_episode_runs_10_steps()` — drives 10 zero-action steps. Asserts no exception, `obs.shape == (16,)`, `obs.dtype == np.float32`, all rewards finite.

### `tests/integration/test_scoring_canary.py`

One test: `test_scripted_flyaway_scores_through_hoop()`.

The two-phase scripted policy (climb-to-approach-point at `[0, 0, HOOP_CENTER[2]]`, then push-through at `HOOP_CENTER + 0.7 * HOOP_OUTWARD_NORMAL`) ports verbatim from `check_env.py` part 3.

Asserts (canary fingerprint from `brain/index.md`):
- `info["scored"] is True` at the terminating step.
- `step == 434`.
- `total_reward == pytest.approx(7.3837, abs=1e-4)`.

### `tests/integration/test_team_env_canary.py`

One test: `test_beeline_red_vs_beeline_blue_canary()`.

The `beeline_action(obs) = clip([obs[12], obs[13], 0.0, obs[14]], -1, 1)` policy ports verbatim. `env.reset(seed=42)`. Run until `env.agents` is empty.

Asserts the locked fingerprint from `check_team_env.py`'s docstring:
- At step 176: `info["red_0"]["tag_entry"] is True`, `rew["blue_0"] == pytest.approx(+5.019, abs=1e-3)`, `rew["red_0"] == pytest.approx(-5.023, abs=1e-3)`.
- At step 684: `terminated`, `info["red_0"]["scored"] is True`, `rew["blue_0"] == pytest.approx(-10.005, abs=1e-3)`, `rew["red_0"] == pytest.approx(+9.999, abs=1e-3)`.
- After loop exit: `red_total == pytest.approx(+1.931, abs=1e-3)`, `blue_total == pytest.approx(-7.618, abs=1e-3)`.

Tolerance choice: `abs=1e-3` accepts platform float-jitter while still catching real physics changes. Existing fingerprints have been stable to 1e-3 across all reported runs.

### `tests/integration/test_warm_start.py`

Skip-if-no-artifact pattern. Module top:

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    "OLD_MODEL" not in os.environ,
    reason="set OLD_MODEL=models/<run>/best_model to run warm-start regression",
)
```

One test: `test_warm_started_policy_matches_old_on_obs_prefix()`.

Reads `os.environ["OLD_MODEL"]` for the path. Same logic as `check_team_warm.py`:

- Load old PPO from path.
- Build `OpponentControlledEnv(QuidditchTeamEnv(...), learner_id="red_0", opponent=BeelineBlue())` wrapped in `DummyVecEnv`.
- Call `warm_start_ppo(old_checkpoint=path, new_env=..., new_input_dim=22, old_input_dim=16, new_dim_init_scale=0.01)`.
- Sample 64 obs (with `obs[..., 16:] = 0.0`); compare deterministic `predict` outputs old vs new.
- Assert `mean_diff < 0.1`.

## Makefile changes

**Removed** (and removed from `.PHONY`):
- `check-sim`, `check-gui`, `team-check`, `team-check-mjcf`, `team-check-tag`, `team-check-crash`, `team-check-warm`.

**Added:**

```make
test: ## ✅ Run all tests (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Run unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit

test-warm: ## ✅ Warm-start preserves single-agent behavior  OLD=models/<run>
	@test -n "$(OLD)" || { echo "ERROR: OLD=models/<run> required"; exit 1; }; \
	 OLD_MODEL="$(OLD)/best_model" $(PYTHON) -m pytest tests/integration/test_warm_start.py
```

`test` is the daily driver (~30–60s wall-clock total). `test-fast` is sub-second for tight unit-iteration loops. `test-warm` keeps the existing `OLD=...` env-var contract from `team-check-warm` so muscle memory doesn't break; it sets `OLD_MODEL` for pytest to read.

`make demo` is unchanged — the new "scenarios" entry slots into the existing menu.

## Scripted GUI scenarios

`demo/scenarios.py` exposes `main()` (the `demo/menu.py` calling convention).

### Top-level shape

```python
"""Scripted 1v1 narratives for visual review of tag/crash/score behavior.

Two scenarios run back-to-back in a single MuJoCo viewer:
  A) Tagged out — Blue tags Red twice, then rams it down.
  B) Through despite the tag — Red gets tagged once but still scores.

Run via:  make demo  → pick "scenarios"
"""

# Toggle to draw colored markers in the 3D scene on each event.
SHOW_EVENT_MARKERS = False

def main() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False, episode_seconds=30.0))
    env.reset(seed=42)
    try:
        run_scenario_a(env)
        idle_pause(env, seconds=1.5)
        env.reset(seed=42)
        run_scenario_b(env)
        idle_until_user_quits(env)
    finally:
        env.close()
```

### Single env / single viewer

The team env is constructed once (with whatever `render_mode` argument exposes the live MuJoCo viewer — verified during implementation against the existing `QuidditchTeamEnv` API). One viewer stays open across both scenarios. Between A and B: `env.reset(seed=42)` — same seed both times for reproducibility, viewer keeps drawing.

### Scripted policies (not waypoints)

Each scenario implements `red_policy(obs) -> action` and `blue_policy(obs) -> action`. They read positions from `obs[9:12]` (and the other drone's position from team-env obs columns) and return delta-setpoint actions in `[-1, 1]^4`. Tuned offline once until they hit the desired beats; committed deterministic.

### Scenario A: Tagged out (~25 s sim time)

Beats:
1. Red beelines toward the hoop.
2. Blue intercepts on the approach line → enters tag sphere → **tag #1** (Blue +5, Red −5).
3. Blue retreats just past tag-sphere boundary → cooldown elapses.
4. Blue re-intercepts → **tag #2** (Blue +5, Red −5).
5. Blue rams Red at >`CRASH_VEL_THR` → drone-drone crash terminates the episode.

### Scenario B: Through despite the tag (~10 s sim time)

Beats:
1. Red beelines toward the hoop.
2. Blue intercepts once at mid-arena → **tag #1** (Blue +5, Red −5).
3. Blue peels off / falls behind (cooldown begins).
4. Red continues to the hoop → **score** (Red +10), terminates.

### Reward display (Option A — terminal)

A helper `log_event(t, label, rew, totals)` prints one line per detected event. Events come from the `info` dict returned by `env.step` — reading `info[agent]["tag_entry"]`, `info[agent]["scored"]`, `info[agent]["drone_drone_crash"]`, etc. Format:

```
[Scenario A: Tagged out]
[t= 4.32s] TAG ENTRY        red −5.000  blue +5.000   totals: red −5.0  blue +5.0
[t= 6.18s] TAG ENTRY        red −5.000  blue +5.000   totals: red −10.0 blue +10.0
[t= 8.91s] DRONE-DRONE CRASH  vrel=2.34 m/s
           red −20.000  blue −5.000   FINAL: red −30.0  blue +5.0
```

### Marker hook (Option B — opt-in)

`draw_event_marker(viewer, kind, pos)`. When `SHOW_EVENT_MARKERS=False` (default), no-op. When `True`, pushes a one-tick `mjvGeom` (sphere) into `viewer.user_scn.geoms`: red sphere at tag, gold at score, large red flash at crash. The function exists in v1 as a no-op so flipping the constant in v2 needs no further plumbing.

### Determinism

`seed=42` for both `env.reset(seed=42)` calls. Scripts are deterministic functions of `obs`, so the same seed reproduces identical beats. Side benefit: the scenarios serve as ad-hoc regression — if the tag stops landing where you expect after a physics tweak, that's a signal worth investigating.

### `demo/menu.py` change

One new tuple appended to `DEMOS`:

```python
("scenarios", "Scripted 1v1: tag-out + score-despite-tag", "demo.scenarios"),
```

### Implementation risk

The scripted policies for both scenarios will need iteration to hit their beats reliably. Budget ~1 hour of tuning per scenario. Fallback if scripted policies prove too flaky: use mid-episode `qpos` writes ("puppeteer mode") to nudge drones to the right position — same trick `check_team_tag.py` uses today for FSM phase setup. This warps visual realism, so use only as last resort.

## Build sequence

1. **Wire pytest first.** Add `pyproject.toml`, `pytest` to `requirements.txt`, an empty `tests/conftest.py`, and one trivial smoke test (`def test_imports(): import envs.quidditch.team_env`). Confirm `pytest` runs green. Provides the rails before any real test ports.
2. **Port unit tests, in this order:** `test_team_mjcf` (no helpers needed) → extract `tests/conftest.py` helpers (`build_team_world`, `qpos_addr`, `set_body_state`) → `test_crash_detector` → `test_tag_state_machine` → `test_render_smoke`. Each port verified by running just that file.
3. **Port integration tests:** `test_simple_env_contract` (parts 1+2 of `check_env`, no canary number) → `test_scoring_canary` (canary number from `brain/index.md`) → `test_team_env_canary` (canary numbers from script docstring) → `test_warm_start` (skip-if-no-artifact).
4. **Verify each ported test passes once before deleting the source `check_*.py`.** Port-and-delete is one commit per script — never two scripts in flight.
5. **Add `demo/scenarios.py`** with terminal-only output (Option A); add the menu entry; tune scripted policies until both scenarios hit their beats reliably under `seed=42`.
6. **Update Makefile last.** Remove old `check-*` targets, add `test`/`test-fast`/`test-warm`. Run `make help` to confirm autogenerated help is clean.
7. **Update `brain/`** — `brain/changelog.md` entry; `brain/decisions.md` entry on the test-architecture choice (why pytest, why two subdirs, why scripted scenarios over policy-driven for v1); `brain/tasks.md` if any tracked items resolve.

## Out of scope (explicitly)

- **CI configuration** (GitHub Actions etc.). Pytest-friendly tests are the prerequisite for CI; the workflow file is a separate piece of work.
- **Image / pixel regression** of rendered frames. `test_render_smoke` only asserts shape/dtype.
- **Marker rendering implementation** (Option B from reward-display). The flag and function stub exist in v1; the `mjvGeom` writes are deferred.
- **Trained-policy demos** for 1v1. The user has noted these will be added later as additional `demo/` entries; the scripted scenarios cover the visual-check role until then.
