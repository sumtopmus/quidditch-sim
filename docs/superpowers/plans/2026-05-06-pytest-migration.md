# Pytest Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For commits:** the user (shurioque) signs git commits manually with a hardware key. **Never run `git commit` yourself.** Every commit step in this plan is a *manual user gate* — show the ready-to-paste command and wait for the user to confirm the commit landed before continuing to the next task.

**Goal:** Replace seven hand-rolled `scripts/check_*.py` scripts with a pytest suite under `tests/` (unit + integration, slow-marker for canaries), and replace the lone visual check `make check-gui` with two scripted 1v1 narratives (`demo/scenarios.py`) playing back-to-back in one MuJoCo viewer.

**Architecture:** `tests/` is the new pytest root with `unit/` (millisecond asserts, no episode loops) and `integration/` (real env step loops, marked `slow`) subdirs. Shared helpers in `tests/conftest.py`. Visual narratives ship as a new entry in `demo/menu.py`. `pyproject.toml` provides minimal pytest config (`pythonpath`, `testpaths`, `markers`). All `check_*.py` scripts are deleted; corresponding Makefile targets are removed and replaced by `make test` / `make test-fast` / `make test-warm`.

**Tech Stack:** pytest, mujoco, stable-baselines3, pettingzoo (already in `requirements.txt`); `pytest` is the only new dependency.

**Worktree:** All work happens in `worktrees/feature/pytest-migration/` on branch `feature/pytest-migration` (already created off `develop`). Spec is at `docs/superpowers/specs/2026-05-06-pytest-migration-design.md` (untracked at start; gets committed in Task 1).

**Working directory:** All shell commands assume the current working directory is the worktree root (`worktrees/feature/pytest-migration`). The very first task includes the `cd`. Subsequent tasks assume you stayed there.

---

## File Structure

**New files:**
- `pyproject.toml` — minimal `[tool.pytest.ini_options]` only, no project metadata.
- `tests/__init__.py` — empty (avoids one pytest discovery gotcha; cheap insurance).
- `tests/conftest.py` — shared imperative helpers `build_team_world`, `qpos_addr`, `set_body_state`. Plain functions, not pytest fixtures.
- `tests/unit/__init__.py` — empty.
- `tests/unit/test_team_mjcf.py` — port of `check_team_mjcf.py`, one test.
- `tests/unit/test_crash_detector.py` — port of `check_team_crash.py`, three tests.
- `tests/unit/test_tag_state_machine.py` — port of `check_team_tag.py`, one test (6 phases inline).
- `tests/unit/test_render_smoke.py` — new, one offscreen-render assert.
- `tests/integration/__init__.py` — empty.
- `tests/integration/test_simple_env_contract.py` — port of `check_env.py` parts 1+2, two tests.
- `tests/integration/test_scoring_canary.py` — port of `check_env.py` part 3, exact-value canary.
- `tests/integration/test_team_env_canary.py` — port of `check_team_env.py`, exact-value canary.
- `tests/integration/test_warm_start.py` — port of `check_team_warm.py`, skip-if-no-`OLD_MODEL`.
- `demo/scenarios.py` — scripted 1v1 narrative pair, terminal reward log.

**Modified files:**
- `requirements.txt` — append `pytest`.
- `Makefile` — remove `check-sim`, `check-gui`, `team-check`, `team-check-mjcf`, `team-check-tag`, `team-check-crash`, `team-check-warm`. Add `test`, `test-fast`, `test-warm`.
- `demo/menu.py` — append `("scenarios", ...)` to `DEMOS`.
- `brain/index.md`, `brain/changelog.md`, `brain/decisions.md`, `brain/tasks.md` — final-task housekeeping.

**Deleted files (one per task, never two in flight):**
- `scripts/check_env.py`
- `scripts/check_team_env.py`
- `scripts/check_team_mjcf.py`
- `scripts/check_team_crash.py`
- `scripts/check_team_tag.py`
- `scripts/check_team_warm.py`

`scripts/train_*.py`, `scripts/eval_*.py`, `scripts/lineage.py`, `scripts/_train_common.py`, `scripts/callbacks.py` are **not** touched.

---

## Task 1: Wire pytest infrastructure + commit the spec

**Goal:** Establish the pytest rails with one trivial smoke test before any real test ports. Also: commit the spec doc that's currently sitting untracked in this worktree.

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py` (empty for now — helpers added in Task 3)
- Create: `tests/test_imports.py` (smoke test)
- Modify: `requirements.txt`

- [ ] **Step 1: Move into the worktree**

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration"
git status --short
```

Expected: shows the untracked spec at `docs/superpowers/specs/2026-05-06-pytest-migration-design.md` and nothing else.

- [ ] **Step 2: Install pytest in the conda env**

```bash
conda run -n uav python -m pip install pytest
```

Expected: pytest installs (or "Requirement already satisfied").

- [ ] **Step 3: Add pytest to requirements.txt**

Append one line at the end of `requirements.txt`:

```
pytest
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: integration tests with real episode loops (full canaries, warm-start)",
]
```

- [ ] **Step 5: Create empty `tests/__init__.py` and `tests/conftest.py`**

Both files are zero bytes for now. `conftest.py` will gain helpers in Task 3.

```bash
mkdir -p tests
: > tests/__init__.py
: > tests/conftest.py
```

- [ ] **Step 6: Create the trivial smoke test**

Create `tests/test_imports.py`:

```python
"""Smoke test — confirms the project's top-level packages import cleanly.

Lives at the tests/ root (not under unit/ or integration/) because it
exercises pytest discovery itself; if this can't find the modules,
nothing else will.
"""
from __future__ import annotations


def test_imports_envs_quidditch() -> None:
    import envs.quidditch.simple_env  # noqa: F401
    import envs.quidditch.team_env    # noqa: F401


def test_imports_core() -> None:
    import core.world          # noqa: F401
    import core.drone.cf2x     # noqa: F401
```

- [ ] **Step 7: Run pytest to verify the rails work**

```bash
conda run -n uav python -m pytest -v
```

Expected: 2 passed in <1s. If you see `ModuleNotFoundError`, double-check that `pyproject.toml` has `pythonpath = ["."]` and that you're running from the worktree root.

- [ ] **Step 8: User runs commit (manual — hardware-key signing)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add docs/superpowers/specs/2026-05-06-pytest-migration-design.md \
      pyproject.toml requirements.txt \
      tests/__init__.py tests/conftest.py tests/test_imports.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: wire pytest scaffold + commit pytest-migration design spec

Introduces tests/ as the pytest root with a minimal pyproject.toml
(pythonpath, testpaths, markers), a smoke import test, and adds pytest
to requirements.txt.  Bundles the design spec for the migration so the
rest of the work has rails plus context to land against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for the user to confirm the commit landed before moving on.

---

## Task 2: Port test_team_mjcf (delete check_team_mjcf.py + Makefile target)

**Goal:** First real port. Single test, no shared helpers required (this comes before the conftest extraction so we lock in a working baseline).

**Files:**
- Create: `tests/unit/__init__.py` (empty)
- Create: `tests/unit/test_team_mjcf.py`
- Delete: `scripts/check_team_mjcf.py`
- Modify: `Makefile` (remove `team-check-mjcf` target + its `.PHONY` entry)

- [ ] **Step 1: Create empty `tests/unit/__init__.py`**

```bash
mkdir -p tests/unit
: > tests/unit/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_team_mjcf.py`:

```python
"""MJCF assembly: required geoms exist with correct contype/conaffinity bits.

Ported from scripts/check_team_mjcf.py.
"""
from __future__ import annotations

import mujoco

from core.drone.cf2x import cf2x_assets, cf2x_fragment
from core.world import World
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment


def test_team_env_mjcf_geoms_present_with_correct_collision_bits() -> None:
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True,
                      with_tag_sphere=True, tag_sphere_rgba=(1.0, 0.0, 0.0, 0.15)),
        cf2x_fragment(prefix="blue_0", with_collisions=True,
                      with_tag_sphere=True, tag_sphere_rgba=(0.0, 0.0, 1.0, 0.15)),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    try:
        m = world.model

        def gid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert i >= 0, f"missing geom: {name!r}"
            return i

        for name in (
            "red_0_probe",
            "blue_0_probe",
            "red_0_tag_sphere",
            "blue_0_tag_sphere",
            "hoop_0_score_tube",
        ):
            g = gid(name)
            assert m.geom_contype[g]    == 0, f"{name}: contype != 0"
            assert m.geom_conaffinity[g] == 0, f"{name}: conaffinity != 0"

        for i in range(16):
            name = f"arena_wall_seg_{i:02d}"
            g = gid(name)
            assert m.geom_contype[g]    == 1, f"{name}: contype != 1"
            assert m.geom_conaffinity[g] == 1, f"{name}: conaffinity != 1"
    finally:
        world.disconnect()
```

- [ ] **Step 3: Run the test — expected to PASS first try (it's a port)**

```bash
conda run -n uav python -m pytest tests/unit/test_team_mjcf.py -v
```

Expected: 1 passed. If it fails, the source script `scripts/check_team_mjcf.py` was your reference — diff the two files for typos.

- [ ] **Step 4: Confirm the original script still works (sanity check before delete)**

```bash
conda run -n uav python scripts/check_team_mjcf.py
```

Expected: prints `OK team-env MJCF: ngeom=...`.

- [ ] **Step 5: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_team_mjcf.py
```

(or use `rm scripts/check_team_mjcf.py` followed by `git add` later — `git rm` stages the deletion immediately.)

- [ ] **Step 6: Remove `team-check-mjcf` from the Makefile**

In `Makefile`:

1. In the `.PHONY` line near the top, remove the `team-check-mjcf` token (and the trailing space if it leaves a double-space).
2. Delete the three lines:
   ```make
   team-check-mjcf: ## ✅ Team-env MJCF assembly check
   	@$(PYTHON) scripts/check_team_mjcf.py

   ```
   (and the blank line that followed).

- [ ] **Step 7: Verify `make help` still parses cleanly**

```bash
make help
```

Expected: prints the full target list, no `team-check-mjcf` entry, no errors.

- [ ] **Step 8: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 3 passed (2 imports + 1 mjcf).

- [ ] **Step 9: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/unit/__init__.py tests/unit/test_team_mjcf.py Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_team_mjcf to pytest, remove old script + target

One-test port of scripts/check_team_mjcf.py to tests/unit/test_team_mjcf.py.
Source script and team-check-mjcf Makefile target removed in the same
commit (port-and-delete is one commit per script).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for the user to confirm before moving on.

---

## Task 3: Extract shared helpers into tests/conftest.py + port test_crash_detector

**Goal:** Establish the imperative-helper pattern in `conftest.py` and use it to land the second unit test. Same commit covers both because `test_crash_detector` is the first consumer of the helpers — extracting them without a consumer is YAGNI.

**Files:**
- Modify: `tests/conftest.py` (was empty; add helpers)
- Create: `tests/unit/test_crash_detector.py`
- Delete: `scripts/check_team_crash.py`
- Modify: `Makefile` (remove `team-check-crash`)

- [ ] **Step 1: Add helpers to `tests/conftest.py`**

Replace the (empty) `tests/conftest.py` with:

```python
"""Shared imperative helpers for unit + integration tests.

Plain functions, not pytest fixtures — tests call them imperatively because
no test in the suite needs automatic teardown beyond what try/finally
already gives them.
"""
from __future__ import annotations

import mujoco

from core.drone.cf2x import cf2x_assets, cf2x_fragment
from core.world import World
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)
from envs.quidditch.crash import CrashDetector
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment


def build_team_world() -> tuple[World, CrashDetector]:
    """Build a two-drone team world (red_0 + blue_0) and its crash detector."""
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True, with_tag_sphere=True),
        cf2x_fragment(prefix="blue_0", with_collisions=True, with_tag_sphere=True),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    return world, CrashDetector(world, ["red_0", "blue_0"])


def qpos_addr(world: World, body_name: str) -> int:
    """qpos address of body's first joint (free joint = 7 entries: xyz + quat)."""
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt = int(world.model.body_jntadr[bid])
    return int(world.model.jnt_qposadr[jnt])


def set_body_state(
    world: World,
    body: str,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Write body's free-joint position, identity quat, and linear velocity."""
    qa = qpos_addr(world, body)
    world.data.qpos[qa : qa + 3] = pos
    world.data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body)
    jnt = int(world.model.body_jntadr[bid])
    qva = int(world.model.jnt_dofadr[jnt])
    world.data.qvel[qva : qva + 3] = vel
    world.data.qvel[qva + 3 : qva + 6] = 0.0
```

- [ ] **Step 2: Write the three crash-detector tests**

Create `tests/unit/test_crash_detector.py`:

```python
"""CrashDetector classification: drone-into-wall, slow drone-drone (no crash),
fast drone-drone (crash). Drives contacts via manual qpos/qvel writes.

Ported from scripts/check_team_crash.py.
"""
from __future__ import annotations

import mujoco

from envs.quidditch.constants import ARENA_RADIUS, CRASH_VEL_THR
from tests.conftest import build_team_world, set_body_state


def test_drone_into_wall_above_threshold() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "blue_0", pos=(0.0, 0.0, 1.5))
        set_body_state(world, "red_0",  pos=(ARENA_RADIUS - 0.1, 0.0, 1.0),
                       vel=(2.0, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        for _ in range(20):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.wall["red_0"] > 0.0:
                assert ev.wall["red_0"] > 0.5, (
                    f"expected meaningful wall vrel, got {ev.wall['red_0']:.3f}"
                )
                return
        raise AssertionError("never made contact with wall in 20 steps")
    finally:
        world.disconnect()


def test_drone_drone_below_threshold_no_crash() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "red_0",  pos=(-0.03, 0.0, 1.0), vel=( 0.25, 0.0, 0.0))
        set_body_state(world, "blue_0", pos=( 0.03, 0.0, 1.0), vel=(-0.25, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        saw_contact = False
        saw_above_thr = False
        for _ in range(30):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.drone_drone is not None:
                saw_contact = True
                if ev.drone_drone[2] > CRASH_VEL_THR:
                    saw_above_thr = True
        assert saw_contact, "expected at least one drone-drone contact"
        assert not saw_above_thr, "did not expect any |v_rel| > CRASH_VEL_THR"
    finally:
        world.disconnect()


def test_drone_drone_above_threshold_crash() -> None:
    world, detector = build_team_world()
    try:
        set_body_state(world, "red_0",  pos=(-0.5, 0.0, 1.0), vel=( 2.0, 0.0, 0.0))
        set_body_state(world, "blue_0", pos=( 0.5, 0.0, 1.0), vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(world.model, world.data)
        for _ in range(120):
            mujoco.mj_step(world.model, world.data)
            ev = detector.events()
            if ev.drone_drone is not None and ev.drone_drone[2] > CRASH_VEL_THR:
                return
        raise AssertionError("never observed |v_rel| > CRASH_VEL_THR")
    finally:
        world.disconnect()
```

- [ ] **Step 3: Run the new tests**

```bash
conda run -n uav python -m pytest tests/unit/test_crash_detector.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Sanity-check the original script still works**

```bash
conda run -n uav python scripts/check_team_crash.py
```

Expected: prints `All crash sub-cases PASSED.`.

- [ ] **Step 5: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_team_crash.py
```

- [ ] **Step 6: Remove `team-check-crash` from the Makefile**

Same procedure as Task 2: drop `team-check-crash` from the `.PHONY` line and delete the target's three lines from the Makefile (target line + recipe line + trailing blank).

- [ ] **Step 7: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 6 passed total (2 imports + 1 mjcf + 3 crash).

- [ ] **Step 8: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/conftest.py tests/unit/test_crash_detector.py Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_team_crash to pytest, extract shared world-builder helpers

Three-test port of scripts/check_team_crash.py to
tests/unit/test_crash_detector.py.  Shared imperative helpers
(build_team_world, qpos_addr, set_body_state) land in tests/conftest.py
since this is the first consumer.  Source script and team-check-crash
Makefile target removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 4: Port test_tag_state_machine

**Goal:** Port `check_team_tag.py` — the 6-phase tag-state-machine walkthrough — to a single pytest function. Phases are sequential (each builds on the prior state), so one test, not six.

**Files:**
- Create: `tests/unit/test_tag_state_machine.py`
- Delete: `scripts/check_team_tag.py`
- Modify: `Makefile` (remove `team-check-tag`)

- [ ] **Step 1: Write the test**

Create `tests/unit/test_tag_state_machine.py`:

```python
"""Tag state machine: enter → duration → exit → cooldown → re-enter
(no entry pulse during cooldown) → cooldown end → exit → re-enter
(entry pulse fires).

Ported from scripts/check_team_tag.py.  Drives the FSM by manually
positioning Blue inside/outside Red's tag sphere with qpos writes,
pinning Red at origin between steps.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig


def _step_zero(env: QuidditchTeamEnv) -> tuple[dict, dict]:
    """One zero-action step for both agents; returns (rewards, info)."""
    a = {ag: np.zeros(4, dtype=np.float32) for ag in env.agents}
    _obs, rew, _term, _trunc, info = env.step(a)
    return rew, info


def _set_blue(env: QuidditchTeamEnv, x: float) -> None:
    """Place blue_0 at (x, 0, 1) and run mj_forward so derived state updates."""
    bid = mujoco.mj_name2id(env._world.model, mujoco.mjtObj.mjOBJ_BODY, "blue_0")
    jnt = int(env._world.model.body_jntadr[bid])
    qa  = int(env._world.model.jnt_qposadr[jnt])
    env._world.data.qpos[qa : qa + 3] = [x, 0.0, 1.0]
    mujoco.mj_forward(env._world.model, env._world.data)


def _pin_red(env: QuidditchTeamEnv) -> None:
    """Re-pin red_0 at origin; called after each step to undo physics drift."""
    bid = mujoco.mj_name2id(env._world.model, mujoco.mjtObj.mjOBJ_BODY, "red_0")
    jnt = int(env._world.model.body_jntadr[bid])
    qa  = int(env._world.model.jnt_qposadr[jnt])
    env._world.data.qpos[qa : qa + 3] = [0.0, 0.0, 1.0]
    mujoco.mj_forward(env._world.model, env._world.data)


def test_tag_state_machine_full_lifecycle() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        env.reset(seed=42)
        env._red_takeoff_grace = 0
        _pin_red(env)

        # Phase 1: blue at 0.5 m (outside) — IDLE.
        _set_blue(env, 0.5)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and not info["red_0"]["tag_during"], (
            f"phase1 expected IDLE, got {info}"
        )

        # Phase 2: blue jumps to 0.15 m (inside) — entry pulse fires.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase2 expected entry, got {info}"
        )
        assert rew["blue_0"] > 4.9, f"phase2 expected blue +5, got {rew['blue_0']:.3f}"

        # Phase 3: blue stays at 0.15 m for one more step — duration only.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase3 expected duration only, got {info}"
        )
        assert 0.0 < rew["blue_0"] < 0.1, (
            f"phase3 expected ~+0.02, got {rew['blue_0']:.3f}"
        )

        # Phase 4: blue jumps to 0.5 m (outside) — exit.
        _set_blue(env, 0.5)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_during"], f"phase4 expected exit, got {info}"

        # Phase 5: blue jumps back to 0.15 m DURING cooldown — duration only,
        # no entry pulse.
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert not info["red_0"]["tag_entry"] and info["red_0"]["tag_during"], (
            f"phase5 expected duration only (cooldown), got {info}"
        )
        assert 0.0 < rew["blue_0"] < 0.1, (
            f"phase5 expected ~+0.02, got {rew['blue_0']:.3f}"
        )

        # Phase 6: leave, wait full cooldown, re-enter — fresh entry pulse.
        _set_blue(env, 0.5)
        cooldown_steps = env._cooldown_ticks + 1
        for _ in range(cooldown_steps):
            _set_blue(env, 0.5)
            _step_zero(env)
            _pin_red(env)
        _set_blue(env, 0.15)
        rew, info = _step_zero(env); _pin_red(env)
        assert info["red_0"]["tag_entry"], (
            f"phase6 expected fresh entry pulse, got {info}"
        )
        assert rew["blue_0"] > 4.9, f"phase6 expected blue +5, got {rew['blue_0']:.3f}"
    finally:
        env.close()
```

- [ ] **Step 2: Run the new test**

```bash
conda run -n uav python -m pytest tests/unit/test_tag_state_machine.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Sanity-check the original script**

```bash
conda run -n uav python scripts/check_team_tag.py
```

Expected: prints `OK tag state machine: all 6 phases passed`.

- [ ] **Step 4: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_team_tag.py
```

- [ ] **Step 5: Remove `team-check-tag` from the Makefile**

Drop `team-check-tag` from `.PHONY`; delete its target/recipe block.

- [ ] **Step 6: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 7 passed (2 imports + 1 mjcf + 3 crash + 1 tag).

- [ ] **Step 7: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/unit/test_tag_state_machine.py Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_team_tag to pytest tag-FSM lifecycle test

One-test port of scripts/check_team_tag.py to
tests/unit/test_tag_state_machine.py — phases stay inline (sequential
state) rather than splitting across functions.  Source script and
team-check-tag Makefile target removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 5: Add test_render_smoke (new — no source script)

**Goal:** New offscreen-render test that didn't exist before. Catches the bug class where a mesh path breaks, an asset is missing, or a camera/material parses fail — none of which the headless episode loops would catch.

**Files:**
- Create: `tests/unit/test_render_smoke.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_render_smoke.py`:

```python
"""One-frame offscreen render smoke test — catches asset/mesh/camera
regressions that headless episode loops never exercise.

New test, no source script.  Builds a minimal single-drone world,
renders one 240x320 frame, asserts shape + dtype.
"""
from __future__ import annotations

import mujoco
import numpy as np

from core.drone.cf2x import cf2x_assets, cf2x_fragment
from core.world import World
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment


def test_offscreen_render_one_frame_no_crash() -> None:
    fragments = [
        cf2x_assets(with_collision_meshes=False),
        cf2x_fragment(),  # default prefix "drone", visual-only
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=False),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    try:
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        try:
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data)
            frame = renderer.render()
            assert frame.shape == (240, 320, 3), f"unexpected shape {frame.shape}"
            assert frame.dtype == np.uint8, f"unexpected dtype {frame.dtype}"
        finally:
            renderer.close()
    finally:
        world.disconnect()
```

- [ ] **Step 2: Run the new test**

```bash
conda run -n uav python -m pytest tests/unit/test_render_smoke.py -v
```

Expected: 1 passed in <2s. **If you see a GLFW or "GLContext could not be created" error**, the offscreen renderer needs a backend. Try:

```bash
MUJOCO_GL=osmesa conda run -n uav python -m pytest tests/unit/test_render_smoke.py -v
# or
MUJOCO_GL=egl    conda run -n uav python -m pytest tests/unit/test_render_smoke.py -v
```

If one of those works, we'll set the env var in the Makefile target rather than baking it into the test (Task 14). On macOS the default backend usually works with `mjpython`, but `python` might need `MUJOCO_GL=osmesa`.

- [ ] **Step 3: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 8 passed (2 imports + 1 mjcf + 3 crash + 1 tag + 1 render).

- [ ] **Step 4: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/unit/test_render_smoke.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: add offscreen render smoke test

New test in tests/unit/test_render_smoke.py — renders one 240x320 frame
of a minimal single-drone world via mujoco.Renderer, asserts shape +
dtype.  Catches asset/mesh/camera regressions invisible to headless
episode loops.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 6: Port test_team_env_canary (deterministic 1v1 fingerprint)

**Goal:** Port `check_team_env.py` — the beeline-vs-beeline canary — to pytest with exact-value asserts at the locked fingerprints (step 176 tag entry, step 684 score, final totals). Marked `slow`.

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/test_team_env_canary.py`
- Delete: `scripts/check_team_env.py`
- Modify: `Makefile` (remove `team-check`)

- [ ] **Step 1: Create empty `tests/integration/__init__.py`**

```bash
mkdir -p tests/integration
: > tests/integration/__init__.py
```

- [ ] **Step 2: Write the test**

Create `tests/integration/test_team_env_canary.py`:

```python
"""Beeline-vs-beeline deterministic canary for QuidditchTeamEnv.

Ported from scripts/check_team_env.py.

Locked fingerprint (2026-05-06):
    step 176: TAG_ENTRY            blue +5.019  red −5.023
    step 684: TERMINATED (SCORE)   blue −10.005 red +9.999
    EPISODE END: red_total +1.931, blue_total −7.618
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = pytest.mark.slow


def _beeline_action(obs: np.ndarray) -> np.ndarray:
    """obs[12:15] is unit_to_goal — maps to (dx, dy, dyaw=0, dz)."""
    return np.clip(
        np.array([obs[12], obs[13], 0.0, obs[14]], dtype=np.float32),
        -1.0, 1.0,
    )


def test_beeline_red_vs_beeline_blue_canary() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        obs, _ = env.reset(seed=42)
        total = {"red_0": 0.0, "blue_0": 0.0}
        step = 0

        seen_tag_entry_step: int | None = None
        seen_tag_entry_rew: dict[str, float] = {}

        seen_terminate_step: int | None = None
        seen_terminate_rew: dict[str, float] = {}
        seen_terminate_info: dict | None = None

        while env.agents:
            actions = {
                "red_0":  _beeline_action(obs["red_0"]),
                "blue_0": _beeline_action(obs["blue_0"]),
            }
            obs, rew, term, _trunc, info = env.step(actions)
            step += 1
            total["red_0"]  += rew["red_0"]
            total["blue_0"] += rew["blue_0"]

            if seen_tag_entry_step is None and info["red_0"].get("tag_entry"):
                seen_tag_entry_step = step
                seen_tag_entry_rew = dict(rew)

            if any(term.values()):
                seen_terminate_step = step
                seen_terminate_rew = dict(rew)
                seen_terminate_info = info
                break

        assert seen_tag_entry_step == 176, (
            f"expected first tag_entry at step 176, got {seen_tag_entry_step}"
        )
        assert seen_tag_entry_rew["blue_0"] == pytest.approx(+5.019, abs=1e-3)
        assert seen_tag_entry_rew["red_0"]  == pytest.approx(-5.023, abs=1e-3)

        assert seen_terminate_step == 684, (
            f"expected SCORE termination at step 684, got {seen_terminate_step}"
        )
        assert seen_terminate_info is not None
        assert seen_terminate_info["red_0"]["scored"], (
            f"expected SCORE termination, got info={seen_terminate_info}"
        )
        assert seen_terminate_rew["blue_0"] == pytest.approx(-10.005, abs=1e-3)
        assert seen_terminate_rew["red_0"]  == pytest.approx(+9.999,  abs=1e-3)

        assert total["red_0"]  == pytest.approx(+1.931, abs=1e-3)
        assert total["blue_0"] == pytest.approx(-7.618, abs=1e-3)
    finally:
        env.close()
```

- [ ] **Step 3: Run the new test**

```bash
conda run -n uav python -m pytest tests/integration/test_team_env_canary.py -v
```

Expected: 1 passed in 5–15s.

- [ ] **Step 4: Sanity-check the original script**

```bash
conda run -n uav python scripts/check_team_env.py
```

Expected: prints the fingerprint lines from the docstring.

- [ ] **Step 5: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_team_env.py
```

- [ ] **Step 6: Remove `team-check` from the Makefile**

Drop `team-check` from `.PHONY`; delete its target/recipe block.

- [ ] **Step 7: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 9 passed.

- [ ] **Step 8: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/integration/__init__.py tests/integration/test_team_env_canary.py Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_team_env beeline canary to pytest with exact-value asserts

One-test port of scripts/check_team_env.py to
tests/integration/test_team_env_canary.py.  Asserts the locked
fingerprint (step 176 tag-entry, step 684 score, final totals) with
abs=1e-3 tolerance — accepts platform float jitter without losing the
signal.  Marked slow.  Source script and team-check Makefile target
removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 7: Port test_simple_env_contract + test_scoring_canary (and delete check_env.py)

**Goal:** `check_env.py` carries three things — SB3 contract check, zero-action smoke episode, scripted-fly-to-hoop canary. Two test files split the responsibilities. Source script and **both** Makefile targets that called it (`check-sim` and `check-gui`) are removed in this same task. (`check-gui` is "removed early" relative to the spec's narrative — its replacement, `demo/scenarios.py`, doesn't exist yet, so for one task `make check-gui` reports a missing target. That's OK; nobody else relies on it.)

**Files:**
- Create: `tests/integration/test_simple_env_contract.py`
- Create: `tests/integration/test_scoring_canary.py`
- Delete: `scripts/check_env.py`
- Modify: `Makefile` (remove `check-sim`, `check-gui`)

- [ ] **Step 1: Write `test_simple_env_contract.py`**

Create `tests/integration/test_simple_env_contract.py`:

```python
"""SB3 env-checker contract + zero-action smoke episode for QuidditchSimpleEnv.

Ported from parts 1 & 2 of scripts/check_env.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.quidditch.simple_env import QuidditchSimpleEnv

pytestmark = pytest.mark.slow


def test_sb3_check_env_passes() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        check_env(env, warn=True)
    finally:
        env.close()


def test_zero_action_episode_runs_10_steps() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        obs, _ = env.reset()
        assert obs.shape == (16,), f"unexpected obs shape {obs.shape}"
        assert obs.dtype == np.float32, f"unexpected obs dtype {obs.dtype}"

        for _ in range(10):
            action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, _info = env.step(action)
            assert np.isfinite(reward), f"non-finite reward: {reward}"
            if terminated or truncated:
                break
    finally:
        env.close()
```

- [ ] **Step 2: Write `test_scoring_canary.py`**

Create `tests/integration/test_scoring_canary.py`:

```python
"""Single-drone scripted scoring canary.

Two-phase script: climb to hoop altitude at the arena centre, then push
past the hoop along its outward normal.  Aiming directly at HOOP_CENTER
is unreliable under the MuJoCo controller — z setpoint scale (0.1) lags
xy scale (0.2), so the drone arrives at hoop plane still below the
aperture and goes around it.

Ported from part 3 of scripts/check_env.py.

Locked fingerprint (2026-05-06):
    SCORED at step 434 / total reward 7.3837
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
from envs.quidditch.simple_env import QuidditchSimpleEnv

pytestmark = pytest.mark.slow


def test_scripted_flyaway_scores_through_hoop() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        obs, _ = env.reset()

        approach_point = np.array(
            [0.0, 0.0, float(HOOP_CENTER[2])], dtype=np.float64
        )
        through_point = HOOP_CENTER + 0.7 * HOOP_OUTWARD_NORMAL  # 0.7 m past hoop
        phase2 = False

        scored_at_step: int | None = None
        total_reward = 0.0
        for step in range(env._max_steps):
            pos = obs[9:12].astype(np.float64)

            if not phase2 and np.linalg.norm(pos - approach_point) < 0.3:
                phase2 = True

            target = through_point if phase2 else approach_point
            vec = target - pos

            if np.linalg.norm(vec) < 0.01:
                action = np.zeros(4, dtype=np.float32)
            else:
                action = np.array([
                    np.clip(vec[0] / 0.2, -1.0, 1.0),
                    np.clip(vec[1] / 0.2, -1.0, 1.0),
                    0.0,
                    np.clip(vec[2] / 0.1, -1.0, 1.0),
                ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info.get("scored"):
                scored_at_step = step
                break

            if terminated or truncated:
                break

        assert scored_at_step == 434, (
            f"expected SCORED at step 434, got {scored_at_step}"
        )
        assert total_reward == pytest.approx(7.3837, abs=1e-4)
    finally:
        env.close()
```

- [ ] **Step 3: Run the new tests**

```bash
conda run -n uav python -m pytest \
  tests/integration/test_simple_env_contract.py \
  tests/integration/test_scoring_canary.py -v
```

Expected: 3 passed in 5–10s total.

- [ ] **Step 4: Sanity-check the original script**

```bash
conda run -n uav python scripts/check_env.py
```

Expected: prints `All checks complete.` after running parts 1–3.

- [ ] **Step 5: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_env.py
```

- [ ] **Step 6: Remove `check-sim` and `check-gui` from the Makefile**

In `Makefile`:

1. Drop `check-sim` and `check-gui` from the `.PHONY` line.
2. Delete the `check-sim` target/recipe pair (two lines of code + one trailing blank).
3. Delete the `check-gui` target/recipe pair (two lines + trailing blank).

- [ ] **Step 7: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed.

- [ ] **Step 8: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/integration/test_simple_env_contract.py \
      tests/integration/test_scoring_canary.py \
      Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_env to pytest contract + scoring canary

Splits scripts/check_env.py into tests/integration/test_simple_env_contract.py
(SB3 check_env + 10-step zero-action smoke) and
tests/integration/test_scoring_canary.py (exact-value canary at step 434,
reward 7.3837).  Both marked slow.  Source script and the check-sim and
check-gui Makefile targets removed; check-gui is replaced by demo/scenarios
(landing in a later commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 8: Port test_warm_start (skip-if-no-OLD_MODEL)

**Goal:** Port `check_team_warm.py` with the skip-if-no-artifact pattern. Test runs only when `OLD_MODEL` env var is set; otherwise pytest reports it as skipped with a clear reason.

**Files:**
- Create: `tests/integration/test_warm_start.py`
- Delete: `scripts/check_team_warm.py`
- Modify: `Makefile` (remove `team-check-warm`)

- [ ] **Step 1: Write the test**

Create `tests/integration/test_warm_start.py`:

```python
"""Warm-start preserves single-agent behavior on obs[:16] when obs[16:] is
zeroed out.

Ported from scripts/check_team_warm.py.  Requires a trained single-agent
checkpoint at the path given by OLD_MODEL=<path/to/best_model>; skipped
otherwise.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.policies.warm_start import warm_start_ppo
from envs.quidditch.opponents import BeelineBlue, OpponentControlledEnv
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        "OLD_MODEL" not in os.environ,
        reason="set OLD_MODEL=models/<run>/best_model to run warm-start regression",
    ),
]


def test_warm_started_policy_matches_old_on_obs_prefix() -> None:
    old_path = os.environ["OLD_MODEL"]
    n_samples = 64
    tol = 0.1

    old = PPO.load(old_path)

    def _env_fn():
        return OpponentControlledEnv(
            QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False)),
            learner_id="red_0",
            opponent=BeelineBlue(),
        )

    new_env = DummyVecEnv([_env_fn])
    try:
        new = warm_start_ppo(
            old_checkpoint=old_path,
            new_env=new_env,
            new_input_dim=22,
            old_input_dim=16,
            new_dim_init_scale=0.01,
        )

        rng = np.random.default_rng(42)
        obs_seed = new_env.reset()
        diffs: list[float] = []
        for _ in range(n_samples):
            obs_zeroed = obs_seed.copy()
            obs_zeroed[..., 16:] = 0.0
            action,     _ = new.predict(obs_zeroed,         deterministic=True)
            old_action, _ = old.predict(obs_seed[..., :16], deterministic=True)
            diffs.append(float(np.linalg.norm(action - old_action)))
            random_a = rng.uniform(-1, 1, size=action.shape).astype(np.float32)
            obs_seed, _, _, _ = new_env.step(random_a)

        mean_diff = float(np.mean(diffs))
        assert mean_diff < tol, (
            f"warm-started policy drifts {mean_diff:.4f} > tol {tol}"
        )
    finally:
        new_env.close()
```

- [ ] **Step 2: Confirm the test SKIPS without OLD_MODEL**

```bash
conda run -n uav python -m pytest tests/integration/test_warm_start.py -v
```

Expected: 1 skipped, with reason `set OLD_MODEL=...`. The `-ra` config flag prints the skip reason in the summary.

- [ ] **Step 3: If a trained checkpoint is available, run the test for real**

This is optional — only do it if `models/ppo_hoop_fixed_start_20260504_023051/best_model.zip` (per `brain/index.md`) exists locally:

```bash
OLD_MODEL=models/ppo_hoop_fixed_start_20260504_023051/best_model \
  conda run -n uav python -m pytest tests/integration/test_warm_start.py -v
```

Expected: 1 passed, prints a final `mean ||a_new − a_old||` statistic well under 0.1.

- [ ] **Step 4: Delete the source script**

```bash
git -C "$PWD" rm scripts/check_team_warm.py
```

- [ ] **Step 5: Remove `team-check-warm` from the Makefile**

Drop `team-check-warm` from `.PHONY`; delete its multi-line target block (the one with the `OLD=` precondition).

- [ ] **Step 6: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped (warm-start).

- [ ] **Step 7: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add tests/integration/test_warm_start.py Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
test: port check_team_warm to pytest with skip-if-no-artifact gate

Port of scripts/check_team_warm.py to
tests/integration/test_warm_start.py — module-level skipif on missing
OLD_MODEL env var so default pytest runs ignore it cleanly.  Source
script and team-check-warm Makefile target removed; replacement
test-warm Makefile target lands with the rest of the new targets in a
later task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation. **All seven `check_*.py` scripts are now gone.** Their Makefile targets are too.

---

## Task 9: Scaffold demo/scenarios.py with terminal reward log + main loop

**Goal:** Land the file structure of `demo/scenarios.py` with the reward-event logger and the two-scenarios `main()` skeleton — but *without* tuned scripted policies yet. The two scenarios will use placeholder policies (zero-action) that exercise the env loop without trying to hit the beats. Beat-tuning happens in Tasks 10 and 11.

This separation lets us verify the env wiring, viewer integration, and reward log work before we start chasing the narratives.

**Files:**
- Create: `demo/scenarios.py`

- [ ] **Step 1: Write the scaffold**

Create `demo/scenarios.py`:

```python
"""Scripted 1v1 narratives for visual review of tag/crash/score behavior.

Two scenarios run back-to-back in a single MuJoCo viewer:
  A) Tagged out — Blue tags Red twice, then rams it down.
  B) Through despite the tag — Red gets tagged once but still scores.

Run via:  make demo  → pick "scenarios"

NOTE: As of this commit, the scripted policies are placeholder zero-actions.
The narrative beats are tuned in subsequent commits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Allow imports from project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

# Toggle to draw colored event markers in the 3D scene (deferred — the
# helper is a no-op until SHOW_EVENT_MARKERS=True).
SHOW_EVENT_MARKERS = False

# Type alias for a single-side policy: takes that side's obs, returns its action.
Policy = Callable[[np.ndarray], np.ndarray]


def _placeholder_policy(_obs: np.ndarray) -> np.ndarray:
    """Zero-action placeholder until per-scenario policies are tuned."""
    return np.zeros(4, dtype=np.float32)


def _log_event(t: float, label: str, rew: dict[str, float],
               totals: dict[str, float]) -> None:
    """One-line terminal log for an event."""
    print(
        f"[t={t:5.2f}s] {label:<22} "
        f"red {rew['red_0']:+.3f}  blue {rew['blue_0']:+.3f}   "
        f"totals: red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _detect_label(info: dict, rew: dict[str, float]) -> str | None:
    """Return a human-readable event label for this step, or None."""
    if info["red_0"].get("scored"):
        return "SCORE"
    if info["red_0"].get("drone_drone_crash"):
        return "DRONE-DRONE CRASH"
    if info["red_0"].get("red_floor"):
        return "RED FLOOR"
    if info["red_0"].get("red_wall_crash"):
        return "RED WALL CRASH"
    if info["red_0"].get("red_oob"):
        return "RED OOB"
    if info["blue_0"].get("blue_floor"):
        return "BLUE FLOOR"
    if info["blue_0"].get("blue_wall_crash"):
        return "BLUE WALL CRASH"
    if info["blue_0"].get("blue_oob"):
        return "BLUE OOB"
    if info["red_0"].get("tag_entry"):
        return "TAG ENTRY"
    return None


def _draw_event_marker(env, kind: str) -> None:
    """No-op until SHOW_EVENT_MARKERS=True (deferred)."""
    if not SHOW_EVENT_MARKERS:
        return
    # Future: append an mjvGeom into env._world.viewer.user_scn.geoms
    # — colored sphere at the relevant body position. Kept as a stub so
    # flipping the flag elsewhere doesn't need new plumbing.
    return


def _run_scenario(name: str, env: QuidditchTeamEnv,
                   red_policy: Policy, blue_policy: Policy,
                   max_seconds: float) -> None:
    """Run one scripted scenario from current env state until termination
    or timeout; log events to terminal."""
    print(f"\n[Scenario {name}]")
    obs, _ = env.reset(seed=42)
    totals = {"red_0": 0.0, "blue_0": 0.0}
    dt_step = 1.0 / 240.0  # 240 Hz physics; adjust if env reports differently
    max_steps = int(max_seconds / dt_step)

    for step in range(max_steps):
        actions = {
            "red_0":  red_policy(obs["red_0"]),
            "blue_0": blue_policy(obs["blue_0"]),
        }
        obs, rew, term, trunc, info = env.step(actions)
        totals["red_0"]  += rew["red_0"]
        totals["blue_0"] += rew["blue_0"]

        label = _detect_label(info, rew)
        if label is not None:
            _log_event(step * dt_step, label, rew, totals)
            _draw_event_marker(env, label)

        if any(term.values()) or any(trunc.values()):
            break

    print(
        f"[Scenario {name}] FINAL  "
        f"red {totals['red_0']:+.2f}  blue {totals['blue_0']:+.2f}"
    )


def _idle_pause(env: QuidditchTeamEnv, seconds: float) -> None:
    """Step the env with zero actions for `seconds`; lets the viewer breathe
    between scenarios so the user can see the final state."""
    dt_step = 1.0 / 240.0
    for _ in range(int(seconds / dt_step)):
        env.step({
            "red_0":  np.zeros(4, dtype=np.float32),
            "blue_0": np.zeros(4, dtype=np.float32),
        })


def main() -> None:
    env = QuidditchTeamEnv(cfg=TeamConfig(
        randomise_red_start=False, episode_seconds=30.0,
    ))
    try:
        # Scenario A — Tagged out (placeholder policies for now).
        _run_scenario(
            "A: Tagged out",
            env,
            red_policy=_placeholder_policy,
            blue_policy=_placeholder_policy,
            max_seconds=25.0,
        )
        _idle_pause(env, seconds=1.5)

        # Scenario B — Through despite the tag (placeholder policies for now).
        _run_scenario(
            "B: Through despite the tag",
            env,
            red_policy=_placeholder_policy,
            blue_policy=_placeholder_policy,
            max_seconds=10.0,
        )

        # Hold the viewer open for inspection.
        print("\nDemo complete — viewer remains open. Ctrl-C to quit.")
        while True:
            _idle_pause(env, seconds=1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run scenarios with placeholder policies (headless first)**

The team env may or may not auto-launch a viewer depending on its render-mode plumbing. First, run headless to confirm the scaffolding doesn't crash:

```bash
conda run -n uav python demo/scenarios.py
```

Expected: prints both `[Scenario A]` and `[Scenario B]` headers, eventually hits some termination event (likely RED FLOOR since the placeholder policies don't fly), prints `Demo complete — viewer remains open.`. **Ctrl-C to exit the idle loop.**

If you see import errors, re-check the `sys.path` insert in the file — it has to come before the `from envs.quidditch...` lines.

- [ ] **Step 3: Run the menu under mjpython to confirm viewer integration (manual)**

This will be done in Task 12 once the menu entry is wired. Skip for now.

- [ ] **Step 4: Run all tests (sanity check that nothing in tests/ is affected)**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped. demo/scenarios.py isn't picked up because `pyproject.toml` scopes `testpaths = ["tests"]`.

- [ ] **Step 5: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add demo/scenarios.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
demo: scaffold scenarios.py with terminal reward logger

New demo module with the two-scenario main() skeleton, terminal
event-log helpers, and a deferred event-marker hook (no-op until
SHOW_EVENT_MARKERS=True).  Scripted policies are placeholders for
now — narrative beat tuning happens in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 10: Tune scenario A — "Tagged out"

**Goal:** Replace the placeholder policies in Scenario A with scripts that hit the three beats reliably under `seed=42`: TAG ENTRY ×2, then DRONE-DRONE CRASH terminating the episode.

**This task is iterative.** The skeleton is fixed; the *parameters* (timing thresholds, target points, ramming speed) get tuned via run-and-watch loops. Budget ~1 hour.

**Files:**
- Modify: `demo/scenarios.py`

- [ ] **Step 1: Confirm the team env obs layout**

The obs layout is documented at `envs/quidditch/team_env.py:10-22`:
```
[0:3]   angular velocity  — body frame, rad/s
[3:6]   attitude euler    — ground frame, rad
[6:9]   linear velocity   — body frame, m/s
[9:12]  position          — ground frame, m
[12:15] unit vector to goal target (hoop center for Red; midpoint for Blue)
[15]    signed distance to hoop plane / ARENA_RADIUS
[16:19] opp_pos - self_pos (world frame)        ← used by Blue to chase Red
[19:22] opp_vel - self_vel (body frame)
```

Confirm those line numbers with:

```bash
sed -n '10,22p' envs/quidditch/team_env.py
```

The two columns we'll lean on: **`obs[9:12]`** is self position (world frame) and **`obs[16:19]`** is the relative vector from self to the opponent. Distance to opponent is `np.linalg.norm(obs[16:19])`.

- [ ] **Step 2: Add the shared `_delta_action` helper + Red's two-phase scoring policy**

Edit `demo/scenarios.py`. Add imports near the top alongside the existing ones:

```python
from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
```

Then add these helpers below `_placeholder_policy` (the placeholder stays for Scenario B's still-placeholder invocation — it'll be deleted in Task 11 once both scenarios are wired):

```python
# === Shared helpers ====================================================

_HOOP_CENTER    = np.asarray(HOOP_CENTER, dtype=np.float64)
_HOOP_NORMAL    = np.asarray(HOOP_OUTWARD_NORMAL, dtype=np.float64)
_APPROACH_POINT = np.array([0.0, 0.0, _HOOP_CENTER[2]], dtype=np.float64)
_THROUGH_POINT  = _HOOP_CENTER + 0.7 * _HOOP_NORMAL

# Action-scale constants matching simple_env / team_env: dx, dy at ±0.2,
# dz at ±0.1 per step. Scaling a target-relative vector by these gains and
# clipping yields a saturated delta-setpoint action.
_DX_SCALE = 0.2
_DZ_SCALE = 0.1


def _delta_action(vec: np.ndarray) -> np.ndarray:
    """Convert a world-frame target-relative vector into a delta-setpoint action."""
    if float(np.linalg.norm(vec)) < 0.01:
        return np.zeros(4, dtype=np.float32)
    return np.array([
        np.clip(vec[0] / _DX_SCALE, -1.0, 1.0),
        np.clip(vec[1] / _DX_SCALE, -1.0, 1.0),
        0.0,
        np.clip(vec[2] / _DZ_SCALE, -1.0, 1.0),
    ], dtype=np.float32)


# === Scenario A: Tagged out ============================================


def _scoring_red_factory():
    """Two-phase Red: climb to hoop altitude at origin, then push past hoop
    along its outward normal. Same pattern as test_scoring_canary so Red can
    actually score when not interfered with."""
    state = {"phase2": False}

    def policy(obs: np.ndarray) -> np.ndarray:
        pos = obs[9:12].astype(np.float64)
        if (not state["phase2"]) and np.linalg.norm(pos - _APPROACH_POINT) < 0.3:
            state["phase2"] = True
        target = _THROUGH_POINT if state["phase2"] else _APPROACH_POINT
        return _delta_action(target - pos)

    return policy


def _scenario_a_blue_factory():
    """Stateful Blue for Scenario A:
      intercept_1 → in_sphere_1 → pullback → intercept_2 → ram

    Distances tuned against the env's tag-sphere radius (~0.2 m) and
    cooldown geometry. INSIDE = 0.18 m; OUTSIDE = 0.55 m so cooldown fully
    elapses between tags. RAM_GAIN scales the to-Red vector before
    `_delta_action` clipping, ensuring saturation throughout the ram so
    the contact velocity exceeds CRASH_VEL_THR.
    """
    INSIDE_THR  = 0.18
    OUTSIDE_THR = 0.55
    RAM_GAIN    = 5.0

    state = {"phase": "intercept_1"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept_1":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere_1"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere_1":
            # Hold on Red — env will fire tag_entry/tag_during. Once Red
            # moves on its beeline and we drift past the boundary, transition.
            if dist > INSIDE_THR + 0.05:
                state["phase"] = "pullback"
            return _delta_action(to_red)

        if state["phase"] == "pullback":
            if dist > OUTSIDE_THR:
                state["phase"] = "intercept_2"
            return _delta_action(-to_red)

        if state["phase"] == "intercept_2":
            if dist < INSIDE_THR:
                state["phase"] = "ram"
            return _delta_action(to_red)

        if state["phase"] == "ram":
            return _delta_action(to_red * RAM_GAIN)

        return np.zeros(4, dtype=np.float32)

    return policy
```

In `main()`, replace the Scenario A invocation:

```python
        _run_scenario(
            "A: Tagged out",
            env,
            red_policy=_scoring_red_factory(),
            blue_policy=_scenario_a_blue_factory(),
            max_seconds=25.0,
        )
```

- [ ] **Step 3: Run scenario A in isolation (headless)**

To skip Scenario B during tuning, temporarily comment out its `_run_scenario` call in `main()`. Then:

```bash
conda run -n uav python demo/scenarios.py
```

Expected: terminal log shows two TAG ENTRY events followed by DRONE-DRONE CRASH. If something different fires (e.g. RED FLOOR before Blue gets there), tune.

- [ ] **Step 4: Tuning loop — adjust tunables, re-run, repeat**

Watch the printed event log. The desired pattern:

```
[t= ~3-5s] TAG ENTRY        red −5.000  blue +5.000
[t= ~7-9s] TAG ENTRY        red −5.000  blue +5.000
[t=~10-12s] DRONE-DRONE CRASH
```

Failure modes and fixes:
- **No TAG ENTRY at all:** Blue isn't catching Red. The starting positions Red and Blue are reset to depend on `TeamConfig`. If Blue starts farther from Red's beeline path than Blue can cover, increase Blue's effective speed by scaling the `_delta_action(to_red)` input (e.g. `to_red * 1.5`). Or check that `obs[16:19]` is non-zero — distance should be a few meters at episode start.
- **Only one TAG ENTRY:** `pullback` doesn't reach `OUTSIDE_THR` before Red moves past, OR cooldown longer than expected. Bump `OUTSIDE_THR` from 0.55 to 0.7, or have Blue actively retreat farther by scaling `_delta_action(-to_red * 2.0)` in the pullback phase.
- **CRASH before second TAG:** Blue's intercept-2 lands too fast, triggering crash threshold before tag pulse fires. Lower the `RAM_GAIN` *into* intercept_2 — i.e., during `intercept_2` use `_delta_action(to_red)` not `_delta_action(to_red * RAM_GAIN)`. (As written, intercept_2 already uses unscaled to_red — this should be fine, but worth verifying.)
- **No CRASH:** Velocity at contact too low. Increase `RAM_GAIN` (5.0 → 10.0). Or extend the run with `max_seconds=30.0` to allow longer ram approach.
- **RED FLOOR before any beats:** Red can't fly. Run Red's policy in isolation — temporarily set `blue_policy=_placeholder_policy` and confirm Red scores or at least flies up. If Red plummets, double-check `_HOOP_CENTER[2]` is non-zero (it should be 2.0 m).

Iterate until both TAG ENTRYs and the CRASH land within the 25-second window. Each tuning cycle is one edit + one run; budget 30–60 minutes total.

- [ ] **Step 5: Restore Scenario B's call**

Un-comment the Scenario B `_run_scenario` invocation in `main()` so the file is back to running both scenarios.

- [ ] **Step 6: Confirm headless run completes both scenarios**

```bash
conda run -n uav python demo/scenarios.py
```

Expected: Scenario A logs two TAG ENTRYs and a CRASH; Scenario B (still placeholder, that's fine) logs whatever its placeholder policies cause. Idle loop kicks in. Ctrl-C to exit.

- [ ] **Step 7: Run all tests (still passing)**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped.

- [ ] **Step 8: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add demo/scenarios.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
demo: tune scenario A scripted policies (Tagged out)

Scenario A scripted policies for Red (beeline to hoop) and Blue
(intercept → tag → pullback → re-intercept → tag → ram → crash) tuned
under seed=42 to land both TAG ENTRY events and a DRONE-DRONE CRASH
within ~12s of sim time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 11: Tune scenario B — "Through despite the tag"

**Goal:** Replace placeholder policies in Scenario B. Beats: TAG ENTRY ×1, then SCORE.

**Files:**
- Modify: `demo/scenarios.py`

- [ ] **Step 1: Add Scenario B helpers below Scenario A's**

In `demo/scenarios.py`, add after the Scenario A section:

```python
# === Scenario B: Through despite the tag ===============================


def _scenario_b_blue_factory():
    """One-tag-then-peel-off:
      intercept → in_sphere → peel_off

    Blue closes once, lets the tag pulse fire, then retreats along
    `PEEL_OFF_DIR` so Red is free to continue toward the hoop.
    """
    INSIDE_THR  = 0.18
    PEEL_OFF_DIR = np.array([0.0, 1.5, 0.5], dtype=np.float64)  # +y mostly, slight up

    state = {"phase": "intercept"}

    def policy(obs: np.ndarray) -> np.ndarray:
        to_red = obs[16:19].astype(np.float64)
        dist   = float(np.linalg.norm(to_red))

        if state["phase"] == "intercept":
            if dist < INSIDE_THR:
                state["phase"] = "in_sphere"
            return _delta_action(to_red)

        if state["phase"] == "in_sphere":
            # Hold for one step inside the sphere — tag_entry fires —
            # then retreat.
            state["phase"] = "peel_off"
            return _delta_action(to_red)

        if state["phase"] == "peel_off":
            return _delta_action(PEEL_OFF_DIR)

        return np.zeros(4, dtype=np.float32)

    return policy
```

In `main()`, replace the Scenario B `_run_scenario` invocation:

```python
        _run_scenario(
            "B: Through despite the tag",
            env,
            red_policy=_scoring_red_factory(),
            blue_policy=_scenario_b_blue_factory(),
            max_seconds=15.0,
        )
```

(`max_seconds=15.0` not 10.0 — Red needs the full hoop approach time after Blue gets out of the way.)

- [ ] **Step 2: Run headless and watch for the SCORE**

```bash
conda run -n uav python demo/scenarios.py
```

Expected: After Scenario A's events, Scenario B prints:

```
[Scenario B: Through despite the tag]
[t= ~2-4s] TAG ENTRY        red −5.000  blue +5.000
[t= ~6-8s] SCORE            red +10.000 blue −10.000
```

Failure modes:
- **No SCORE:** Red can't reach hoop. Did the previous Scenario A's reset state leave Red somewhere unflyable? Confirm `env.reset(seed=42)` is being called at the top of `_run_scenario` (it is — sanity-check).
- **DRONE-DRONE CRASH instead of SCORE:** Blue didn't peel off in time. Loosen the `dist_to_red < 0.18` threshold OR increase `PEEL_OFF_VEC` magnitude.
- **No TAG ENTRY:** Same fix as Scenario A — verify `red_pos` comes from the right obs columns.

- [ ] **Step 3: Tune until both beats land**

Iterate as in Task 10 Step 4.

- [ ] **Step 4: Remove the now-unused `_placeholder_policy`**

Both scenarios are wired up to real policies. Delete the `_placeholder_policy` function from `demo/scenarios.py` (it has no remaining callers).

- [ ] **Step 5: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped.

- [ ] **Step 6: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add demo/scenarios.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
demo: tune scenario B scripted policies (Through despite the tag)

Scenario B Red reuses the two-phase scoring policy; Blue intercepts
once, lets the tag pulse fire, then peels off along PEEL_OFF_DIR so
Red is free to continue and score.  Tuned under seed=42 to land
TAG ENTRY then SCORE within ~12s of sim time.  Removes the
_placeholder_policy stub that's now unreferenced.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 12: Wire scenarios into demo/menu.py + smoke test under mjpython

**Goal:** Add the new entry to the demo menu and confirm `make demo → scenarios` opens the MuJoCo viewer correctly under `mjpython`.

**Files:**
- Modify: `demo/menu.py`

- [ ] **Step 1: Add the menu entry**

Edit `demo/menu.py`. The `DEMOS` list currently looks like:

```python
DEMOS: list[tuple[str, str, str]] = [
    ("hover",    "Hover at 1 m in the Quidditch arena (hoop + wall)", "demo.hover_demo"),
    ("waypoint", "Fly a triangular waypoint loop in empty space",      "demo.waypoint_demo"),
]
```

Add one tuple to the end:

```python
DEMOS: list[tuple[str, str, str]] = [
    ("hover",    "Hover at 1 m in the Quidditch arena (hoop + wall)", "demo.hover_demo"),
    ("waypoint", "Fly a triangular waypoint loop in empty space",      "demo.waypoint_demo"),
    ("scenarios", "Scripted 1v1: tag-out + score-despite-tag",         "demo.scenarios"),
]
```

- [ ] **Step 2: Run the demo menu (manual — viewer opens)**

```bash
conda run -n uav mjpython demo/menu.py
```

When prompted, type `scenarios` (or its number) and press Enter. Expected behavior:
1. The MuJoCo viewer window opens.
2. Scenario A plays — you see Red and Blue moving; the terminal shows two TAG ENTRY events and a DRONE-DRONE CRASH.
3. After a 1.5s pause, Scenario B plays — terminal shows TAG ENTRY then SCORE.
4. The viewer holds with the final pose; terminal prints "Demo complete — viewer remains open. Ctrl-C to quit."

Ctrl-C in the terminal to exit.

If the viewer doesn't open: the team env's `__init__` may not be opening one. Search `envs/quidditch/team_env.py` for `viewer` references — you may need to pass a `render_mode="human"` or equivalent kwarg in `_run_scenario`'s env construction. The single-agent env uses `render_mode="human"` for this — check whether `TeamConfig` has a similar field.

- [ ] **Step 3: Confirm `make help` shows the menu still works**

```bash
make help
make demo  # answer 'q' at the prompt to exit cleanly
```

Expected: `make demo` lists the three options including "scenarios".

- [ ] **Step 4: Run all tests**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped.

- [ ] **Step 5: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add demo/menu.py

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
demo: register scenarios in the demo menu

Adds a "scenarios" entry to demo/menu.py's DEMOS list so the scripted
1v1 narratives are pickable from `make demo`.  Replaces the role
previously played by `make check-gui` (removed two commits ago).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 13: Add new Makefile targets (test, test-fast, test-warm)

**Goal:** Provide the new test entry points to replace what the old `check-*` and `team-check-*` targets did.

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the three new targets**

Add these three target blocks to the Makefile in the section after the help target (near where `check-sim` used to live):

```make
test: ## ✅ Run all tests (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit

test-warm: ## ✅ Warm-start preserves single-agent behavior  OLD=models/<run>
	@test -n "$(OLD)" || { echo "ERROR: OLD=models/<run> required"; exit 1; }; \
	 OLD_MODEL="$(OLD)/best_model" $(PYTHON) -m pytest tests/integration/test_warm_start.py
```

- [ ] **Step 2: Add the new targets to `.PHONY`**

Add `test test-fast test-warm` to the `.PHONY` line near the top of the Makefile.

- [ ] **Step 3: Verify the help target lists them**

```bash
make help
```

Expected: `test`, `test-fast`, and `test-warm` show up alongside the other documented targets, with their `## ...` descriptions.

- [ ] **Step 4: Verify each target works**

```bash
make test
```

Expected: 12 passed, 1 skipped.

```bash
make test-fast
```

Expected: **6 passed** (1 mjcf + 3 crash + 1 tag + 1 render = 6 unit tests). `test_imports.py` lives at `tests/` root, not `tests/unit/`, so the path filter excludes it.

```bash
make test-warm OLD=models/<some-run>  # only if you have a model
```

Expected: 1 passed (if model exists) or fails the `test -n` precondition.

If the canonical model from `brain/index.md` is around:

```bash
make test-warm OLD=models/ppo_hoop_fixed_start_20260504_023051
```

- [ ] **Step 5: User runs commit (manual)**

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  add Makefile

git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/pytest-migration" \
  commit -m "$(cat <<'EOF'
make: add test, test-fast, test-warm targets

Three new Makefile targets replace the seven check-* targets removed
across the migration: test (all), test-fast (unit only), test-warm
(opt-in warm-start regression with OLD=<run> contract preserved from
the old team-check-warm).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for confirmation.

---

## Task 14: Update brain/ files

**Goal:** Capture the migration in the brain/ persistent memory per the project's session-end protocol.

**Files (outside the worktree — `brain/` lives at the umbrella project level):**
- Modify: `../../brain/index.md` (Recent Context + Active Priorities)
- Modify: `../../brain/changelog.md` (new dated entry)
- Modify: `../../brain/decisions.md` (ADR for test architecture)
- Modify: `../../brain/tasks.md` (close any tracked items; list new follow-ups if any)

(`brain/` is shared across the worktree set; not committed to git. Edits are direct file writes, not commits.)

- [ ] **Step 1: Append a changelog entry to `brain/changelog.md`**

At the top of the changelog (or wherever the most-recent-first ordering puts new entries), add:

```markdown
## 2026-05-06 — pytest migration + scripted 1v1 demos

Replaced seven hand-rolled `scripts/check_*.py` scripts with a `tests/`
suite under pytest:
- `tests/unit/`: `test_team_mjcf`, `test_crash_detector`, `test_tag_state_machine`, `test_render_smoke` (new).
- `tests/integration/`: `test_simple_env_contract`, `test_scoring_canary`, `test_team_env_canary`, `test_warm_start` (skip-if-no-`OLD_MODEL`).
- All integration tests carry `@pytest.mark.slow`.
- Shared imperative helpers in `tests/conftest.py` (`build_team_world`, `qpos_addr`, `set_body_state`).
- Canary fingerprints converted to exact-value asserts: scoring canary at `step=434, reward=7.3837 (abs=1e-4)`; team-env canary at `step 176/684` with totals `+1.931 / -7.618 (abs=1e-3)`.

`make check-sim`, `make check-gui`, and the five `make team-check*`
targets are gone; replaced by `make test`, `make test-fast`,
`make test-warm OLD=<run>`. Old `OLD=` contract preserved on the warm
target for muscle-memory.

`make check-gui` (visual scoring run) replaced by a new
`demo/scenarios.py` entry under `make demo`: two scripted 1v1
narratives playing back-to-back in one viewer — Scenario A "Tagged
out" (Blue tags Red ×2 then crash-rams), Scenario B "Through despite
the tag" (Red tagged once then scores). Reward events logged to
terminal; deferred 3D-marker hook (`SHOW_EVENT_MARKERS` flag) stubs
out as a no-op until later.

New file: `pyproject.toml` with `[tool.pytest.ini_options]` only
(`pythonpath`, `testpaths`, `markers`) — replaces per-script
`sys.path.insert` shims.

Files touched (high level):
- Created: `tests/` tree, `pyproject.toml`, `demo/scenarios.py`,
  `docs/superpowers/specs/2026-05-06-pytest-migration-design.md`,
  `docs/superpowers/plans/2026-05-06-pytest-migration.md`.
- Modified: `Makefile`, `requirements.txt` (+pytest), `demo/menu.py`.
- Deleted: all seven `scripts/check_*.py` files.
```

- [ ] **Step 2: Update `brain/index.md`**

In the **Recent Context** section, prepend a new dated paragraph:

```markdown
**2026-05-06 (pytest migration).** Seven `scripts/check_*.py` scripts
replaced with a `tests/` pytest suite (unit + integration; `slow`
marker for canaries). `make check-sim` / `check-gui` / the five
`team-check*` targets gone; replaced by `make test` / `test-fast` /
`test-warm OLD=<run>`. New `demo/scenarios.py` (under `make demo`)
plays two scripted 1v1 narratives back-to-back: Scenario A (tags ×2
+ crash) and Scenario B (one tag + score). `pyproject.toml` carries
pytest config only. Spec + plan committed at
`docs/superpowers/specs/2026-05-06-pytest-migration-design.md` and
`docs/superpowers/plans/2026-05-06-pytest-migration.md`.
```

If there's an "Active Priorities" or "Known Issues" entry mentioning the old check scripts, update it accordingly.

- [ ] **Step 3: Add a decisions entry to `brain/decisions.md`**

Append a new ADR (the file already follows ADR-style; copy the prevailing format). Suggested content:

```markdown
## 2026-05-06 — Test architecture: pytest with unit/integration split

**Context.** The seven `scripts/check_*.py` scripts mixed contract
asserts, deterministic regression canaries, MJCF unit checks, and a
visual viewer mode. Each had a Makefile target and per-script
`sys.path` shim. Adding new asserts meant adding a Makefile target
and copying boilerplate.

**Decision.** Move all asserted checks into `tests/` driven by pytest.
Two subdirs: `tests/unit/` (millisecond asserts, no episode loops) and
`tests/integration/` (real env steps, marked `slow`). Shared
imperative helpers in `tests/conftest.py` (not fixtures — tests call
them directly). `pyproject.toml` carries `pythonpath = ["."]` so
imports work without per-file shims.

The visual check (`make check-gui`) is **not** in the test suite —
it's a developer-facing demo under `make demo` (`demo/scenarios.py`).
Two scripted 1v1 narratives play back-to-back in one viewer; rewards
logged to terminal. A `SHOW_EVENT_MARKERS` flag stubs out 3D event
markers for later.

**Why pytest over keeping `check_*.py`:** asserted by default, fail-loud
in CI, marker-based filtering for fast iteration, no per-script
boilerplate.

**Why `tests/` split into unit+integration directories:** the split
also corresponds to the slow-test split, so the directory itself
communicates "this might be slow." `make test-fast` filters by path
(`tests/unit/`) rather than by marker.

**Why exact-value canary asserts (vs tolerance):** existing fingerprints
are stable to 1e-3 across runs and platforms; using `pytest.approx(..., abs=1e-3)`
catches any drift larger than that, which would indicate a real physics change.

**Why scripted scenarios (vs trained-policy demos):** trained 1v1 models
don't exist yet (Phase 2 hasn't shipped). Scripted demos give us the
visual coverage now; trained-policy demos slot in later as additional
`demo/` entries.

**Why one big tag-FSM test (vs 6):** phases share state; splitting them
would force re-running phase 1–5 setup before phase 6, multiplying
runtime for no clarity gain.

**Out of scope for this migration:** CI workflow (separate work);
image/pixel regression of rendered frames (smoke test asserts
shape+dtype only); Option-B 3D event markers (flag in place, not
implemented yet).
```

- [ ] **Step 4: Update `brain/tasks.md`**

If `tasks.md` had any entries mentioning the old check scripts ("clean up check_team_*", "consolidate checks", etc.), mark them done. Add a follow-up entry:

```markdown
- [ ] Consider implementing 3D event markers in `demo/scenarios.py` (`SHOW_EVENT_MARKERS=True`). One-tick `mjvGeom` per event: red sphere at tag, gold at score, large red flash at crash. Deferred from 2026-05-06 pytest migration.
- [ ] Once trained 1v1 models exist, add additional `demo/` entries that load and play those policies (vs the current scripted ones).
- [ ] Consider adding a CI workflow that runs `make test-fast` on PR + `make test` on merge.
```

- [ ] **Step 5: Verify brain/ edits look right**

Just open each file and skim the section you modified — `brain/` isn't in git so there's no diff command (unless the user has a separate brain repo).

```bash
ls -la "../../brain/"
```

Expected: shows the four files were touched recently.

- [ ] **Step 6: No commit needed for brain/**

`brain/` lives outside the git repo. No commit step here.

---

## Task 15: Final verification

**Goal:** Confirm everything works end-to-end and the worktree is ready for merge into `develop`.

- [ ] **Step 1: Run the full test suite**

```bash
conda run -n uav python -m pytest -v
```

Expected: 12 passed, 1 skipped (warm-start, no OLD_MODEL).

- [ ] **Step 2: Run unit tests only**

```bash
make test-fast
```

Expected: 6 passed, sub-second.

- [ ] **Step 3: Run scenarios under viewer (manual)**

```bash
make demo
# pick: scenarios
```

Expected: viewer opens, both scenarios play with their tuned beats, terminal logs events. Ctrl-C to exit.

- [ ] **Step 4: Confirm git state is clean**

```bash
git -C "$PWD" status
git -C "$PWD" log --oneline develop..HEAD
```

Expected: working tree clean. Log shows ~13 commits (Tasks 1–13) on `feature/pytest-migration` ahead of `develop`.

- [ ] **Step 5: Confirm no dead Makefile targets reference deleted scripts**

```bash
grep -nE "scripts/check_" Makefile
```

Expected: no output. (If there's any match, a Makefile cleanup commit was missed.)

- [ ] **Step 6: Hand off to user for merge into develop**

The user (per umbrella GitFlow) merges with `--no-ff`:

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/repo" checkout develop
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/repo" \
  merge --no-ff feature/pytest-migration -m "$(cat <<'EOF'
Merge branch 'feature/pytest-migration' into develop

Replaces scripts/check_*.py with a pytest suite under tests/ (unit +
integration; slow-marked canaries) and replaces make check-gui with
demo/scenarios.py — two scripted 1v1 narratives under make demo.
Adds pyproject.toml with pytest config, removes the seven check-*
Makefile targets, and adds make test / test-fast / test-warm.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

After merge confirms, clean up:

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/repo" \
  worktree remove worktrees/feature/pytest-migration
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/repo" \
  branch -d feature/pytest-migration
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/repo" \
  checkout main
```

Per the umbrella convention, `repo/` returns to `main` after develop work lands. (The user's repo/ was on `develop` at the start of this work — they may want to leave it there or return to `main` per convention.)

---

## Summary

| Task | Files touched (high level) | Output |
|---|---|---|
| 1 | `pyproject.toml`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_imports.py`, `requirements.txt` + spec commit | pytest rails, 2 tests |
| 2 | `tests/unit/test_team_mjcf.py`, delete `check_team_mjcf.py`, Makefile | 1 unit test |
| 3 | `tests/conftest.py` (helpers), `tests/unit/test_crash_detector.py`, delete `check_team_crash.py`, Makefile | 3 unit tests, helpers extracted |
| 4 | `tests/unit/test_tag_state_machine.py`, delete `check_team_tag.py`, Makefile | 1 unit test |
| 5 | `tests/unit/test_render_smoke.py` | 1 new unit test |
| 6 | `tests/integration/test_team_env_canary.py`, delete `check_team_env.py`, Makefile | 1 integration canary |
| 7 | `tests/integration/test_simple_env_contract.py`, `test_scoring_canary.py`, delete `check_env.py`, Makefile | 3 integration tests, all 7 source scripts gone after this |
| 8 | `tests/integration/test_warm_start.py`, delete `check_team_warm.py`, Makefile | 1 skip-if-no-artifact test |
| 9 | `demo/scenarios.py` (scaffold) | event log + main loop |
| 10 | `demo/scenarios.py` (Scenario A tuning) | tagged-out narrative |
| 11 | `demo/scenarios.py` (Scenario B tuning) | score-despite-tag narrative |
| 12 | `demo/menu.py` | scenarios picked from `make demo` |
| 13 | `Makefile` | new test/test-fast/test-warm targets |
| 14 | `brain/*` | persistent memory updated |
| 15 | (verification only) | merge handoff |

**Final state at end of plan:**
- 12 tests passing, 1 skipped (warm-start without artifact).
- 0 `scripts/check_*.py` files.
- 0 `make check-*` / `make team-check*` Makefile targets.
- 3 new Makefile targets: `test`, `test-fast`, `test-warm`.
- 1 new demo entry: `make demo → scenarios`.
- `brain/` updated with the migration's context.
- One feature branch (`feature/pytest-migration`) ready to merge `--no-ff` into `develop`.
