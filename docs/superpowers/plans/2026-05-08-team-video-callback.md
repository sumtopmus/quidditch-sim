# Team-env Eval Video Callback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `VideoRecorderCallback` through to team training so `make train-team-red`, `make train-team-red-warm`, and `make train-team-blue` produce eval-trigger videos under `runs/<name>/<trial>/videos/step_*.mp4` and embed them in TensorBoard at `eval/video/<cam>` — mirroring the single-agent behaviour.

**Architecture:** Three coordinated changes: (1) rename single-agent drone prefix `drone` → `red_0` in `envs/quidditch/simple_env.py` so the same cam names resolve in both modes; (2) make `VideoRecorderCallback` env-agnostic by switching `env._quad.render_cells` → `env._world.render_cells` and adding a `_world` passthrough property to `OpponentControlledEnv`; (3) move video callback construction into `_train_common.build_callbacks` behind an optional `video_env_fn` parameter, keeping `train_ppo.py` (single-agent canary path) untouched.

**Tech Stack:** Python 3.11, MuJoCo 3.x, Stable-Baselines3 PPO, PettingZoo `ParallelEnv`, gymnasium, pytest, TOML config, imageio + moviepy<2.0 for mp4/TB writing.

**Worktree:** `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback` (branch `feature/team-video-callback` off `develop`). All `git -C` commands below use this path.

**Commit handling:** The user signs commits with a hardware key and runs `git commit` themselves. Every "commit" step in this plan should: (a) stage files with `git add`, (b) print the commit command to the user, (c) pause for the user to confirm the commit landed before continuing. Never run `git commit` directly.

---

## Task 1: Establish baseline canaries

Verify the develop-state canaries pass before any change so post-change regressions are unambiguous.

**Files:**
- Read-only: `tests/integration/test_scoring_canary.py`, `tests/unit/`, `tests/integration/`

- [ ] **Step 1.1: Activate the conda env**

The MuJoCo / SB3 stack lives in the `uav` conda env (see `brain/index.md`).

Run:
```bash
conda activate uav
```

Expected: prompt prefix shows `(uav)`. If `conda activate` is not available in this shell, run `eval "$(conda shell.bash hook)" && conda activate uav` once at the top of the session.

- [ ] **Step 1.2: Run the fast unit suite**

Run from the worktree root:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test-fast
```

Expected: all unit tests pass (`tests/unit/test_team_mjcf.py`, `tests/unit/test_tag_state_machine.py`, `tests/unit/test_crash_detector.py`, `tests/unit/test_render_smoke.py`, `tests/test_imports.py`). Total runtime under ~30s.

- [ ] **Step 1.3: Run the scoring canary specifically**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/integration/test_scoring_canary.py -v
```

Expected: PASS. The test asserts `SCORED at step 434 / total reward 7.3837`. **Record this output verbatim**; we will compare against it after the prefix rename in Task 3.

- [ ] **Step 1.4: Run the full test suite**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test
```

Expected: all tests pass except `tests/integration/test_warm_start.py` which skips with "MODEL env var not set" — that's the expected gate behaviour for the warm-start canary.

If any of these fail on develop unmodified, **stop and investigate** — the develop branch is broken before we touched anything.

---

## Task 2: Rename single-agent prefix `drone` → `red_0`

Refactor task — preserve behaviour, change name. The scoring canary from Task 1 is the verification harness.

**Files:**
- Modify: `envs/quidditch/simple_env.py:155`, `:166`, `:167`

The simple env is the only place that constructs the single drone with prefix `"drone"`. Other references (`core/quadrotor.py:81` default, `core/drone/cf2x.py:153` default, `tests/unit/test_render_smoke.py:27` default usage) are *defaults that callers can override*, and we want to leave them unchanged: they document the historical convention and the team env passes its prefix explicitly. Only `simple_env.py` actively pins the prefix.

- [ ] **Step 2.1: Survey current "drone" prefix usage**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && grep -rn '"drone"\|prefix=.drone\|drone_imu\|drone_framepos\|drone_velocimeter\|drone_gyro\|drone_framequat\|drone_probe\|drone_tag' --include="*.py" --include="*.toml" . 2>/dev/null | grep -v "/runs/\|/models/" | head -30
```

Expected output (the only call sites that pin `"drone"`):
```
core/quadrotor.py:9:    Quadrotor(world, prefix="drone")
core/quadrotor.py:77:    prefix is ``"drone"`` so the body is named ``"drone"``, sensors
core/quadrotor.py:78:    ``"drone_gyro"`` etc.
core/quadrotor.py:81:    def __init__(self, world: World, prefix: str = "drone") -> None:
core/quadrotor.py:253:            cf2x_fragment(prefix="drone"),
core/quadrotor.py:260:        quad = cls(world, prefix="drone")
core/drone/cf2x.py:153:    prefix: str = "drone",
core/drone/cf2x.py:164:            site, scoring probe, and the four sensors.  Default "drone"
tests/unit/test_render_smoke.py:27:        cf2x_fragment(),  # default prefix "drone", visual-only
envs/quidditch/simple_env.py:155:                cf2x_fragment(prefix="drone"),
envs/quidditch/simple_env.py:166:            self._quad = Quadrotor(self._world, prefix="drone")
envs/quidditch/simple_env.py:167:            self._scorer = GeomDistanceScorer(self._world, ["drone"], ["hoop"])
```

If any other call site appears (e.g. in `demo/`, `scripts/`, or new `tests/`), **stop and surface it** — those need a decision before continuing.

- [ ] **Step 2.2: Rename in simple_env.py**

Edit `envs/quidditch/simple_env.py`. Three changes inside the `_build_world` method (around lines 152-167):

```python
# OLD:
            self._world = World(
                fragments=[
                    cf2x_assets(),
                    cf2x_fragment(prefix="drone"),
                    hoop_fragment(...),
                    arena_wall_fragment(...),
                ],
                render=(self.render_mode == "human"),
                seed=seed,
            )
            self._quad = Quadrotor(self._world, prefix="drone")
            self._scorer = GeomDistanceScorer(self._world, ["drone"], ["hoop"])

# NEW:
            self._world = World(
                fragments=[
                    cf2x_assets(),
                    cf2x_fragment(prefix="red_0"),
                    hoop_fragment(...),
                    arena_wall_fragment(...),
                ],
                render=(self.render_mode == "human"),
                seed=seed,
            )
            self._quad = Quadrotor(self._world, prefix="red_0")
            self._scorer = GeomDistanceScorer(self._world, ["red_0"], ["hoop"])
```

(Read the file first; the `World(...)` call has more arguments — only change the three `"drone"` → `"red_0"` occurrences shown.)

Use `Edit` with `replace_all=true` on the literal `"drone"` token only if you confirm by Read that the only `"drone"` occurrences in `simple_env.py` are these three.

- [ ] **Step 2.3: Add the load-bearing-prefix comment near `_build_world`**

The team env's `_build_agent_obs` already mirrors the simple-env obs slot layout (`brain/index.md` line 35). The simple-env prefix is now part of the team-warm-start contract too. Add a comment immediately above the `cf2x_fragment(prefix="red_0")` line:

```python
            # Prefix "red_0" matches the team env's attacker prefix so the same
            # per-drone cam names (e.g. "red_0_tpv") resolve in both training
            # modes.  Renaming this also renames every body/sensor/cam emitted
            # by cf2x_fragment — see tests/integration/test_scoring_canary.py
            # for the physics-equivalence canary.
```

Single comment, ~5 lines. Don't expand it.

- [ ] **Step 2.4: Run the scoring canary**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/integration/test_scoring_canary.py -v
```

Expected: PASS, asserting `SCORED at step 434 / total reward 7.3837` — **byte-identical** to the Task 1.3 baseline.

If it fails or prints different step/reward values, the prefix rename is reordering MuJoCo state somehow. **STOP and investigate** before moving on. The likely suspect would be alphabetic sorting in `merge_all` (`core/mjcf/`) or `build_mjcf` — though both currently preserve insertion order. Do NOT proceed with the rest of the plan until the canary holds.

- [ ] **Step 2.5: Run the full test suite**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test
```

Expected: all tests pass (warm-start test still skips on missing `MODEL=`).

- [ ] **Step 2.6: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add envs/quidditch/simple_env.py
```

Then surface this commit command to the user and pause:

```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
refactor: rename single-agent drone prefix to red_0

Aligns simple-env body/sensor/cam namespace with the team env, so the
same per-drone cam names (red_0_tpv, etc.) resolve in both training
modes.  Scoring canary unchanged (step 434 / 7.3837).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Wait for the user to confirm before continuing.

---

## Task 3: Add `_world` passthrough to `OpponentControlledEnv`

The video callback's new contract is `env._world.render_cells(...)`. `OpponentControlledEnv` doesn't expose `_world` today, so we add a one-line passthrough property. TDD: failing test → property → green.

**Files:**
- Create: `tests/unit/test_opponent_env_world.py`
- Modify: `envs/quidditch/opponents.py` (add property to `OpponentControlledEnv`, ~line 250)

- [ ] **Step 3.1: Write the failing test**

Create `tests/unit/test_opponent_env_world.py`:

```python
"""OpponentControlledEnv exposes the underlying World via a passthrough
``_world`` property so the video callback can render through the wrapper."""
from __future__ import annotations

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec


def test_opponent_env_exposes_world_property() -> None:
    team = QuidditchTeamEnv(cfg=TeamConfig(), render_mode="rgb_array")
    env = OpponentControlledEnv(
        team,
        learner_id="red_0",
        opponent=from_spec("beeline_blue"),
    )
    env.reset(seed=0)

    # _world on the wrapper is the same object as on the inner team env.
    assert env._world is team._world
    # And it's a usable World — has the rendering API the callback needs.
    assert hasattr(env._world, "render_cells")
    assert hasattr(env._world, "render_frame")
```

- [ ] **Step 3.2: Run the test — confirm it fails**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_opponent_env_world.py -v
```

Expected: FAIL with `AttributeError: 'OpponentControlledEnv' object has no attribute '_world'`.

- [ ] **Step 3.3: Add the `_world` passthrough property**

Edit `envs/quidditch/opponents.py`. Inside the `OpponentControlledEnv` class (find it around line 195), add a property — the right place is near the bottom of the class, just before or after `def render(self):`:

```python
    @property
    def _world(self):
        """Passthrough to the wrapped team env's World so the video callback
        (and any other reach-through consumer) can call ``render_cells`` /
        ``render_frame`` without knowing about the wrapper."""
        return self.team_env._world
```

Don't import `World` for the type annotation; leave the return type implicit (a stringified `"World"` would force a circular-import dance for one method).

- [ ] **Step 3.4: Run the test — confirm green**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_opponent_env_world.py -v
```

Expected: PASS.

- [ ] **Step 3.5: Run the fast suite for regression check**

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test-fast
```

Expected: all unit tests still pass.

- [ ] **Step 3.6: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add envs/quidditch/opponents.py tests/unit/test_opponent_env_world.py
```

Surface to user:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
feat(opponents): add _world passthrough to OpponentControlledEnv

Exposes the wrapped team env's World on the single-agent shim so the
video callback can call render_cells/render_frame without knowing
about the wrapper layer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Make `VideoRecorderCallback` env-agnostic

Switch the callback's grid-mode reach-in from `env._quad.render_cells` to `env._world.render_cells`. This is the line that couples it to the simple env. TDD via a new team-env capture-cells test that fails on the unmodified callback.

**Files:**
- Create: `tests/unit/test_video_callback_team.py`
- Modify: `scripts/callbacks.py:136-150` (the `_capture_cells` method) and surrounding docstring

- [ ] **Step 4.1: Write the failing test**

Create `tests/unit/test_video_callback_team.py`:

```python
"""VideoRecorderCallback captures render cells from a team env wrapped in
OpponentControlledEnv — the same way it does for QuidditchSimpleEnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from scripts.callbacks import VideoRecorderCallback


def _make_team_env() -> OpponentControlledEnv:
    team = QuidditchTeamEnv(cfg=TeamConfig(), render_mode="rgb_array")
    return OpponentControlledEnv(
        team,
        learner_id="red_0",
        opponent=from_spec("beeline_blue"),
    )


def test_capture_cells_team_grid(tmp_path: Path) -> None:
    cb = VideoRecorderCallback(
        env_fn=_make_team_env,
        video_dir=str(tmp_path),
        record_freq=1,
        sim_hz=120,
        grid=True,
        grid_cams=("south", "east", "top", "fixed"),
        cell_width=64,
        cell_height=48,
    )
    env = _make_team_env()
    env.reset(seed=0)

    cells = cb._capture_cells(env)

    assert cells is not None
    assert len(cells) == 4
    for cell in cells:
        assert cell.shape == (48, 64, 3)
        assert cell.dtype == np.uint8
        # Renders are not all-black — every cam sees the arena.
        assert int(cell.sum()) > 0


def test_capture_cells_team_single_cam(tmp_path: Path) -> None:
    cb = VideoRecorderCallback(
        env_fn=_make_team_env,
        video_dir=str(tmp_path),
        record_freq=1,
        sim_hz=120,
        grid=False,
    )
    env = _make_team_env()
    env.reset(seed=0)

    cells = cb._capture_cells(env)

    assert cells is not None
    assert len(cells) == 1
    assert cells[0].ndim == 3 and cells[0].shape[2] == 3
    assert cells[0].dtype == np.uint8
```

- [ ] **Step 4.2: Run the test — confirm it fails**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_video_callback_team.py -v
```

Expected: `test_capture_cells_team_grid` FAILS with `AttributeError: 'OpponentControlledEnv' object has no attribute '_quad'`. (`test_capture_cells_team_single_cam` may already pass since it goes through `env.render()`.)

- [ ] **Step 4.3: Migrate `_capture_cells` to `env._world`**

Edit `scripts/callbacks.py`. Replace the `_capture_cells` method (around line 136) and update the class docstring to drop simple-env-only language.

In the class docstring (~lines 56-77), replace:

```python
    By default writes a 2x2 grid stitching four named cameras together
    (south / east / top / chase-cam) at 1920x1080.  Pass ``grid=False`` to
    fall back to the env's single-cam ``render()`` output (the cinematic
    "fixed" cam at 640x480).  Grid mode bypasses ``env.render()`` and reaches
    into ``env._quad.render_grid(...)`` directly — same precedent as
    eval_ppo.py reaching into ``env._quad`` for the live viewer.
```

with:

```python
    By default writes a 2x2 grid stitching four named cameras together at
    1920x1080.  Pass ``grid=False`` to fall back to the env's single-cam
    ``render()`` output (the cinematic "fixed" cam at 640x480).  Grid mode
    bypasses ``env.render()`` and reaches into ``env._world.render_cells(...)``
    directly — works for any env (single-agent QuidditchSimpleEnv or team
    QuidditchTeamEnv wrapped in OpponentControlledEnv) that exposes ``_world``.
```

In `_capture_cells` (around line 136-150), replace:

```python
    def _capture_cells(self, env) -> list[np.ndarray] | None:
        """Capture per-cam frames for one timestep.

        Grid mode returns one (cell_h × cell_w × 3) array per cam in
        ``self.grid_cams``.  Single-cam mode returns a 1-element list
        wrapping ``env.render()`` (the cinematic "fixed" cam).  Stored
        unstitched so each cam can be logged to TB independently; the
        on-disk mp4 stitches them at write-time.
        """
        if self.grid:
            return env._quad.render_cells(
                self.grid_cams, self.cell_width, self.cell_height
            )
        frame = env.render()
        return [frame] if frame is not None else None
```

with:

```python
    def _capture_cells(self, env) -> list[np.ndarray] | None:
        """Capture per-cam frames for one timestep.

        Grid mode returns one (cell_h × cell_w × 3) array per cam in
        ``self.grid_cams``.  Single-cam mode returns a 1-element list
        wrapping ``env.render()`` (the cinematic "fixed" cam).  Stored
        unstitched so each cam can be logged to TB independently; the
        on-disk mp4 stitches them at write-time.
        """
        if self.grid:
            return env._world.render_cells(
                self.grid_cams, self.cell_width, self.cell_height
            )
        frame = env.render()
        return [frame] if frame is not None else None
```

(Single change: `env._quad` → `env._world`.)

- [ ] **Step 4.4: Run the team-env tests — confirm green**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_video_callback_team.py -v
```

Expected: both tests PASS.

- [ ] **Step 4.5: Run the full test suite — single-agent regression check**

The single-agent training path also reaches `_capture_cells`. Run:

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test
```

Expected: all tests pass (warm-start still skips). The scoring canary is the strongest single-agent regression check.

- [ ] **Step 4.6: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add scripts/callbacks.py tests/unit/test_video_callback_team.py
```

Surface to user:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
refactor(callbacks): make VideoRecorderCallback env-agnostic via _world

Switches the grid-mode reach-in from env._quad.render_cells to
env._world.render_cells.  Same effective call (Quadrotor.render_cells
is already a passthrough to World.render_cells), but works for any
env that exposes _world — including the OpponentControlledEnv shim
used by team training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `video_env_fn` parameter to `_train_common.build_callbacks`

Move video callback construction into the shared callback factory, behind an optional `video_env_fn`. When `None`, behave as today; when provided, build a `VideoRecorderCallback` from the `[training.callbacks.video]` config.

**Files:**
- Create: `tests/unit/test_train_common_video.py`
- Modify: `scripts/_train_common.py:110-144`

- [ ] **Step 5.1: Write the failing test**

Create `tests/unit/test_train_common_video.py`:

```python
"""build_callbacks accepts an optional video_env_fn that, when provided,
appends a VideoRecorderCallback configured from the shared
[training.callbacks.video] block."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.quidditch.simple_env import QuidditchSimpleEnv
from scripts._train_common import build_callbacks
from scripts.callbacks import VideoRecorderCallback


_BASE_CFG: dict = {
    "training": {
        "eval": {"eval_freq_steps": 1000, "n_eval_episodes": 1},
        "callbacks": {
            "checkpoint_freq_steps": 1000,
            "video_every_n_evals": 4,
            "video_fps": 20,
            "video": {
                "grid": True,
                "cells": ["south", "east", "top", "fixed"],
                "cell_width": 64,
                "cell_height": 48,
            },
        },
    }
}


def test_build_callbacks_without_video(tmp_path: Path) -> None:
    cbs = build_callbacks(
        run_dir=tmp_path,
        eval_env_fn=lambda: QuidditchSimpleEnv(),
        config=_BASE_CFG,
        n_envs=1,
    )
    assert any(isinstance(c, CheckpointCallback) for c in cbs)
    assert any(isinstance(c, EvalCallback) for c in cbs)
    assert not any(isinstance(c, VideoRecorderCallback) for c in cbs)


def test_build_callbacks_with_video(tmp_path: Path) -> None:
    cbs = build_callbacks(
        run_dir=tmp_path,
        eval_env_fn=lambda: QuidditchSimpleEnv(),
        config=_BASE_CFG,
        n_envs=1,
        video_env_fn=lambda: QuidditchSimpleEnv(render_mode="rgb_array"),
    )
    video_cbs = [c for c in cbs if isinstance(c, VideoRecorderCallback)]
    assert len(video_cbs) == 1

    vc = video_cbs[0]
    assert vc.grid is True
    assert vc.grid_cams == ("south", "east", "top", "fixed")
    assert vc.cell_width == 64
    assert vc.cell_height == 48
    assert vc.fps == 20
    # record_freq = (eval_freq_steps / n_envs) * video_every_n_evals = 1000 * 4 = 4000
    assert vc.record_freq == 4000
```

- [ ] **Step 5.2: Run the test — confirm `with_video` fails**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_train_common_video.py -v
```

Expected:
- `test_build_callbacks_without_video` PASSES (it's testing pre-existing behaviour)
- `test_build_callbacks_with_video` FAILS with `TypeError: build_callbacks() got an unexpected keyword argument 'video_env_fn'`

- [ ] **Step 5.3: Add `video_env_fn` parameter**

Edit `scripts/_train_common.py`. Replace the entire `build_callbacks` function (lines 110-144) with:

```python
def build_callbacks(
    *,
    run_dir: Path,
    eval_env_fn: Callable[[], Any],
    config: dict[str, Any],
    n_envs: int,
    video_env_fn: Callable[[], Any] | None = None,
) -> list:
    """Build the standard SB3 callback set: checkpoint + eval + (optional) video.

    When ``video_env_fn`` is provided, a ``VideoRecorderCallback`` is appended
    using the ``[training.callbacks.video]`` sub-block.  ``video_env_fn`` should
    return an env in ``rgb_array`` mode (single-agent ``QuidditchSimpleEnv`` or
    team-agent ``OpponentControlledEnv``); the callback resets and rolls out one
    deterministic episode at every ``video_every_n_evals``-th eval trigger.
    """
    eval_freq = max(config["training"]["eval"]["eval_freq_steps"] // n_envs, 1)
    ckpt_freq = max(
        config["training"]["callbacks"]["checkpoint_freq_steps"] // n_envs, 1
    )

    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([eval_env_fn])

    cbs: list = [
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=eval_freq,
            n_eval_episodes=config["training"]["eval"]["n_eval_episodes"],
            deterministic=True,
        ),
    ]

    if video_env_fn is not None:
        # Local import to avoid a hard dep on imageio/moviepy when video is off.
        from scripts.callbacks import VideoRecorderCallback
        from core.world import CONTROL_HZ

        video_cfg = config["training"]["callbacks"].get("video", {})
        video_freq = eval_freq * config["training"]["callbacks"]["video_every_n_evals"]
        cbs.append(
            VideoRecorderCallback(
                env_fn=video_env_fn,
                video_dir=str(run_dir / "videos"),
                record_freq=video_freq,
                fps=config["training"]["callbacks"]["video_fps"],
                sim_hz=CONTROL_HZ,
                grid=video_cfg.get("grid", True),
                grid_cams=tuple(video_cfg["cells"]) if "cells" in video_cfg else None,
                cell_width=video_cfg.get("cell_width", 960),
                cell_height=video_cfg.get("cell_height", 540),
            )
        )

    return cbs
```

(`CONTROL_HZ` is in `core/world.py:44`; the local import keeps `_train_common` decoupled.)

- [ ] **Step 5.4: Run the tests — confirm green**

Run:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && pytest tests/unit/test_train_common_video.py -v
```

Expected: both tests PASS.

- [ ] **Step 5.5: Run the fast suite for regression**

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test-fast
```

Expected: all unit tests pass.

- [ ] **Step 5.6: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add scripts/_train_common.py tests/unit/test_train_common_video.py
```

Surface to user:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
feat(_train_common): optional video_env_fn appends VideoRecorderCallback

When video_env_fn is provided, build_callbacks reads
[training.callbacks.video] and appends a VideoRecorderCallback
configured the same way single-agent does inline.  Default
behaviour (no video_env_fn) is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire video into `train_team_ppo.py`

Pass `video_env_fn` to `build_callbacks` from the team-training entry script. The video env mirrors the eval env construction (same opponent spec, same env config) but uses `render_mode="rgb_array"`.

**Files:**
- Modify: `scripts/train_team_ppo.py:122` (the `build_callbacks(...)` call)

- [ ] **Step 6.1: Read the current eval-env construction**

Read `scripts/train_team_ppo.py` from line 40 to line 130 to confirm the eval-env factory pattern. The relevant snippet (around lines 47-62) is the inner function `make_env_fn` that returns the SB3-wrappable env. Identify the variables in scope at the `build_callbacks(...)` call site (`learner_id`, `opp_spec`, `env_team_cfg`, `cfg`, etc.).

- [ ] **Step 6.2: Add the video env factory**

Edit `scripts/train_team_ppo.py`. Just before the `build_callbacks(...)` call at line 122, add a small factory closure:

```python
    # Video env factory: mirrors the eval env but in rgb_array mode so
    # the offscreen renderer produces frames.  build_callbacks only
    # appends the video callback when video_env_fn is provided.
    def _make_video_env() -> "OpponentControlledEnv":
        team = QuidditchTeamEnv(cfg=cfg_team, render_mode="rgb_array")
        opp  = from_spec(args.opponent, deterministic=True)
        return OpponentControlledEnv(
            team, learner_id=args.learner, opponent=opp,
        )
```

Use whatever variable names actually hold `cfg_team` and `args.opponent` / `args.learner` in the entry script — read the surrounding code first. The pattern parallels `make_env_fn` (the eval factory).

If `OpponentControlledEnv` and `from_spec` aren't already imported at module top, add them; the file already imports `QuidditchTeamEnv` and `TeamConfig` (per `:27`), so add to the existing import line:

```python
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
```

(Or extend if already present — check first.)

- [ ] **Step 6.3: Pass `video_env_fn` to `build_callbacks`**

Modify the `build_callbacks(...)` call (line 122) to add the kwarg:

```python
    callbacks = build_callbacks(
        run_dir=run_dir,
        eval_env_fn=make_env_fn,   # whatever it's already called
        config=config,
        n_envs=n_envs,
        video_env_fn=_make_video_env,
    )
```

- [ ] **Step 6.4: Verify the entry script imports cleanly**

Static import check:

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && python -c "import scripts.train_team_ppo"
```

Expected: no output, exit code 0. If it fails (import error, missing name), fix the import or factory before continuing.

- [ ] **Step 6.5: Run the fast suite**

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make test-fast
```

Expected: all unit tests pass.

- [ ] **Step 6.6: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add scripts/train_team_ppo.py
```

Surface to user:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
feat(train-team): wire VideoRecorderCallback into team training

Constructs an rgb_array-mode OpponentControlledEnv and passes its
factory through build_callbacks' new video_env_fn kwarg, so
make train-team-* now writes eval mp4s and embeds them in TB on
the same cadence as single-agent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update `[training.callbacks.video]` cells in config + template

Set the default cells to the arena-centric set chosen in brainstorming. Same default for both single-agent and team training (they now share the cam namespace via the prefix rename).

**Files:**
- Modify: `config/training.toml:32`
- Modify: `templates/training.toml` (corresponding line — verify by Read first)

- [ ] **Step 7.1: Update `config/training.toml`**

Read `config/training.toml`, find the `[training.callbacks.video]` block (around line 30), and change:

```toml
cells       = ["south", "east", "top", "tpv"]
```

to:

```toml
cells       = ["south", "east", "top", "fixed"]
```

Add an inline comment on the change rationale on the same line if the existing style uses inline comments — otherwise leave it bare.

- [ ] **Step 7.2: Update `templates/training.toml`**

Read `templates/training.toml` to find the matching block (it should mirror `config/training.toml`). Apply the same change.

- [ ] **Step 7.3: Verify the toml parses**

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && python -c "import tomllib; print(tomllib.load(open('config/training.toml','rb'))['training']['callbacks']['video']['cells'])"
```

Expected: `['south', 'east', 'top', 'fixed']`.

Also for the template:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && python -c "import tomllib; print(tomllib.load(open('templates/training.toml','rb'))['training']['callbacks']['video']['cells'])"
```

Expected: same output.

- [ ] **Step 7.4: Stage and commit (user-driven)**

Stage:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" add config/training.toml templates/training.toml
```

Surface to user:
```bash
git -C "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" commit -m "$(cat <<'EOF'
config: replace broken "tpv" cell with "fixed" in video grid default

The "tpv" cam name in single-agent never resolved (the actual cam was
prefix-scoped, e.g. drone_tpv → red_0_tpv after the prefix rename), so
the bottom-right cell was rendering a free-cam fallback shot.  Switch
to "fixed" — the cinematic broadcast cam — which works in both
training modes without per-mode tuning.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end smoke (manual)

Verify the wiring against the live code path: run a short team training and inspect the produced video.

**Files:** None modified. Read-only verification.

- [ ] **Step 8.1: Run a short team training**

This step depends on the `make train-team-red` target accepting a `TIMESTEPS=` override. Confirm by reading `Makefile:174-178` and `scripts/train_team_ppo.py` `argparse` setup. The eval cadence is `[training.eval].eval_freq_steps` and video fires every `video_every_n_evals`-th eval — pick a `TIMESTEPS` that triggers at least one eval-with-video.

For default `eval_freq_steps=50_000`, `video_every_n_evals=4` → first video at step 200,000. Run:

```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && make train-team-red TIMESTEPS=210000 RUN_NAME=video_smoke 2>&1 | tail -40
```

**Estimated runtime:** ~5-10 minutes on M-series CPU (no GPU). If you don't have time, skip this manual step but flag it in the commit summary as "manual smoke pending".

- [ ] **Step 8.2: Verify the mp4 exists and is non-empty**

After training finishes, find the run trial dir and check:

```bash
find "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback/runs/video_smoke" -name "step_*.mp4" -size +10k
```

Expected: at least one mp4 path printed, file size > 10 KB.

- [ ] **Step 8.3: Spot-check the video visually**

Open the mp4 in any player (QuickTime works on macOS):

```bash
open "$(find /Users/shurioque/Library/Mobile\ Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback/runs/video_smoke -name "step_*.mp4" | head -1)"
```

Expected:
- 2x2 grid layout
- All four cells render scene content (no all-black cells, no free-cam frames showing skybox only)
- Both Red and Blue drones visible (Red tinted (0.75, 0.10, 0.10, 1), Blue tinted (0.10, 0.20, 0.75, 1))
- Approximate duration: episode_seconds × video_fps frames at 20 fps ≈ 30s

If the video is broken or shows only one drone, the eval env or video env factory is misconfigured — debug the env construction in `train_team_ppo.py`.

- [ ] **Step 8.4: Verify TB shows the videos**

```bash
ls "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback/runs/video_smoke"/*/events.out.tfevents.*
```

Then briefly:
```bash
cd "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback" && tensorboard --logdir runs/video_smoke --port 6007 &
```

Open http://localhost:6007 in a browser, navigate to **Images** → look for `eval/video/south`, `eval/video/east`, `eval/video/top`, `eval/video/fixed`. Each should have a clip at step 200,000.

Kill TB after verification:
```bash
pkill -f "tensorboard --logdir runs/video_smoke"
```

(If you can't run TB locally, skip this step — the on-disk mp4 from Step 8.3 is sufficient evidence the pipeline works.)

- [ ] **Step 8.5: Clean up the smoke-test run**

```bash
rm -rf "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/team-video-callback/runs/video_smoke"
```

(Optional: keep it if you want it for further verification.)

No commit for this task — verification only.

---

## Task 9: Update brain notes

Mark the team-video-callback known issue as resolved and record the prefix rename for future agents.

**Files:**
- Modify: `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/brain/index.md`
- Modify: `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/brain/changelog.md`

The brain dir lives at the **umbrella level**, OUTSIDE the worktree. Edit it directly at the umbrella path (these files are not committed by git).

- [ ] **Step 9.1: Remove the "Known Issues" line about team videos**

Read `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/brain/index.md` line 36. Delete the line:

```
- **Team-env training has no video callback yet** — `_train_common.build_callbacks` skips video (coupled to simple-env render). Punted; track via the open follow-up.
```

- [ ] **Step 9.2: Update the "Recent Context" section**

Add a new top entry under "Recent Context" with today's date, briefly noting:
- `feature/team-video-callback` work landed: `VideoRecorderCallback` env-agnostic via `_world`, `_world` passthrough on `OpponentControlledEnv`, `video_env_fn` parameter on `build_callbacks`.
- Single-agent prefix renamed `drone` → `red_0`. Scoring canary unchanged (still `step 434 / 7.3837`).
- Default video grid changed `["south", "east", "top", "tpv"]` → `["south", "east", "top", "fixed"]` — fixes a latent bug where `tpv` never resolved (cam was always `drone_tpv`, now `red_0_tpv`).

Keep the entry concise (~5-10 lines), in the existing style.

- [ ] **Step 9.3: Update changelog.md**

Append an entry to `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/brain/changelog.md` matching the existing per-day style. Files modified: list those touched in tasks 2-7 above.

- [ ] **Step 9.4: No git commit**

Brain files are deliberately outside the git repo (`umbrella/brain/`, not `umbrella/repo/brain/`). Do not commit them.

---

## Self-review

Re-read against the spec at `docs/superpowers/specs/2026-05-08-team-video-callback-design.md`:

- ✅ Prefix rename `drone` → `red_0` (Task 2)
- ✅ Canary verification (Steps 1.3 + 2.4)
- ✅ `_world` passthrough on `OpponentControlledEnv` (Task 3)
- ✅ `VideoRecorderCallback._capture_cells` switched to `_world` (Task 4)
- ✅ `video_env_fn` parameter on `build_callbacks` (Task 5)
- ✅ Wire-up in `train_team_ppo.py` (Task 6)
- ✅ Config default updated (Task 7)
- ✅ Smoke test verifying end-to-end (Task 8)
- ✅ Brain notes updated (Task 9)
- ✅ All commits user-driven (no `git commit` runs in any task)
- ✅ Tests on every code-changing task (Tasks 3, 4, 5)
- ✅ No placeholder text — every code block is concrete
- ✅ File paths and commands are absolute and copy-paste ready

Out of scope per spec — confirmed not in plan:
- Per-drone `tpv`/`fpv` defaults that auto-substitute the learner prefix
- Generalizing single-agent `train_ppo.py` to use `_train_common.build_callbacks`
