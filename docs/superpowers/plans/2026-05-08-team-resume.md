# Team-Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--resume` to `scripts/train_team_ppo.py` so paused/finished team-play training runs can be continued from a checkpoint, with the current config's `learning_rate` taking effect on resume.

**Architecture:** Mirror the existing single-agent resume pattern (`train_ppo.py:264-270, 372-384`). Resume creates a *new* timestamped trial dir under the same `run_name`, loads the SB3 checkpoint, overrides `learning_rate` from config, and trains until the absolute `total_timesteps` target is reached. A `[resume]` block in `info.toml` records the source checkpoint. A new `make resume-team` Makefile target provides the user-facing CLI.

**Tech Stack:** Python 3.11, Stable-Baselines3 PPO, MuJoCo, pytest, Make. The `uav` conda env is the runtime; `KMP_DUPLICATE_LIB_OK=TRUE` is already set on the team scripts to suppress macOS conda libomp duplicate-init aborts.

**User constraint:** This user runs `git commit` manually for hardware-key signing. Each task ends with a "ready-to-commit command" step — **do not run `git commit` yourself**; print the command for the user to paste.

**Worktree:** All work happens in `worktrees/feature/team-resume/` on branch `feature/team-resume`. The worktree was already created by the planning agent — verify with `pwd` and `git branch --show-current` at start.

---

## Reference Reading (do this first)

Before Task 1, read these files end-to-end so the patterns are in your head:

- [scripts/train_ppo.py:264-270, 369-413, 459-469](../../../scripts/train_ppo.py) — single-agent `--resume` and `--pretrain` patterns to mirror
- [scripts/train_team_ppo.py](../../../scripts/train_team_ppo.py) — current team training script (the file you'll modify)
- [scripts/_train_common.py:56-104](../../../scripts/_train_common.py) — `write_run_info` (extend in Task 4)
- [scripts/callbacks.py:15-42](../../../scripts/callbacks.py) — `ResumeProgressCallback` (reuse, no changes)
- [tests/integration/test_team_env_canary.py](../../../tests/integration/test_team_env_canary.py) — canary test to keep green
- [tests/integration/test_warm_start.py](../../../tests/integration/test_warm_start.py) — pattern reference for SB3+team-env integration tests
- [docs/superpowers/specs/2026-05-08-team-resume-design.md](../specs/2026-05-08-team-resume-design.md) — the spec this plan implements

---

## Task 1: Argparse — add `--resume`, mutually exclude with `--warm-start`

**Files:**
- Modify: `scripts/train_team_ppo.py:34-44` (the `parse_args` function)
- Test: `tests/unit/test_team_resume_args.py` (create)

- [ ] **Step 1: Write the failing unit test**

Create `tests/unit/test_team_resume_args.py`:

```python
"""Argparse contract for train_team_ppo.py: --resume and --warm-start are
mutually exclusive; --resume is captured into args.resume."""
from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest


@contextmanager
def _argv(*args: str):
    saved = sys.argv
    sys.argv = ["train_team_ppo.py", *args]
    try:
        yield
    finally:
        sys.argv = saved


def _parse():
    # Import inside the test so each call re-evaluates argparse defaults.
    from scripts.train_team_ppo import parse_args
    return parse_args()


def test_resume_is_captured() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--resume", "runs/x/y/checkpoints/ppo_1000_steps.zip"):
        args = _parse()
    assert args.resume == "runs/x/y/checkpoints/ppo_1000_steps.zip"
    assert args.warm_start == ""


def test_warm_start_and_resume_are_mutually_exclusive() -> None:
    with _argv("--learner", "blue_0", "--opponent", "beeline_red",
               "--resume", "ckpt.zip", "--warm-start", "model.zip"):
        with pytest.raises(SystemExit):
            _parse()


def test_resume_default_is_none() -> None:
    with _argv("--learner", "red_0", "--opponent", "beeline_blue"):
        args = _parse()
    assert args.resume is None
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_args.py -v
```

Expected: All three tests fail. The first two with `AttributeError: 'Namespace' object has no attribute 'resume'`, the third the same.

- [ ] **Step 3: Add `--resume` to argparse with mutual exclusion**

Edit `scripts/train_team_ppo.py`. Replace lines 34-44 (the entire `parse_args` function) with:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuidditchTeamEnv via SB3 PPO")
    p.add_argument("--learner", choices=["red_0", "blue_0"], required=False, default=None)
    p.add_argument("--opponent", required=False, default=None,
                   help="opponent spec, e.g. 'beeline_blue'")
    p.add_argument("--config", default="config/training.toml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs",    type=int, default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--seed",      type=int, default=None)

    # --warm-start and --resume are mutually exclusive: warm-start does the
    # 16->22 input-layer surgery from a single-agent checkpoint; resume picks
    # up an existing team-play trial mid-flight (same I/O dimensions, keeps
    # optimizer state, keeps step counter).
    g = p.add_mutually_exclusive_group()
    g.add_argument("--warm-start", default="",
                   help="path to old single-agent best_model.zip (input-layer surgery)")
    g.add_argument("--resume", default=None, metavar="PATH",
                   help="path to a checkpoint .zip to resume from. Keeps the step "
                        "counter; trains for the remaining steps to total_timesteps. "
                        "Loads --learner / --opponent from the parent trial's info.toml "
                        "if not given on the CLI.")
    return p.parse_args()
```

Note `--learner` and `--opponent` become optional — Task 6 reads them from the parent trial when resuming.

- [ ] **Step 4: Run the test, verify it passes**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_args.py -v
```

Expected: All three tests pass.

- [ ] **Step 5: Verify nothing else broke**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit -v
```

Expected: All unit tests pass.

- [ ] **Step 6: Print the ready-to-commit command for the user**

Print exactly this for the user to copy/paste (do **not** run `git commit` yourself — this user uses a hardware key for signing):

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/train_team_ppo.py tests/unit/test_team_resume_args.py
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(team-resume): add --resume CLI arg, mutually exclusive with --warm-start"
```

Wait for the user's confirmation that the commit landed before proceeding to Task 2.

---

## Task 2: Add a "missing args without resume" guard

When `--resume` is *not* given, `--learner` and `--opponent` are still required (they were required before; we only relaxed argparse to support resume). Re-add the validation in `main()`.

**Files:**
- Modify: `scripts/train_team_ppo.py:main()` (top of function)
- Test: `tests/unit/test_team_resume_args.py` (extend)

- [ ] **Step 1: Add a failing test for the guard**

Append to `tests/unit/test_team_resume_args.py`:

```python
def test_missing_learner_without_resume_errors_in_main(monkeypatch, capsys) -> None:
    """Without --resume, --learner is still required."""
    from scripts import train_team_ppo

    with _argv("--opponent", "beeline_blue"):  # no --learner, no --resume
        with pytest.raises(SystemExit):
            train_team_ppo.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_args.py::test_missing_learner_without_resume_errors_in_main -v
```

Expected: Fails — `main()` will likely raise `KeyError` or pass `learner=None` into the env wiring.

- [ ] **Step 3: Add the guard at the top of `main()`**

In `scripts/train_team_ppo.py`, find the `def main()` line and add this guard immediately after `args = parse_args()`:

```python
    if args.resume is None:
        if args.learner is None or args.opponent is None:
            print("ERROR: --learner and --opponent are required unless --resume is given",
                  file=sys.stderr)
            sys.exit(2)
```

- [ ] **Step 4: Run, verify it passes**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_args.py -v
```

Expected: All four tests pass.

- [ ] **Step 5: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/train_team_ppo.py tests/unit/test_team_resume_args.py
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(team-resume): require --learner/--opponent only when not resuming"
```

---

## Task 3: Extend `_train_common.write_run_info` with a `resume=` block

**Files:**
- Modify: `scripts/_train_common.py:56-104` (the `write_run_info` function)
- Test: `tests/unit/test_write_run_info_resume.py` (create)

- [ ] **Step 1: Write the failing unit test**

Create `tests/unit/test_write_run_info_resume.py`:

```python
"""write_run_info emits a [resume] block when resume= is given."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import tomllib


def test_resume_block_is_emitted(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "ppo_hoop_blue_1" / "20260508_010101"
    run_dir.mkdir(parents=True)

    args = argparse.Namespace(run_name="ppo_hoop_blue_1", config=None)
    started = datetime(2026, 5, 8, 1, 1, 1)

    write_run_info(
        run_dir,
        config={},
        args=args,
        extra={"learner": "blue_0",
               "opponent_spec": "frozen:models/red/best_model"},
        resume={"checkpoint": "runs/ppo_hoop_blue_1/20260507_194423/"
                              "checkpoints/ppo_10000000_steps.zip",
                "resumed_at": 10002432},
        started=started,
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert info["resume"]["checkpoint"].endswith("ppo_10000000_steps.zip")
    assert info["resume"]["resumed_at"] == 10002432
    # extra block still present
    assert info["extra"]["learner"] == "blue_0"


def test_no_resume_block_when_not_given(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "ppo_hoop_red_1" / "20260508_010101"
    run_dir.mkdir(parents=True)

    args = argparse.Namespace(run_name="ppo_hoop_red_1", config=None)

    write_run_info(
        run_dir, config={}, args=args,
        extra={"learner": "red_0", "opponent_spec": "beeline_blue"},
        started=datetime(2026, 5, 8, 1, 1, 1),
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert "resume" not in info
    assert info["extra"]["learner"] == "red_0"
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_write_run_info_resume.py -v
```

Expected: First test fails (`write_run_info() got an unexpected keyword argument 'resume'`); second test passes (it doesn't use `resume=`).

- [ ] **Step 3: Extend `write_run_info`**

In `scripts/_train_common.py`, replace the `write_run_info` function signature and body. The current signature (lines 56-65) is:

```python
def write_run_info(
    run_dir: Path,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
    started: datetime | None = None,
    elapsed_s: float | None = None,
    steps_trained: int | None = None,
) -> None:
```

Add `resume: dict[str, Any] | None = None,` after `extra=`:

```python
def write_run_info(
    run_dir: Path,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    started: datetime | None = None,
    elapsed_s: float | None = None,
    steps_trained: int | None = None,
) -> None:
```

Then, inside the function body, after the `extra_block` definition (around line 90, just before `content = (` is built), add:

```python
    resume_block = ""
    if resume:
        resume_block = (
            "\n[resume]\n"
            f'checkpoint  = "{resume["checkpoint"]}"\n'
            f'resumed_at  = {resume["resumed_at"]}\n'
        )
```

Then in the `content = (...)` literal, insert `f"{resume_block}"` immediately before `f"{extra_block}"`:

```python
    content = (
        "# Run info — written by train_team_ppo.py.\n"
        "\n"
        "[run]\n"
        f'name          = "{getattr(args, "run_name", None) or run_dir.parent.name}"\n'
        f'trial         = "{run_dir.name}"\n'
        f'started       = "{started.isoformat(timespec="seconds")}"\n'
        f"{elapsed_line}\n"
        f"{finished_line}\n"
        f"{steps_line}\n"
        f"{resume_block}"
        f"{extra_block}"
    )
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_write_run_info_resume.py -v
```

Expected: Both tests pass.

- [ ] **Step 5: Verify all unit tests still pass**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit -v
```

Expected: All unit tests pass.

- [ ] **Step 6: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/_train_common.py tests/unit/test_write_run_info_resume.py
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(_train_common): write_run_info accepts resume= for [resume] block"
```

---

## Task 4: Helper — read learner/opponent from a parent trial

When `--resume <path>` is given without explicit `--learner` / `--opponent`, read them from the parent trial's `info.toml [extra]` block.

**Files:**
- Modify: `scripts/train_team_ppo.py` (add a private helper near the top)
- Test: `tests/unit/test_team_resume_parent_lookup.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_team_resume_parent_lookup.py`:

```python
"""Resolving --learner / --opponent from a parent trial's info.toml."""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_parent_trial(tmp_path: Path, *,
                       learner: str = "blue_0",
                       opponent: str = "frozen:models/red/best_model") -> Path:
    trial = tmp_path / "ppo_hoop_blue_1" / "20260507_194423"
    (trial / "checkpoints").mkdir(parents=True)
    (trial / "info.toml").write_text(
        '[run]\n'
        'name = "ppo_hoop_blue_1"\n'
        'trial = "20260507_194423"\n'
        '\n'
        '[extra]\n'
        f'learner = "{learner}"\n'
        f'opponent_spec = "{opponent}"\n'
        'warm_start_from = ""\n'
    )
    ckpt = trial / "checkpoints" / "ppo_10000000_steps.zip"
    ckpt.write_bytes(b"")  # placeholder — only path-resolution is tested here
    return ckpt


def test_lookup_returns_extra_block_values(tmp_path: Path) -> None:
    from scripts.train_team_ppo import _read_parent_extra

    ckpt = _make_parent_trial(tmp_path)
    learner, opponent = _read_parent_extra(str(ckpt))
    assert learner == "blue_0"
    assert opponent == "frozen:models/red/best_model"


def test_lookup_returns_none_if_no_info_toml(tmp_path: Path) -> None:
    from scripts.train_team_ppo import _read_parent_extra

    bare = tmp_path / "ppo_hoop_x" / "20260101_000000" / "checkpoints"
    bare.mkdir(parents=True)
    ckpt = bare / "ppo_500_steps.zip"
    ckpt.write_bytes(b"")
    learner, opponent = _read_parent_extra(str(ckpt))
    assert learner is None
    assert opponent is None
```

- [ ] **Step 2: Run, verify it fails**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_parent_lookup.py -v
```

Expected: Fails with `ImportError: cannot import name '_read_parent_extra' from scripts.train_team_ppo`.

- [ ] **Step 3: Add the helper to `scripts/train_team_ppo.py`**

In `scripts/train_team_ppo.py`, after the imports (right after the `from scripts._train_common import ...` block) add:

```python
import tomllib


def _read_parent_extra(checkpoint_path: str) -> tuple[str | None, str | None]:
    """Walk from a checkpoint path up to its trial dir and read learner/opponent
    from info.toml [extra]. Returns (None, None) if anything is missing."""
    trial_dir = Path(checkpoint_path).resolve().parent.parent
    info = trial_dir / "info.toml"
    if not info.exists():
        return None, None
    try:
        data = tomllib.loads(info.read_text())
    except Exception:
        return None, None
    extra = data.get("extra", {})
    return extra.get("learner"), extra.get("opponent_spec")
```

- [ ] **Step 4: Run, verify it passes**

```bash
conda run --no-capture-output -n uav python -m pytest tests/unit/test_team_resume_parent_lookup.py -v
```

Expected: Both tests pass.

- [ ] **Step 5: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/train_team_ppo.py tests/unit/test_team_resume_parent_lookup.py
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(team-resume): _read_parent_extra resolves learner/opponent from parent info.toml"
```

---

## Task 5: Wire `--resume` into `main()` — load checkpoint, override lr, train to absolute target

**Files:**
- Modify: `scripts/train_team_ppo.py:main()` (the model-construction and `model.learn` blocks)

This is the central task. Read [train_ppo.py:369-413, 459-469](../../../scripts/train_ppo.py) before starting — the team version is structurally identical.

- [ ] **Step 1: Write a failing integration test**

Create `tests/integration/test_team_resume.py`:

```python
"""End-to-end: train a tiny team-play model, save a checkpoint, resume from it,
verify num_timesteps continues and info.toml has a [resume] block."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = pytest.mark.slow


def _make_env_fn():
    def _thunk():
        team = QuidditchTeamEnv(cfg=TeamConfig(
            randomise_red_start=False, episode_seconds=2.0))
        return OpponentControlledEnv(team, learner_id="red_0",
                                     opponent=from_spec("beeline_blue"))
    return _thunk


def test_resume_continues_step_counter_and_writes_resume_block(tmp_path: Path) -> None:
    """Train ~256 steps, save checkpoint, resume, train ~256 more, assert
    num_timesteps continues and info.toml has [resume]."""
    # 1) Cold-train a tiny model and save a checkpoint.
    env = DummyVecEnv([_make_env_fn()])
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32,
                n_epochs=1, learning_rate=3e-4, verbose=0)
    model.learn(total_timesteps=128)
    cold_steps = int(model.num_timesteps)
    assert cold_steps >= 128

    parent_dir = tmp_path / "runs" / "team_resume_test" / "20260508_000000"
    (parent_dir / "checkpoints").mkdir(parents=True)
    ckpt = parent_dir / "checkpoints" / f"ppo_{cold_steps}_steps.zip"
    model.save(str(ckpt))

    # Write a parent info.toml so _read_parent_extra works.
    (parent_dir / "info.toml").write_text(
        '[run]\nname = "team_resume_test"\ntrial = "20260508_000000"\n'
        '\n[extra]\n'
        'learner = "red_0"\n'
        'opponent_spec = "beeline_blue"\n'
        'warm_start_from = ""\n'
    )

    # Snapshot a minimal config the resume run will read.
    cfg_path = tmp_path / "training.toml"
    cfg_path.write_text(
        '[training]\n'
        'run_name = "team_resume_test"\n'
        'total_timesteps = 256\n'  # = cold_steps + ~128 more
        'n_envs = 1\n'
        'seed = 42\n'
        '\n[training.ppo]\n'
        'n_steps = 64\nbatch_size = 32\nn_epochs = 1\n'
        'lr = 1.5e-4\n'  # different from cold-train's 3e-4 — verify override
        'gamma = 0.99\ngae_lambda = 0.95\nclip_range = 0.2\nent_coef = 0.01\n'
        '\n[training.eval]\n'
        'eval_freq_steps = 10000\nn_eval_episodes = 1\n'
        '\n[training.callbacks]\n'
        'checkpoint_freq_steps = 10000\nvideo_every_n_evals = 999\nvideo_fps = 20\n'
        '\n[env]\n'
        'randomise_start = false\nepisode_seconds = 2.0\nmode = "team"\n'
        '\n[env.team]\n'
        'red_prefix = "red_0"\nblue_prefix = "blue_0"\nhoop_prefix = "hoop_0"\n'
        'midpoint_alpha = 0.5\ntag_radius = 0.3\ntag_cooldown_s = 1.0\n'
        'crash_vel_thr = 1.5\nwalls_collide = true\n'
        '\n[training.team]\n'
        'learner = "red_0"\nopponent_spec = "beeline_blue"\n'
        'warm_start_from = ""\n'
        '\n[training.team.warm_start]\n'
        'new_dim_init_scale = 0.01\n'
    )

    # 2) Resume from the checkpoint via subprocess (matches real CLI usage).
    env_overrides = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    cwd = tmp_path  # resume writes new trial under tmp_path/runs/...
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train_team_ppo.py"),
         "--resume", str(ckpt),
         "--config", str(cfg_path)],
        cwd=str(cwd), env=env_overrides,
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"resume failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"

    # 3) Find the new trial dir (sibling of the parent under team_resume_test/).
    runs_root = cwd / "runs" / "team_resume_test"
    trials = sorted(p for p in runs_root.iterdir()
                    if p.is_dir() and p.name != "20260508_000000")
    assert len(trials) >= 1, f"no resume trial dir created under {runs_root}"
    resume_trial = trials[-1]

    # 4) Verify info.toml has the [resume] block.
    info = tomllib.loads((resume_trial / "info.toml").read_text())
    assert "resume" in info, f"missing [resume] block: {info.keys()}"
    assert info["resume"]["checkpoint"].endswith(ckpt.name)
    assert info["resume"]["resumed_at"] == cold_steps
    assert info["extra"]["learner"] == "red_0"
    assert info["extra"]["opponent_spec"] == "beeline_blue"

    # 5) Verify num_timesteps progressed past the checkpoint.
    final = PPO.load(str(resume_trial / "final_model.zip"))
    assert final.num_timesteps > cold_steps, (
        f"final num_timesteps {final.num_timesteps} should exceed "
        f"checkpoint's {cold_steps}")
    assert final.num_timesteps >= 256

    # 6) Verify lr override took effect — SB3 stores lr as a callable schedule.
    lr_at_end = float(final.learning_rate(1.0)) if callable(final.learning_rate) \
                else float(final.learning_rate)
    assert lr_at_end == pytest.approx(1.5e-4, rel=1e-6), (
        f"expected lr=1.5e-4 from config to override checkpoint's 3e-4, got {lr_at_end}")
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
conda run --no-capture-output -n uav python -m pytest tests/integration/test_team_resume.py -v
```

Expected: Fails — the resume subprocess will exit non-zero because `train_team_ppo.py` doesn't yet handle `--resume`.

- [ ] **Step 3: Implement the resume branch in `main()`**

Open `scripts/train_team_ppo.py`. The current `main()` builds the env, then `ppo_kwargs`, then has a single `if args.warm_start: ... else: PPO(...)` block at the model-construction step. We're adding a third branch for resume.

**3a.** Add `Path` and `sys` imports if not already present, and add `from scripts.callbacks import ResumeProgressCallback` near the other imports (top of file). The team script does not currently import from `scripts.callbacks`; add this import line:

```python
from scripts.callbacks import ResumeProgressCallback
```

**3b.** Just after the existing `args = parse_args()` line and the missing-args guard (added in Task 2), add a block that resolves learner/opponent from the parent when resuming and they were not given on the CLI:

```python
    if args.resume is not None:
        if args.learner is None or args.opponent is None:
            parent_learner, parent_opponent = _read_parent_extra(args.resume)
            args.learner = args.learner or parent_learner
            args.opponent = args.opponent or parent_opponent
        if args.learner is None or args.opponent is None:
            print(
                f"ERROR: could not resolve --learner/--opponent from parent of {args.resume}. "
                "Pass them on the CLI.",
                file=sys.stderr,
            )
            sys.exit(2)
```

**3c.** Find the model-construction block. Currently (post-Task 1):

```python
    if args.warm_start:
        from core.policies.warm_start import warm_start_ppo
        ws_cfg = config.get("training", {}).get("team", {}).get("warm_start", {})
        model = warm_start_ppo(
            old_checkpoint=args.warm_start,
            new_env=vec_env,
            new_input_dim=22, old_input_dim=16,
            new_dim_init_scale=ws_cfg.get("new_dim_init_scale", 0.01),
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )
```

Replace it with:

```python
    resumed_at: int | None = None
    if args.resume is not None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️  Resuming from {args.resume}")
        # Override learning_rate from config so e.g. lr=1e-4 takes effect on
        # resume.  Other PPO hyperparameters keep their checkpoint values —
        # changing gamma/n_steps/etc. mid-training breaks invariants the value
        # function and rollout buffer rely on.
        model = PPO.load(
            args.resume,
            env=vec_env,
            tensorboard_log=str(run_dir),
            verbose=1,
            learning_rate=ppo_kwargs["learning_rate"],
        )
        resumed_at = int(model.num_timesteps)
        total_timesteps = args.timesteps or config["training"].get("total_timesteps", 5_000_000)
        remaining = max(total_timesteps - resumed_at, 0)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ⏱  Checkpoint at "
            f"{resumed_at:,} steps; {remaining:,} remaining to {total_timesteps:,}"
        )
    elif args.warm_start:
        from core.policies.warm_start import warm_start_ppo
        ws_cfg = config.get("training", {}).get("team", {}).get("warm_start", {})
        model = warm_start_ppo(
            old_checkpoint=args.warm_start,
            new_env=vec_env,
            new_input_dim=22, old_input_dim=16,
            new_dim_init_scale=ws_cfg.get("new_dim_init_scale", 0.01),
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )
```

**3d.** The `write_run_info` calls (there are two — one before training, one after) currently pass `extra={...}`. Add `resume=` to both. The `resume_info` dict is `{"checkpoint": args.resume, "resumed_at": resumed_at}` when `args.resume is not None` else `None`. Insert this just before the first `write_run_info` call:

```python
    resume_info = (
        {"checkpoint": args.resume, "resumed_at": resumed_at}
        if args.resume is not None else None
    )
```

Then in **both** `write_run_info(...)` calls, add `resume=resume_info,` next to the existing `extra=` argument. Both call sites must pass it; the second one (in `finally:`) already has the variable in scope.

**3e.** Update `model.learn(...)` to keep the step counter and bound the bar correctly:

The current call (line ~129 in the unmodified file):

```python
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=True)
```

Replace with:

```python
    extra_callbacks = (
        [ResumeProgressCallback(total_timesteps)]
        if args.resume is not None else []
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[*callbacks, *extra_callbacks],
            reset_num_timesteps=args.resume is None,
            progress_bar=args.resume is None,
        )
```

**3f.** `total_timesteps` is referenced both in the resume branch (step 3c) and in the `model.learn` call. To avoid double-resolving, hoist its definition up — find the existing line:

```python
    total_timesteps = args.timesteps or config["training"].get("total_timesteps", 5_000_000)
```

and **move it to before the model-construction block** (just after `ppo_kwargs` is built). Then the resume branch references the already-defined variable; remove the duplicate definition inside the resume branch.

- [ ] **Step 4: Run the integration test, verify it passes**

```bash
conda run --no-capture-output -n uav python -m pytest tests/integration/test_team_resume.py -v
```

Expected: Test passes. May take 60-120 s.

- [ ] **Step 5: Run the full test suite, verify no regressions**

```bash
conda run --no-capture-output -n uav python -m pytest -v
```

Expected: All tests pass. Pay attention to:
- `tests/integration/test_team_env_canary.py` — the deterministic step-176 / step-684 fingerprint MUST stay green (we changed no env code).
- `tests/integration/test_scoring_canary.py` — `step 434 / reward 7.3837` MUST stay green.

If either canary breaks: **stop**, revert your changes to `scripts/train_team_ppo.py` and `scripts/_train_common.py`, and report. The plan must not change env behavior.

- [ ] **Step 6: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/train_team_ppo.py tests/integration/test_team_resume.py
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(team-resume): wire --resume into main(), override lr from config"
```

---

## Task 6: Add `make resume-team` target

**Files:**
- Modify: `Makefile` (add new target + `.PHONY`)

- [ ] **Step 1: Add the target**

Open `Makefile`. Find the existing `.PHONY` line (around line 44). Append `resume-team` to its end.

Then find the `train-team-blue` target (around line 183) and after it (before `eval-team`) insert:

```makefile
resume-team: ## ▶️  Resume team-play training  RUN_NAME=... [TRIAL=...] [CHECKPOINT=...] [LEARNER=...] [OPPONENT=...]
	@ckpt="$(or $(CHECKPOINT),$(_LATEST_CKPT))"; \
	 test -n "$$ckpt" || { echo "ERROR: no checkpoint found in $(_TRIAL_DIR)/checkpoints/ — set RUN_NAME= and TRIAL="; exit 1; }; \
	 $(PYTHON) scripts/train_team_ppo.py --resume "$$ckpt" --run-name "$(_RESUME_RUN)" \
	   $(if $(LEARNER),--learner $(LEARNER)) \
	   $(if $(OPPONENT),--opponent "$(OPPONENT)")
```

`LEARNER` and `OPPONENT` are optional — when omitted, `train_team_ppo.py` reads them from the parent trial's `info.toml [extra]` (Task 4 helper).

- [ ] **Step 2: Smoke-test the target with a dry-run check**

```bash
make -n resume-team RUN_NAME=ppo_hoop_blue_1
```

Expected: prints the resolved command without errors. The command should reference the correct latest checkpoint under `runs/ppo_hoop_blue_1/<latest-trial>/checkpoints/`.

If `_LATEST_CKPT` is empty, the `make` call will print the ERROR branch — that's fine for a fresh project but indicates the user has no checkpoints yet.

- [ ] **Step 3: Verify `make help` shows the new target**

```bash
make help | grep resume-team
```

Expected: shows `resume-team` with its help string.

- [ ] **Step 4: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add Makefile
git -C "$(git rev-parse --show-toplevel)" commit -m "feat(make): add resume-team target mirroring resume"
```

---

## Task 7: Documentation pass

**Files:**
- Modify: `scripts/train_team_ppo.py` (top-of-file docstring)

- [ ] **Step 1: Update the module docstring**

Open `scripts/train_team_ppo.py`. Replace the existing docstring (lines 1-10) with:

```python
"""Training entry for QuidditchTeamEnv.  Wraps the team env in an
OpponentControlledEnv so SB3 PPO sees a single-agent Gym env.

Run modes:
    # Cold start
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue

    # Cold start, frozen-policy opponent
    python scripts/train_team_ppo.py --learner blue_0 \\
        --opponent frozen:models/.../best_model.zip

    # Warm-start from a single-agent checkpoint (16->22 input-layer surgery)
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue \\
        --warm-start models/ppo_hoop_rand_start_20260505_174509/best_model.zip

    # Resume an existing team-play trial from its latest checkpoint.  --learner
    # and --opponent are read from the parent trial's info.toml [extra] block
    # if not given on the CLI.  Config's lr overrides the checkpoint's lr.
    python scripts/train_team_ppo.py \\
        --resume runs/ppo_hoop_blue_1/20260507_194423/checkpoints/ppo_10000000_steps.zip
"""
```

- [ ] **Step 2: Print the ready-to-commit command**

```bash
git -C "$(git rev-parse --show-toplevel)" add scripts/train_team_ppo.py
git -C "$(git rev-parse --show-toplevel)" commit -m "docs(team-resume): document --resume in train_team_ppo module docstring"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run the full test suite from clean**

```bash
conda run --no-capture-output -n uav python -m pytest -v
```

Expected: All tests pass. Note the timing — full suite should be < 5 min.

- [ ] **Step 2: Smoke-test a real resume against the user's actual blue run**

This is the canonical end-to-end check. The user has `runs/ppo_hoop_blue_1/20260507_194423/` with checkpoints up to `ppo_10000000_steps.zip`. Test that the resume command parses and starts (kill it after ~30 s; we're not running 20M more steps in CI).

```bash
timeout 30 make resume-team RUN_NAME=ppo_hoop_blue_1 || true
```

Expected: the script:
1. Prints `▶️  Resuming from runs/ppo_hoop_blue_1/20260507_194423/checkpoints/ppo_10000000_steps.zip`
2. Prints `⏱  Checkpoint at 10,002,432 steps; <N> remaining to <total>`
3. Begins training (rollouts collected). The `timeout` will kill it cleanly.

After kill, verify a new trial dir exists under `runs/ppo_hoop_blue_1/`:

```bash
ls -la runs/ppo_hoop_blue_1/ | tail -3
```

Expected: a new `<YYYYMMDD_HHMMSS>` directory (today's date) sibling to the parent trial.

If the smoke test was successful, also confirm the new trial dir has `[resume]` in its `info.toml`:

```bash
grep -A2 '\[resume\]' runs/ppo_hoop_blue_1/<new-trial>/info.toml
```

Expected:

```
[resume]
checkpoint  = "runs/ppo_hoop_blue_1/20260507_194423/checkpoints/ppo_10000000_steps.zip"
resumed_at  = 10002432
```

- [ ] **Step 3: Clean up the smoke-test trial dir**

The smoke-test run is incomplete and shouldn't pollute the user's runs dir.

```bash
ls -dt runs/ppo_hoop_blue_1/*/ | head -1
# verify it's the smoke-test trial (today's date), then:
rm -rf runs/ppo_hoop_blue_1/<smoke-test-trial>/
```

- [ ] **Step 4: Report status to the user**

Summarize:
- All tests green (count from pytest output)
- `make resume-team` works against the real blue run
- Branch `feature/team-resume` is N commits ahead of `develop`
- Ready for merge: `git -C <repo> checkout develop && git -C <repo> merge --no-ff feature/team-resume`

The merge step is the user's call — do **not** run it yourself per the global CLAUDE.md GitFlow conventions and this user's hardware-key signing constraint.

---

## Self-Review (planner did this)

**1. Spec coverage.** Each spec section maps to a task:
- §1 New trial dir on resume → Task 5 (uses existing `make_run_dir`)
- §2 Lr-only override → Task 5 step 3c (`learning_rate=ppo_kwargs["learning_rate"]`)
- §3 Mutually exclusive `--resume` / `--warm-start` → Task 1
- §4 `total_timesteps` absolute → Task 5 step 3e (`reset_num_timesteps=False`)
- §5 `[resume]` block → Task 3
- §6 `resume-team` target → Task 6
- §7 ResumeProgressCallback → Task 5 step 3a + 3e
- Failure modes table → covered by tests in Tasks 1, 3, 5

**2. Placeholder scan.** No `TBD`, `TODO`, `implement later`, `add appropriate`, `similar to Task N`. All code is complete and copy-pastable.

**3. Type consistency.** `_read_parent_extra` returns `tuple[str | None, str | None]` everywhere it's referenced. `resume_info` has the shape `{"checkpoint": str, "resumed_at": int}` consistently across `write_run_info` (Task 3) and the call sites (Task 5). `args.resume` is `str | None` (default `None`); `args.warm_start` is `str` (default `""`) — these match the existing argparse types and are checked correctly with `is not None` vs truthiness.
