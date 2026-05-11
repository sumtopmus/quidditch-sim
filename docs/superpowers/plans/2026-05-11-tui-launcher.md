# TUI Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Textual TUI launcher (`make ui`) that replaces the demo-only stdin prompt with a full-screen dashboard: form-driven pickers (no more typing model paths), live training-progress widget reading a JSON status file written by an always-on SB3 callback, and a stripped-down Makefile.

**Architecture:** New `tui/` package + new `core/training/tui_progress_callback.py` + three Python ports of Make recipes (`scripts/promote.py`, `scripts/list_runs.py`, `scripts/repro.py`). The TUI never re-implements logic; it shells out to existing entry-point scripts. Disk state (promoted models, runs, trials, checkpoints) is read fresh by `tui/state/scan.py`. Training progress flows through `runs/<run>/<trial>/tui_progress.json`, atomically rewritten by `TUIProgressCallback` from inside SB3 training.

**Tech Stack:** Python 3.11, Textual (new pip dep), `rich` (already present), `stable_baselines3` (already present), pytest (already used).

**Working directory:** All commands and paths in this plan are relative to the worktree root: `worktrees/feature/tui-launcher/` under the umbrella `drone-sim/` project. The conda env is `uav`.

**Commit policy:** The user runs git commits manually (hardware key signing). Each task ends with a ready-to-paste commit command — the implementing agent assembles it but does not execute `git commit`. The agent may run `git status`/`git diff`/`git add` for staging.

---

## File Structure

```
worktrees/feature/tui-launcher/
├── core/
│   └── training/
│       ├── __init__.py                # new
│       └── tui_progress_callback.py   # new — SB3 BaseCallback → JSON
├── demo/
│   └── menu.py                        # modify — add positional CLI arg
├── scripts/
│   ├── _train_common.py               # modify — wire callback in build_callbacks
│   ├── train_ppo.py                   # modify — append callback inline
│   ├── train_team_ppo.py              # modify — pass extra args to build_callbacks
│   ├── promote.py                     # new — port of Make 'promote' recipe
│   ├── list_runs.py                   # new — port of Make 'list-runs' recipe
│   └── repro.py                       # new — port of Make 'repro' recipe
├── tui/
│   ├── __init__.py
│   ├── __main__.py                    # python -m tui entrypoint
│   ├── app.py                         # DroneSimApp (Textual App)
│   ├── app.tcss                       # CSS theme
│   ├── theme.py                       # Glyphs + ASCII fallback
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── base.py                    # Action, FieldSpec dataclasses
│   │   ├── demos.py | train.py | eval.py | manage.py
│   │   └── registry.py                # ACTIONS list
│   ├── widgets/
│   │   ├── action_tree.py
│   │   ├── action_form.py
│   │   ├── state_pane.py
│   │   ├── training_widget.py
│   │   └── log_overlay.py
│   ├── process/
│   │   ├── manager.py                 # subprocess slots
│   │   └── progress.py                # tails tui_progress.json
│   └── state/
│       └── scan.py                    # disk scanner
├── tests/unit/
│   ├── test_tui_actions.py            # new
│   ├── test_tui_scan.py               # new
│   ├── test_tui_progress_callback.py  # new
│   ├── test_tui_progress_tail.py      # new
│   ├── test_promote_script.py         # new
│   ├── test_list_runs_script.py       # new
│   └── test_repro_script.py           # new
├── Makefile                           # modify — strip 15 targets, add ui + demo
├── README.md                          # modify — one-line mention
├── CLAUDE.md                          # modify — one-line mention
└── requirements.txt                   # modify — add textual
```

---

## Task 1: Add `textual` dependency and verify install

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add textual to requirements.txt**

Open `requirements.txt`. After the `rich` line (in the "Progress bar" block), add:

```
# Terminal UI framework for `make ui` launcher (pulls in rich, already pinned).
textual>=0.79
```

- [ ] **Step 2: Install the new dep into the conda env**

Run: `conda run --no-capture-output -n uav pip install -r requirements.txt`
Expected: `textual` is installed (no failures); other deps remain pinned.

- [ ] **Step 3: Smoke-test the import**

Run: `conda run --no-capture-output -n uav python -c "import textual; print(textual.__version__)"`
Expected: prints a version number ≥ 0.79.

- [ ] **Step 4: Stage and commit (user runs)**

```bash
git -C "$PWD" add requirements.txt
git -C "$PWD" commit -m "$(cat <<'EOF'
deps: add textual for TUI launcher

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extract `scripts/promote.py` from the Make recipe

**Why first:** the Makefile strip later removes the inline `promote` recipe. Porting it now means the strip is a clean delete with no behavior loss.

**Files:**
- Create: `scripts/promote.py`
- Create: `tests/unit/test_promote_script.py`

The current Make recipe (Makefile:103-122) copies three files (`best_model.zip`, `info.toml`, `config_snapshot.toml`) from a trial dir to `models/<flat-name>/`, where the flat-name is the trial-relative path with `/` replaced by `_`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_promote_script.py`:

```python
"""Unit tests for scripts/promote.py — file-copy semantics match the
former Make `promote` recipe (Makefile lines 103-122 pre-strip)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _make_trial_dir(tmp_path: Path, *, run: str, trial: str, with_info: bool = True,
                    with_config: bool = True) -> Path:
    trial_dir = tmp_path / "runs" / run / trial
    trial_dir.mkdir(parents=True)
    (trial_dir / "best_model.zip").write_bytes(b"fake-zip-data")
    if with_info:
        (trial_dir / "info.toml").write_text('name = "x"\n')
    if with_config:
        (trial_dir / "config_snapshot.toml").write_text('[training]\n')
    return trial_dir


def _run_promote(trial_dir: Path, models_root: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "promote.py"),
         "--trial", str(trial_dir), "--models-root", str(models_root)],
        capture_output=True, text=True,
    )


def test_promote_copies_three_files_with_flat_name(tmp_path: Path) -> None:
    trial = _make_trial_dir(tmp_path, run="ppo_hoop_red_1", trial="20260506_103058")
    models_root = tmp_path / "models"
    result = _run_promote(trial, models_root)
    assert result.returncode == 0, result.stderr
    dest = models_root / "ppo_hoop_red_1_20260506_103058"
    assert (dest / "best_model.zip").read_bytes() == b"fake-zip-data"
    assert (dest / "run_info.toml").exists()           # renamed from info.toml
    assert (dest / "config.toml").exists()             # renamed from config_snapshot.toml


def test_promote_missing_best_model_errors(tmp_path: Path) -> None:
    trial_dir = tmp_path / "runs" / "x" / "y"
    trial_dir.mkdir(parents=True)
    # no best_model.zip
    result = _run_promote(trial_dir, tmp_path / "models")
    assert result.returncode != 0
    assert "best_model.zip" in result.stderr or "best_model.zip" in result.stdout


def test_promote_missing_optional_files_succeeds(tmp_path: Path) -> None:
    trial = _make_trial_dir(tmp_path, run="r", trial="t",
                            with_info=False, with_config=False)
    models_root = tmp_path / "models"
    result = _run_promote(trial, models_root)
    assert result.returncode == 0, result.stderr
    dest = models_root / "r_t"
    assert (dest / "best_model.zip").exists()
    assert not (dest / "run_info.toml").exists()
    assert not (dest / "config.toml").exists()
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_promote_script.py -v`
Expected: FAIL — `FileNotFoundError` because `scripts/promote.py` doesn't exist yet.

- [ ] **Step 3: Implement scripts/promote.py**

Create `scripts/promote.py`:

```python
"""Promote a trial's best_model.zip (+ optional metadata) into models/<flat-name>/.

Port of the former `make promote` recipe. Callable from the TUI's Manage →
promote action and from the shell for one-offs.

Usage:
    python scripts/promote.py --trial runs/<run>/<trial>  [--models-root models]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def promote(trial_dir: Path, models_root: Path) -> Path:
    if not trial_dir.is_dir():
        raise SystemExit(f"ERROR: trial directory not found: {trial_dir}")
    src_best = trial_dir / "best_model.zip"
    if not src_best.is_file():
        raise SystemExit(
            f"ERROR: {src_best} not found — has training produced a best_model.zip yet?"
        )

    # Flat name: strip the "runs/" prefix if present, then replace path separators
    # with underscores so "ppo_hoop_red_1/20260506_103058" → "ppo_hoop_red_1_20260506_103058".
    parts = trial_dir.parts
    try:
        runs_ix = parts.index("runs")
        rel_parts = parts[runs_ix + 1:]
    except ValueError:
        rel_parts = parts[-2:]  # fall back to last two segments
    flat_name = "_".join(rel_parts)

    dest = models_root / flat_name
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_best, dest / "best_model.zip")
    info_src = trial_dir / "info.toml"
    if info_src.is_file():
        shutil.copy2(info_src, dest / "run_info.toml")
    config_src = trial_dir / "config_snapshot.toml"
    if config_src.is_file():
        shutil.copy2(config_src, dest / "config.toml")

    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Promote a trial's best model into models/.")
    p.add_argument("--trial", required=True, type=Path,
                   help="Path to the trial directory (e.g. runs/ppo_hoop/20260506_103058)")
    p.add_argument("--models-root", default=Path("models"), type=Path,
                   help="Models directory (default: models)")
    args = p.parse_args(argv)

    dest = promote(args.trial, args.models_root)
    print()
    print(f"  Trial:    {args.trial}")
    print(f"  Promoted  →  {dest}/")
    print()
    print("  To commit:")
    print(f"    git add {dest}")
    print(f"    git commit -m 'model: promote {dest.name} best model'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_promote_script.py -v`
Expected: 3 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add scripts/promote.py tests/unit/test_promote_script.py
git -C "$PWD" commit -m "$(cat <<'EOF'
refactor: extract promote logic to scripts/promote.py

Port of the former make 'promote' recipe. Same file-copy semantics
(best_model.zip + optional info.toml + config_snapshot.toml → models/<flat-name>/).
Prepares for the Makefile strip when the TUI lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extract `scripts/list_runs.py` from the Make recipe

The current Make recipe (Makefile:153-169) walks `runs/` and `models/` and prints a flat tree.

**Files:**
- Create: `scripts/list_runs.py`
- Create: `tests/unit/test_list_runs_script.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_list_runs_script.py`:

```python
"""Unit tests for scripts/list_runs.py — output enumerates runs and models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(runs_root: Path, models_root: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "list_runs.py"),
         "--runs-root", str(runs_root), "--models-root", str(models_root)],
        capture_output=True, text=True,
    )


def test_list_runs_groups_by_config(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "ppo_hoop" / "20260101_000000").mkdir(parents=True)
    (runs / "ppo_hoop" / "20260202_000000").mkdir(parents=True)
    (runs / "ppo_hoop_team" / "20260303_000000").mkdir(parents=True)
    models = tmp_path / "models"
    (models / "red_v1").mkdir(parents=True)
    (models / "red_v1" / "best_model.zip").write_bytes(b"x")

    result = _run(runs, models)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "ppo_hoop/" in out
    assert "20260101_000000" in out
    assert "ppo_hoop_team/" in out
    assert "red_v1" in out
    assert "best_model.zip" in out


def test_list_runs_no_runs_no_models(tmp_path: Path) -> None:
    result = _run(tmp_path / "runs", tmp_path / "models")
    assert result.returncode == 0
    out = result.stdout
    assert "(none)" in out  # at least one "(none)" for missing runs and missing models
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_list_runs_script.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement scripts/list_runs.py**

Create `scripts/list_runs.py`:

```python
"""List training runs and promoted models — port of the former `make list-runs` recipe.

Output format mirrors the original shell version:

  === runs/ ===
    <config_name>/
      <trial_name>
      <trial_name>
  === models/ ===
    <model_dir>/
      <file>
      <file>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def list_runs(runs_root: Path, models_root: Path) -> int:
    print(f"=== {runs_root}/ ===")
    if not runs_root.is_dir() or not any(runs_root.iterdir()):
        print("  (none)")
    else:
        for cfg in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            print(f"  {cfg.name}/")
            trials = sorted(
                (t for t in cfg.iterdir() if t.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
            for t in trials:
                print(f"    {t.name}")

    print()
    print(f"=== {models_root}/ ===")
    if not models_root.is_dir() or not any(p for p in models_root.iterdir() if p.is_dir()):
        print("  (none — run 'make promote' after a successful training run)")
    else:
        for m in sorted(p for p in models_root.iterdir() if p.is_dir()):
            print(f"  {m}/")
            for child in sorted(m.iterdir()):
                print(f"      {child.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List training runs and promoted models.")
    p.add_argument("--runs-root", default=Path("runs"), type=Path)
    p.add_argument("--models-root", default=Path("models"), type=Path)
    args = p.parse_args(argv)
    return list_runs(args.runs_root, args.models_root)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_list_runs_script.py -v`
Expected: 2 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add scripts/list_runs.py tests/unit/test_list_runs_script.py
git -C "$PWD" commit -m "$(cat <<'EOF'
refactor: extract list-runs logic to scripts/list_runs.py

Port of the former make 'list-runs' recipe. Walks runs/ and models/
and prints a flat tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extract `scripts/repro.py` from the Make recipe

Current Make recipe (Makefile:124-133) copies `models/<MODEL>/config.toml` to `config/training.toml`, with an "older format" warning if `env.toml` exists.

**Files:**
- Create: `scripts/repro.py`
- Create: `tests/unit/test_repro_script.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_repro_script.py`:

```python
"""Unit tests for scripts/repro.py — config restoration semantics."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(model_dir: Path, config_target: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "repro.py"),
         "--model-dir", str(model_dir),
         "--config-target", str(config_target)],
        capture_output=True, text=True,
    )


def test_repro_copies_config_to_target(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "red_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text('[training]\nlr = 5e-5\n')
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode == 0, result.stderr
    assert target.read_text() == '[training]\nlr = 5e-5\n'


def test_repro_warns_on_legacy_env_toml(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "old_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text('[training]\n')
    (model_dir / "env.toml").write_text('[env]\n')
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode == 0
    assert "env.toml" in (result.stdout + result.stderr)


def test_repro_missing_config_errors(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "broken"
    model_dir.mkdir(parents=True)
    target = tmp_path / "config" / "training.toml"

    result = _run(model_dir, target)
    assert result.returncode != 0
    assert "config.toml" in (result.stderr + result.stdout)
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_repro_script.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement scripts/repro.py**

Create `scripts/repro.py`:

```python
"""Restore config/training.toml from a promoted model's config.toml.

Port of the former `make repro MODEL=<name>` recipe.

Usage:
    python scripts/repro.py --model-dir models/<name>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def repro(model_dir: Path, config_target: Path) -> int:
    src = model_dir / "config.toml"
    if not src.is_file():
        raise SystemExit(
            f"ERROR: {src} not found — model promoted before config snapshots were added?"
        )
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, config_target)
    print(f"Restored {config_target} from {src}")

    env_src = model_dir / "env.toml"
    if env_src.is_file():
        print(
            f"NOTE: {env_src} is from an older promote format; its [env] section is "
            "now part of config/training.toml — verify the values match."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Restore config/training.toml from a promoted model.")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Path to the promoted model directory (e.g. models/red_v1)")
    p.add_argument("--config-target", default=Path("config/training.toml"), type=Path)
    args = p.parse_args(argv)
    return repro(args.model_dir, args.config_target)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_repro_script.py -v`
Expected: 3 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add scripts/repro.py tests/unit/test_repro_script.py
git -C "$PWD" commit -m "$(cat <<'EOF'
refactor: extract repro logic to scripts/repro.py

Port of the former make 'repro' recipe. Copies a promoted model's
config.toml back to config/training.toml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add a positional CLI arg to `demo/menu.py`

Currently `demo/menu.py` always prompts via stdin. We want `mjpython demo/menu.py hover` to skip the prompt and run that demo directly, so the TUI can shell out non-interactively. The interactive path stays for anyone running `mjpython demo/menu.py` with no arg.

**Files:**
- Modify: `demo/menu.py`

- [ ] **Step 1: Read the existing file**

Read `demo/menu.py` (66 lines). The relevant function is `main()` at line 52.

- [ ] **Step 2: Modify main() to accept a key**

Replace `demo/menu.py`'s `main()` function (lines 52-62) with:

```python
def main(argv: list[str] | None = None) -> None:
    import sys as _sys
    keys = [k for k, _, _ in DEMOS]

    raw_arg = (argv if argv is not None else _sys.argv[1:])
    if raw_arg:
        chosen = raw_arg[0].strip().lower()
        try:
            idx = keys.index(chosen)
        except ValueError:
            print(f"unknown demo {chosen!r}; choices: {', '.join(keys)}", file=_sys.stderr)
            _sys.exit(2)
    else:
        i = _prompt()
        if i is None:
            print("No demo selected.")
            return
        idx = i

    key, _, module_path = DEMOS[idx]
    print(f"\n>>> running '{key}' ({module_path})\n")
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise RuntimeError(f"demo module {module_path!r} has no main()")
    module.main()
```

- [ ] **Step 3: Quick smoke test (interactive path still works)**

Run: `echo q | conda run --no-capture-output -n uav python demo/menu.py`
Expected: prints the menu, exits cleanly with "No demo selected."

- [ ] **Step 4: Quick smoke test (CLI-arg path)**

Run: `conda run --no-capture-output -n uav python demo/menu.py bogus`
Expected: exit code 2, stderr says "unknown demo 'bogus'; choices: hover, waypoint, takedown, score-through-tag".

Run (this opens MuJoCo viewer — close it manually):
`conda run --no-capture-output -n uav mjpython demo/menu.py hover`
Expected: hovers a drone in the arena viewer. Close to exit.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add demo/menu.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(demo): accept demo key as positional CLI arg

Allows 'mjpython demo/menu.py hover' to skip the stdin prompt.
The interactive path is unchanged when no arg is given. Used by
the upcoming TUI launcher to dispatch demos non-interactively.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `TUIProgressCallback` (the JSON-writing SB3 callback)

**Files:**
- Create: `core/training/__init__.py` (empty)
- Create: `core/training/tui_progress_callback.py`
- Create: `tests/unit/test_tui_progress_callback.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tui_progress_callback.py`:

```python
"""Unit tests for core/training/tui_progress_callback.py — JSON schema, atomicity."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.training.tui_progress_callback import TUIProgressCallback


def _fake_model(num_timesteps: int, ep_rewards: list[float]) -> MagicMock:
    m = MagicMock()
    m.num_timesteps = num_timesteps
    # SB3 stores recent episodes here; each entry is a dict with 'r' (return) and 'l' (length)
    m.ep_info_buffer = [{"r": r, "l": 100.0} for r in ep_rewards]
    m.start_time = time.time_ns() - 60_000_000_000  # 60 seconds ago, ns precision
    return m


def test_callback_writes_schema_v1_json(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=5000, ep_rewards=[1.0, 1.2, 1.5, 1.8])
    cb._on_step()

    p = tmp_path / "tui_progress.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["schema_version"] == 1
    assert data["step"] == 5000
    assert data["total_steps"] == 1_000_000
    assert data["kind"] == "single"
    assert data["learner"] is None
    assert data["opponent"] is None
    assert data["ep_rew_mean"] == pytest.approx((1.0 + 1.2 + 1.5 + 1.8) / 4)
    assert isinstance(data["recent_rewards"], list)
    assert data["elapsed_sec"] >= 59  # ~60s


def test_callback_team_kind_records_learner_and_opponent(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=10_000_000,
        kind="team",
        learner="blue_0",
        opponent_spec="frozen:models/red_v1/best_model",
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=12345, ep_rewards=[2.0, 2.1])
    cb._on_step()

    data = json.loads((tmp_path / "tui_progress.json").read_text())
    assert data["kind"] == "team"
    assert data["learner"] == "blue_0"
    assert data["opponent"] == "frozen:models/red_v1/best_model"


def test_callback_throttles_writes_by_write_every(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1000,
    )
    p = tmp_path / "tui_progress.json"

    cb.model = _fake_model(num_timesteps=500, ep_rewards=[1.0])
    cb._on_step()
    assert not p.exists(), "should not write before write_every threshold"

    cb.model = _fake_model(num_timesteps=1000, ep_rewards=[1.0])
    cb._on_step()
    assert p.exists(), "should write at multiples of write_every"


def test_callback_atomic_write_via_temp_rename(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    cb.model = _fake_model(num_timesteps=100, ep_rewards=[1.0])
    cb._on_step()
    # No .tmp left behind after a successful write.
    assert not (tmp_path / "tui_progress.json.tmp").exists()
    assert (tmp_path / "tui_progress.json").exists()


def test_callback_recent_rewards_bounded_to_16(tmp_path: Path) -> None:
    cb = TUIProgressCallback(
        run_dir=tmp_path,
        total_timesteps=1_000_000,
        kind="single",
        learner=None,
        opponent_spec=None,
        write_every=1,
    )
    # Feed 25 distinct ep_rew_mean values by repeatedly stepping.
    for i in range(25):
        cb.model = _fake_model(num_timesteps=(i + 1) * 100,
                               ep_rewards=[float(i)])
        cb._on_step()

    data = json.loads((tmp_path / "tui_progress.json").read_text())
    assert len(data["recent_rewards"]) == 16
    # last value is the most recent ep_rew_mean (24.0 since the buffer has one ep with r=24)
    assert data["recent_rewards"][-1] == pytest.approx(24.0)
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_progress_callback.py -v`
Expected: FAIL — `ModuleNotFoundError: core.training`.

- [ ] **Step 3: Create the module**

Create `core/training/__init__.py` (empty file).

Create `core/training/tui_progress_callback.py`:

```python
"""SB3 callback that writes a structured JSON progress file consumed by the TUI launcher.

The file is written atomically (tmp + os.replace) every `write_every` env steps.
Schema is documented in docs/superpowers/specs/2026-05-11-tui-launcher-design.md
section 3.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Literal

from stable_baselines3.common.callbacks import BaseCallback

SCHEMA_VERSION = 1
_RECENT_LEN = 16


class TUIProgressCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path | str,
        total_timesteps: int,
        kind: Literal["single", "team"],
        learner: str | None,
        opponent_spec: str | None,
        write_every: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._run_dir = Path(run_dir)
        self._total = int(total_timesteps)
        self._kind = kind
        self._learner = learner
        self._opponent_spec = opponent_spec
        self._write_every = (
            write_every if write_every is not None
            else max(1, self._total // 500)
        )
        self._recent: deque[float] = deque(maxlen=_RECENT_LEN)
        self._best_reward: float | None = None
        self._best_step: int = 0
        self._target = self._run_dir / "tui_progress.json"
        self._tmp = self._run_dir / "tui_progress.json.tmp"

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step % self._write_every != 0:
            return True
        self._write_snapshot(step)
        return True

    def _write_snapshot(self, step: int) -> None:
        ep_rew_mean = _safe_mean(self.model.ep_info_buffer, "r")
        ep_len_mean = _safe_mean(self.model.ep_info_buffer, "l")
        if ep_rew_mean is not None:
            self._recent.append(ep_rew_mean)
            if self._best_reward is None or ep_rew_mean > self._best_reward:
                self._best_reward = ep_rew_mean
                self._best_step = step

        elapsed_sec = max(time.time() - (self.model.start_time / 1e9), 0.0) \
            if isinstance(self.model.start_time, int) else 0.0
        fps = (step / elapsed_sec) if elapsed_sec > 0 else 0.0

        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "ts": time.time(),
            "run_name": self._run_dir.parent.name if self._run_dir.parent else "",
            "trial": self._run_dir.name,
            "kind": self._kind,
            "learner": self._learner,
            "opponent": self._opponent_spec,
            "step": step,
            "total_steps": self._total,
            "fps": fps,
            "elapsed_sec": elapsed_sec,
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "best_so_far": (
                {"reward": self._best_reward, "step": self._best_step}
                if self._best_reward is not None else None
            ),
            "recent_rewards": list(self._recent),
        }
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._tmp.write_text(json.dumps(snapshot))
        os.replace(self._tmp, self._target)


def _safe_mean(buf, key: str) -> float | None:
    if not buf:
        return None
    vals = [e[key] for e in buf if key in e]
    if not vals:
        return None
    return sum(vals) / len(vals)
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_progress_callback.py -v`
Expected: 5 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add core/training/__init__.py core/training/tui_progress_callback.py tests/unit/test_tui_progress_callback.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(training): TUIProgressCallback — atomic JSON status file

Writes runs/<run>/<trial>/tui_progress.json every ~total/500 steps
(atomic via tmp + os.replace) so the upcoming TUI launcher can render
a live progress widget without parsing SB3 log tables. Schema v1
documented in the TUI design spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extend `_train_common.build_callbacks` with the TUI callback; thread kwargs from the team script

Add `TUIProgressCallback` to the callback list returned by `build_callbacks`, gated on new optional kwargs. The team script (`scripts/train_team_ppo.py`) passes those kwargs immediately; the single-agent script's refactor (Task 8) adopts the same path so the wire-up lives in one place forever.

**Files:**
- Modify: `scripts/_train_common.py`
- Modify: `scripts/train_team_ppo.py`

- [ ] **Step 1: Read both files to confirm current state**

Read `scripts/_train_common.py` lines 121-220 (the `build_callbacks` function — confirm the signature and the `cbs.append(...)` block where new callbacks are added).

Read `scripts/train_team_ppo.py` lines 225-250 (the `callbacks = build_callbacks(...)` call site).

- [ ] **Step 2: Extend `build_callbacks` signature in `_train_common.py`**

Open `scripts/_train_common.py`. In `build_callbacks` (starts ~line 121), add four kwargs to the signature after `frame_stack: int = 1,`:

```python
def build_callbacks(
    *,
    run_dir: Path,
    eval_env_fn: Callable[[], Any],
    config: dict[str, Any],
    n_envs: int,
    video_env_fn: Callable[[], Any] | None = None,
    verbose: int = 0,
    frame_stack: int = 1,
    total_timesteps: int | None = None,
    kind: Literal["single", "team"] = "team",
    learner: str | None = None,
    opponent_spec: str | None = None,
) -> list:
```

Add the `Literal` import at the top of the file (next to existing typing imports):

```python
from typing import Any, Callable, Literal
```

At the end of `build_callbacks`, just before `return cbs`, append:

```python
    if total_timesteps is not None:
        from core.training.tui_progress_callback import TUIProgressCallback
        cbs.append(
            TUIProgressCallback(
                run_dir=run_dir,
                total_timesteps=total_timesteps,
                kind=kind,
                learner=learner,
                opponent_spec=opponent_spec,
            )
        )
```

The `total_timesteps is not None` guard means existing callers (currently only the team script — once Task 8 updates the single-agent path, both will pass it) keep working unchanged if for any reason they don't pass it.

- [ ] **Step 3: Pass the new kwargs from `scripts/train_team_ppo.py`**

Open `scripts/train_team_ppo.py`. Find the `callbacks = build_callbacks(...)` call (around line 232). Append the four kwargs:

```python
    callbacks = build_callbacks(
        run_dir=trial_dir,
        eval_env_fn=eval_env_fn,
        config=cfg,
        n_envs=args.n_envs,
        video_env_fn=video_env_fn,
        verbose=args.verbose,
        frame_stack=cfg["training"]["frame_stack"],
        total_timesteps=args.timesteps,
        kind="team",
        learner=args.learner,
        opponent_spec=args.opponent,
    )
```

(Match the existing arg names — the kwargs you add must come after the existing ones. The exact existing call site may have slightly different formatting; preserve it.)

- [ ] **Step 4: Run the unit tests + existing team canaries to confirm no regression**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_progress_callback.py tests/unit/test_team_mjcf.py tests/unit/test_tag_state_machine.py -v`
Expected: all pass (the callback test still works; the team unit tests are unaffected).

Run: `conda run --no-capture-output -n uav python -m pytest tests/integration/test_team_env_canary.py -v`
Expected: passes — the team env canary is unchanged.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add scripts/_train_common.py scripts/train_team_ppo.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(training): wire TUIProgressCallback into team training

build_callbacks now optionally appends TUIProgressCallback when
total_timesteps + kind/learner/opponent_spec are supplied. The team
script passes them; the single-agent inline path is wired in the next
commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Refactor `scripts/train_ppo.py` to use `_train_common.build_callbacks`

The single-agent script currently builds checkpoint + eval + video callbacks inline (lines ~322-366). This duplicates what `build_callbacks` already does for the team script, and forces a second wire-up site for any new callback (like the TUI one). Switch `train_ppo.py` to call `build_callbacks` and drop the inline construction. The `TUIProgressCallback` now lives in `build_callbacks` only — no second wire-up site.

**Behavior change worth flagging:** the unified `name_prefix` is `"ppo"`, so new single-agent training runs produce `ppo_<step>_steps.zip` instead of `ppo_hoop_<step>_steps.zip`. Existing on-disk checkpoints keep their old names; the resume / scan tooling handles both prefixes (Task 11 makes the regex prefix-agnostic).

The `ResumeProgressCallback` (resume-only, non-verbose-only) stays inline at the `model.learn(...)` call site — it is single-agent-specific and not part of the shared callback set.

**Files:**
- Modify: `scripts/train_ppo.py`

- [ ] **Step 1: Read the inline callback section + the `model.learn(...)` call**

Read `scripts/train_ppo.py` lines 280-330 (env construction), 322-366 (inline callbacks), 459-470 (extra_callbacks + model.learn).

- [ ] **Step 2: Replace the inline callback construction with a `build_callbacks` call**

In `scripts/train_ppo.py`:

(2a) At the top of the file, alongside the existing `from callbacks import ...` (line 58), add:

```python
from scripts._train_common import build_callbacks
```

Remove the now-unused imports of `CheckpointCallback` and `EvalCallback` from the `stable_baselines3.common.callbacks` import block (lines 51-55) — `build_callbacks` constructs them internally. Keep `VideoRecorderCallback` and `ResumeProgressCallback` imports (they're still referenced — `ResumeProgressCallback` directly, `VideoRecorderCallback` only by `build_callbacks`, but the local import in `build_callbacks` is fine; you can drop the top-of-file import of `VideoRecorderCallback`).

(2b) Replace the entire callback construction block (`scripts/train_ppo.py` lines ~322-366 — the `# ---- callbacks ----` section through the closing `)` of `video_cb`) with:

```python
    # ---- callbacks ----
    # Shared with team training via scripts._train_common.build_callbacks:
    # checkpoint + eval + video + TUI progress, all built consistently.
    callbacks = build_callbacks(
        run_dir=Path(trial_dir),
        eval_env_fn=lambda: QuidditchSimpleEnv(render_mode=None, **base_env_kwargs),
        config=cfg,
        n_envs=args.n_envs,
        video_env_fn=lambda: QuidditchSimpleEnv(
            render_mode="rgb_array", **base_env_kwargs
        ),
        verbose=verbose,
        total_timesteps=args.timesteps,
        kind="single",
        learner=None,
        opponent_spec=None,
    )
```

(2c) Remove the now-redundant `eval_env = make_vec_env(...)` block at lines ~315-320 — `build_callbacks` constructs its own eval env from `eval_env_fn`. Also remove the now-orphaned `from stable_baselines3.common.env_util import make_vec_env` import IF it is no longer referenced anywhere else in the file. Check via grep before deletion:

Run: `grep -n make_vec_env scripts/train_ppo.py`
Expected: either no remaining references (delete the import) or one remaining reference for `train_env` (keep the import).

(2d) Update the `model.learn(...)` call (lines ~464-469) to use the new `callbacks` variable. The original was:

```python
    extra_callbacks = (
        []
        if (args.verbose or args.resume is None)
        else [ResumeProgressCallback(args.timesteps)]
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, video_cb, *extra_callbacks],
        reset_num_timesteps=args.resume is None,
        progress_bar=not args.verbose and args.resume is None,
    )
```

Replace with:

```python
    extra_callbacks = (
        []
        if (args.verbose or args.resume is None)
        else [ResumeProgressCallback(args.timesteps)]
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks + extra_callbacks,
        reset_num_timesteps=args.resume is None,
        progress_bar=not args.verbose and args.resume is None,
    )
```

- [ ] **Step 3: Run the single-agent scoring canary — load-bearing check**

This is the most important test in the refactor. The canary trains a fixed seed run until the first scoring event and asserts `SCORED at step 434 / total reward 7.3837`. If the refactor introduces ANY behavioral drift (RNG, env reset, env wrapper composition), this asserts will fail.

Run: `conda run --no-capture-output -n uav python -m pytest tests/integration/test_scoring_canary.py -v`
Expected: PASS — exact assertion `SCORED at step 434 / total reward 7.3837` holds.

If it fails: do NOT proceed. The most likely cause is a subtle difference in eval-env seeding or wrapper order. Diff the eval env construction between the old inline path and `build_callbacks` — the inline path used `make_vec_env(..., seed=seed)` which calls `env.seed/action_space.seed`; `build_callbacks` does not. If the canary is sensitive to that (it should not be — the eval env runs in its own context, not the training rollout), restore the seed call by adding `eval_env.seed(seed)` after the `build_callbacks` returns, or push the seed inside `eval_env_fn` via a closure.

- [ ] **Step 4: Run the full test suite to catch anything else**

Run: `conda run --no-capture-output -n uav python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add scripts/train_ppo.py
git -C "$PWD" commit -m "$(cat <<'EOF'
refactor(train_ppo): use _train_common.build_callbacks

Replaces the inline checkpoint+eval+video construction with a call
to the shared build_callbacks, matching the team script. The
TUIProgressCallback is wired in only one place now. Scoring canary
(step 434 / 7.3837) unchanged.

Note: unified checkpoint name_prefix is "ppo" (was "ppo_hoop" for
single-agent only). New single-agent runs produce ppo_<step>_steps.zip;
existing on-disk checkpoints keep their old prefix and remain
resumable — the scan/regex tooling is prefix-agnostic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Manual smoke test of the JSON file from a brief training run

Not a code change — a confidence check that the JSON file appears with the right shape before we build a TUI that depends on it.

**No files. Manual verification step.**

- [ ] **Step 1: Kick off a short training run**

Run: `conda run --no-capture-output -n uav python scripts/train_ppo.py --run-name tui_smoke --timesteps 5000`
Wait ~30 seconds; ctrl-C if it hasn't ended yet.

- [ ] **Step 2: Confirm the JSON file appeared with valid contents**

Run: `find runs/tui_smoke -name tui_progress.json -exec cat {} \;`
Expected: a JSON object with `schema_version: 1`, `step` > 0, `total_steps: 5000`, `kind: "single"`, etc.

- [ ] **Step 3: Clean up the smoke run**

Run: `rm -rf runs/tui_smoke`

**No commit for this task.** If the file did not appear or was malformed, halt and fix the wire-up before proceeding.

---

## Task 10: Create the `tui/` package skeleton + `theme.py`

**Files:**
- Create: `tui/__init__.py` (empty)
- Create: `tui/__main__.py` (placeholder; fleshed out in Task 22)
- Create: `tui/theme.py`

- [ ] **Step 1: Create `tui/__init__.py`**

Create `tui/__init__.py`:

```python
"""Textual TUI launcher for the drone-quidditch project.

Run with: ``python -m tui`` or ``make ui``.
"""
```

- [ ] **Step 2: Create `tui/__main__.py` (placeholder)**

Create `tui/__main__.py`:

```python
"""Entrypoint for `python -m tui`.

The full implementation lands in Task 22. Until then, this is a stub so
the rest of the package can import cleanly and ``python -m tui`` exits 0.
"""
from __future__ import annotations


def main() -> int:
    print("tui: not yet implemented (see Task 22 in the implementation plan)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Create `tui/theme.py`**

Create `tui/theme.py`:

```python
"""Visual constants for the TUI: Nerd Font glyphs + ASCII fallback.

Nerd Font codepoints used (private-use area U+E000+):
- nf-fa-gamepad           : demos
- nf-fa-rocket            : single-agent train
- nf-fa-flag              : team train (red / blue)
- nf-fa-refresh           : resume / repro
- nf-fa-crosshairs        : eval
- nf-fa-bar_chart         : tensorboard
- nf-fa-trophy            : promote
- nf-fa-folder_open       : list-runs
- nf-fa-link              : lineage
- nf-fa-lock              : disabled action

These glyphs render correctly only with a Nerd Font installed. When
``Glyphs.ASCII`` is active (selected via the ``--ascii`` CLI flag) the
class attributes are replaced with bracket-letter forms.
"""
from __future__ import annotations


class Glyphs:
    DEMO        = ""
    TRAIN       = ""
    TRAIN_TEAM  = ""
    RESUME      = ""
    EVAL        = ""
    TENSORBOARD = ""
    PROMOTE     = ""
    LIST_RUNS   = ""
    LINEAGE     = ""
    REPRO       = ""
    LOCK        = ""
    LIVE        = "●"
    RUNNING     = "⏵"
    UP_ARROW    = "↑"
    DOWN_ARROW  = "↓"


class GlyphsASCII:
    DEMO        = "[D]"
    TRAIN       = "[T]"
    TRAIN_TEAM  = "[Tm]"
    RESUME      = "[R]"
    EVAL        = "[E]"
    TENSORBOARD = "[B]"
    PROMOTE     = "[P]"
    LIST_RUNS   = "[L]"
    LINEAGE     = "[Ln]"
    REPRO       = "[Rp]"
    LOCK        = "[x]"
    LIVE        = "*"
    RUNNING     = ">"
    UP_ARROW    = "^"
    DOWN_ARROW  = "v"


# Default — overridden in tui/app.py based on --ascii flag.
ACTIVE: type = Glyphs


def use_ascii() -> None:
    global ACTIVE
    ACTIVE = GlyphsASCII


def use_nerd() -> None:
    global ACTIVE
    ACTIVE = Glyphs
```

- [ ] **Step 4: Verify importability**

Run: `conda run --no-capture-output -n uav python -c "from tui import theme; print(theme.Glyphs.DEMO, theme.Glyphs.LIVE)"`
Expected: prints two characters (one will be a Nerd Font glyph that may render as a box in a non-Nerd-Font terminal — that's fine).

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/__init__.py tui/__main__.py tui/theme.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): package skeleton + glyph theme

Empty placeholder __main__ + Nerd Font glyph constants with an
ASCII fallback class. The full launcher is wired up across the
next ~14 tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Disk state scanner (`tui/state/scan.py`)

Pure-function module the form pickers and state pane will read on demand.

**Files:**
- Create: `tui/state/__init__.py` (empty)
- Create: `tui/state/scan.py`
- Create: `tests/unit/test_tui_scan.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tui_scan.py`:

```python
"""Unit tests for tui/state/scan.py — promoted models, runs, trials, checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tui.state import scan


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    # models/
    (tmp_path / "models" / "red_v1").mkdir(parents=True)
    (tmp_path / "models" / "red_v1" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "models" / "blue_v1").mkdir(parents=True)
    (tmp_path / "models" / "blue_v1" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "models" / "no_best").mkdir(parents=True)  # no best_model.zip — should be skipped

    # runs/
    (tmp_path / "runs" / "ppo_hoop" / "20260101_000000").mkdir(parents=True)
    (tmp_path / "runs" / "ppo_hoop" / "20260202_000000").mkdir(parents=True)
    (tmp_path / "runs" / "ppo_hoop" / "20260202_000000" / "best_model.zip").write_bytes(b"x")
    (tmp_path / "runs" / "ppo_hoop_team" / "20260303_000000").mkdir(parents=True)

    # checkpoints in latest ppo_hoop trial — mix legacy (ppo_hoop_<step>) and
    # current (ppo_<step>) prefixes; both must be picked up by the regex.
    ckpt_dir = tmp_path / "runs" / "ppo_hoop" / "20260202_000000" / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "ppo_hoop_1000_steps.zip").write_bytes(b"x")   # legacy prefix
    (ckpt_dir / "ppo_5000_steps.zip").write_bytes(b"x")        # current prefix
    (ckpt_dir / "ppo_10000_steps.zip").write_bytes(b"x")       # current prefix

    # live progress file in one trial
    progress = {"schema_version": 1, "step": 5000}
    (tmp_path / "runs" / "ppo_hoop_team" / "20260303_000000" / "tui_progress.json"
     ).write_text(json.dumps(progress))

    return tmp_path


def test_promoted_models_lists_dirs_with_best_model(fixture_tree: Path) -> None:
    models = scan.promoted_models(fixture_tree / "models")
    names = [m.name for m in models]
    assert "red_v1" in names
    assert "blue_v1" in names
    assert "no_best" not in names


def test_run_names_sorted(fixture_tree: Path) -> None:
    names = scan.run_names(fixture_tree / "runs")
    assert names == ["ppo_hoop", "ppo_hoop_team"]


def test_trials_in_run_sorted_desc(fixture_tree: Path) -> None:
    trials = scan.trials_in_run("ppo_hoop", fixture_tree / "runs")
    names = [t.name for t in trials]
    assert names == ["20260202_000000", "20260101_000000"]
    assert trials[0].has_best_model is True
    assert trials[1].has_best_model is False


def test_checkpoints_sorted_desc_by_step(fixture_tree: Path) -> None:
    """The fixture mixes legacy `ppo_hoop_<step>_steps.zip` with current
    `ppo_<step>_steps.zip` filenames; both must be picked up and sorted
    together by step number."""
    ckpts = scan.checkpoints(
        "ppo_hoop", "20260202_000000", fixture_tree / "runs"
    )
    steps = [c.step for c in ckpts]
    assert steps == [10000, 5000, 1000]
    # The 1000-step file uses the legacy prefix — confirm it appears in the
    # returned path so consumers (the resume picker) can pass it as-is to
    # `--resume`, preserving filename-on-disk regardless of prefix.
    legacy = next(c for c in ckpts if c.step == 1000)
    assert legacy.path.name == "ppo_hoop_1000_steps.zip"


def test_live_trial_detection_uses_progress_file_mtime(fixture_tree: Path) -> None:
    progress = fixture_tree / "runs" / "ppo_hoop_team" / "20260303_000000" / "tui_progress.json"
    # File was just written by the fixture → mtime is now.
    trials = scan.trials_in_run("ppo_hoop_team", fixture_tree / "runs")
    assert trials[0].is_live is True

    # Backdate mtime by 60s — outside the 30s window.
    old = time.time() - 60
    import os
    os.utime(progress, (old, old))
    trials = scan.trials_in_run("ppo_hoop_team", fixture_tree / "runs")
    assert trials[0].is_live is False


def test_latest_trial_picks_newest_across_runs(fixture_tree: Path) -> None:
    latest = scan.latest_trial(fixture_tree / "runs")
    assert latest is not None
    assert latest.name == "20260303_000000"  # newest timestamp overall
```

- [ ] **Step 2: Verify the tests fail**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_scan.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `tui/state/scan.py`**

Create `tui/state/__init__.py` (empty).

Create `tui/state/scan.py`:

```python
"""Fresh-read disk scanner for the TUI: promoted models, runs, trials, checkpoints.

No caching — every call is a fresh ``os.scandir``. Cheap for low-hundreds of dirs.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

_LIVE_WINDOW_SEC = 30.0
# Prefix-agnostic: handles legacy single-agent files (ppo_hoop_<step>_steps.zip)
# alongside new ones (ppo_<step>_steps.zip) and any future prefix. The regex
# anchors on `_<digits>_steps.zip` at end of name.
_CKPT_RE = re.compile(r"_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class PromotedModel:
    name: str
    path: Path
    alias: str | None = None


@dataclass(frozen=True)
class Trial:
    run_name: str
    name: str
    path: Path
    has_best_model: bool
    is_live: bool


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


def promoted_models(models_root: Path = Path("models")) -> list[PromotedModel]:
    if not models_root.is_dir():
        return []
    out: list[PromotedModel] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "best_model.zip").is_file():
            continue
        alias_file = entry / "alias.txt"
        alias = alias_file.read_text().strip() if alias_file.is_file() else None
        out.append(PromotedModel(name=entry.name, path=entry, alias=alias))
    return out


def run_names(runs_root: Path = Path("runs")) -> list[str]:
    if not runs_root.is_dir():
        return []
    return sorted(p.name for p in runs_root.iterdir() if p.is_dir())


def trials_in_run(run_name: str, runs_root: Path = Path("runs")) -> list[Trial]:
    run_dir = runs_root / run_name
    if not run_dir.is_dir():
        return []
    trials: list[Trial] = []
    for t in run_dir.iterdir():
        if not t.is_dir():
            continue
        progress = t / "tui_progress.json"
        is_live = False
        if progress.is_file():
            is_live = (time.time() - progress.stat().st_mtime) < _LIVE_WINDOW_SEC
        trials.append(Trial(
            run_name=run_name,
            name=t.name,
            path=t,
            has_best_model=(t / "best_model.zip").is_file(),
            is_live=is_live,
        ))
    return sorted(trials, key=lambda x: x.name, reverse=True)


def checkpoints(run_name: str, trial_name: str,
                runs_root: Path = Path("runs")) -> list[Checkpoint]:
    ckpt_dir = runs_root / run_name / trial_name / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    ckpts: list[Checkpoint] = []
    for f in ckpt_dir.iterdir():
        m = _CKPT_RE.search(f.name)
        if m and f.is_file():
            ckpts.append(Checkpoint(step=int(m.group(1)), path=f))
    return sorted(ckpts, key=lambda c: c.step, reverse=True)


def latest_trial(runs_root: Path = Path("runs")) -> Trial | None:
    if not runs_root.is_dir():
        return None
    candidates: list[Trial] = []
    for run in run_names(runs_root):
        candidates.extend(trials_in_run(run, runs_root))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.name)
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_scan.py -v`
Expected: 6 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/state/__init__.py tui/state/scan.py tests/unit/test_tui_scan.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): disk state scanner

Pure functions returning frozen dataclasses for promoted models,
runs, trials, and checkpoints. Used by both the form pickers and
the right-hand state pane. Live-trial detection keyed on
tui_progress.json mtime within 30s.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Action dataclasses (`tui/actions/base.py`)

**Files:**
- Create: `tui/actions/__init__.py` (will re-export from registry in Task 13)
- Create: `tui/actions/base.py`

- [ ] **Step 1: Create `tui/actions/__init__.py`**

Create `tui/actions/__init__.py`:

```python
"""TUI action catalog — Action + FieldSpec types and the registry."""
from tui.actions.base import Action, FieldSpec, FieldKind  # noqa: F401
from tui.actions.registry import ACTIONS  # noqa: F401
```

- [ ] **Step 2: Create `tui/actions/base.py`**

Create `tui/actions/base.py`:

```python
"""Declarative action types for the TUI's action catalog.

An Action describes one form-driven invocation: name, glyph, group, fields,
and a ``build_argv`` callable that turns form values into a subprocess argv.
Actions are read by the action tree (left pane) and form (centre pane).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class FieldKind(str, Enum):
    TEXT = "text"
    INT = "int"
    BOOL = "bool"
    PICKER_MODELS = "picker:models"
    PICKER_RUNS = "picker:runs"
    PICKER_TRIALS_IN_RUN = "picker:trials_in_run"
    PICKER_CHECKPOINTS_IN_TRIAL = "picker:checkpoints_in_trial"


@dataclass(frozen=True)
class FieldSpec:
    name: str                       # form-field id, also passed to build_argv
    label: str                      # user-facing label
    kind: FieldKind
    required: bool = False
    default: Any = None
    depends_on: tuple[str, ...] = ()  # other field names whose values seed this picker


@dataclass(frozen=True)
class Action:
    key: str                        # unique id used in tree / hotkeys / CLI
    label: str                      # user-facing name shown in tree
    group: Literal["Demo", "Train", "Eval", "Manage"]
    glyph_attr: str                 # name of an attribute on tui.theme.ACTIVE (e.g. "DEMO")
    fields: tuple[FieldSpec, ...] = ()
    requires_mjpython: bool = False
    is_training: bool = False       # → occupies the training slot in ProcessManager
    is_tensorboard: bool = False    # → the tensorboard daemon
    build_argv: Callable[[dict[str, Any]], list[str]] = field(
        default=lambda _: [],
        repr=False,
    )
```

- [ ] **Step 3: Verify import (no test yet — registry Task 13 has the table-driven tests)**

Run: `conda run --no-capture-output -n uav python -c "from tui.actions.base import Action, FieldSpec, FieldKind; print(FieldKind.PICKER_MODELS.value)"`
Expected: prints `picker:models`.

- [ ] **Step 4: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/actions/__init__.py tui/actions/base.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): Action and FieldSpec dataclasses

Declarative types for the action catalog: each Action carries its
fields, glyph, group, mjpython requirement, and a build_argv callable
that maps form values to subprocess argv. The registry lands next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Action registry — all 16 actions + argv tests

**Files:**
- Create: `tui/actions/demos.py`
- Create: `tui/actions/train.py`
- Create: `tui/actions/eval.py`
- Create: `tui/actions/manage.py`
- Create: `tui/actions/registry.py`
- Create: `tests/unit/test_tui_actions.py`

- [ ] **Step 1: Write the failing tests (table-driven)**

Create `tests/unit/test_tui_actions.py`:

```python
"""Table-driven tests: action.build_argv(form_values) → expected argv."""
from __future__ import annotations

import pytest

from tui.actions import ACTIONS


def _action(key: str):
    for a in ACTIONS:
        if a.key == key:
            return a
    raise KeyError(key)


def test_registry_has_16_actions() -> None:
    assert len(ACTIONS) == 16
    groups = {a.group for a in ACTIONS}
    assert groups == {"Demo", "Train", "Eval", "Manage"}


@pytest.mark.parametrize("key,values,expected", [
    ("hover", {}, ["mjpython", "demo/menu.py", "hover"]),
    ("waypoint", {}, ["mjpython", "demo/menu.py", "waypoint"]),
    ("takedown", {}, ["mjpython", "demo/menu.py", "takedown"]),
    ("score-through-tag", {}, ["mjpython", "demo/menu.py", "score-through-tag"]),
])
def test_demo_argvs(key, values, expected):
    assert _action(key).build_argv(values) == expected


def test_train_single_no_optional_args() -> None:
    a = _action("train")
    assert a.is_training is True
    argv = a.build_argv({"RUN_NAME": "", "PRETRAIN": ""})
    assert argv == ["python", "scripts/train_ppo.py"]


def test_train_single_with_run_name() -> None:
    a = _action("train")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop_x", "PRETRAIN": ""})
    assert argv == ["python", "scripts/train_ppo.py", "--run-name", "ppo_hoop_x"]


def test_train_single_with_pretrain() -> None:
    a = _action("train")
    argv = a.build_argv({"RUN_NAME": "", "PRETRAIN": "models/red_v1"})
    assert argv == ["python", "scripts/train_ppo.py",
                    "--pretrain", "models/red_v1/best_model"]


def test_train_team_red_with_warm_start() -> None:
    a = _action("train-team-red")
    argv = a.build_argv({"RUN_NAME": "phase2a", "WARM_START": "models/rs"})
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--learner", "red_0", "--opponent", "beeline_blue",
        "--run-name", "phase2a",
        "--warm-start", "models/rs/best_model",
    ]


def test_train_team_blue_requires_red() -> None:
    a = _action("train-team-blue")
    argv = a.build_argv({"RUN_NAME": "phase2b", "RED": "models/red_v1"})
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--learner", "blue_0",
        "--opponent", "frozen:models/red_v1/best_model",
        "--run-name", "phase2b",
    ]


def test_resume_argv() -> None:
    a = _action("resume")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260101_000000",
        "CHECKPOINT": "runs/ppo_hoop/20260101_000000/checkpoints/ppo_5000_steps",
    })
    assert argv == [
        "python", "scripts/train_ppo.py",
        "--run-name", "ppo_hoop",
        "--resume", "runs/ppo_hoop/20260101_000000/checkpoints/ppo_5000_steps",
    ]


def test_resume_team_argv() -> None:
    a = _action("resume-team")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop_blue_1",
        "TRIAL": "20260101_000000",
        "CHECKPOINT": "runs/ppo_hoop_blue_1/20260101_000000/checkpoints/ppo_10000_steps",
        "LEARNER": "", "OPPONENT": "",
    })
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--resume", "runs/ppo_hoop_blue_1/20260101_000000/checkpoints/ppo_10000_steps",
        "--run-name", "ppo_hoop_blue_1",
    ]


def test_eval_gui_uses_mjpython() -> None:
    a = _action("eval")
    assert a.requires_mjpython is True  # the action declares it; runner picks mjpython
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260202_000000",
        "EPISODES": 10,
        "GUI": True,
    })
    assert argv == [
        "mjpython", "scripts/eval_ppo.py",
        "--model", "runs/ppo_hoop/20260202_000000/best_model",
        "--episodes", "10",
    ]


def test_eval_headless_uses_python_and_no_render() -> None:
    a = _action("eval")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260202_000000",
        "EPISODES": 50,
        "GUI": False,
    })
    assert argv == [
        "python", "scripts/eval_ppo.py",
        "--model", "runs/ppo_hoop/20260202_000000/best_model",
        "--episodes", "50",
        "--no-render",
    ]


def test_eval_team_with_flags() -> None:
    a = _action("eval-team")
    argv = a.build_argv({
        "RED": "models/red_v1", "BLUE": "models/blue_v1",
        "EPISODES": 5, "GUI": True, "DETERMINISTIC": True,
    })
    assert argv == [
        "mjpython", "scripts/eval_team.py",
        "--red", "models/red_v1", "--blue", "models/blue_v1",
        "--episodes", "5",
        "--gui",
        "--deterministic",
    ]


def test_tensorboard_all_runs() -> None:
    a = _action("tensorboard")
    assert a.is_tensorboard is True
    argv = a.build_argv({"RUN_NAME": ""})
    assert argv == ["tensorboard", "--logdir", "runs"]


def test_tensorboard_single_run() -> None:
    a = _action("tensorboard")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop"})
    assert argv == ["tensorboard", "--logdir", "runs/ppo_hoop"]


def test_promote_argv() -> None:
    a = _action("promote")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop", "TRIAL": "20260101_000000"})
    assert argv == [
        "python", "scripts/promote.py",
        "--trial", "runs/ppo_hoop/20260101_000000",
    ]


def test_list_runs_argv() -> None:
    a = _action("list-runs")
    assert a.build_argv({}) == ["python", "scripts/list_runs.py"]


def test_lineage_argv() -> None:
    a = _action("lineage")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop", "TRIAL": "20260101_000000"})
    assert argv == ["python", "scripts/lineage.py", "runs/ppo_hoop/20260101_000000"]


def test_repro_argv() -> None:
    a = _action("repro")
    argv = a.build_argv({"MODEL": "models/red_v1"})
    assert argv == ["python", "scripts/repro.py", "--model-dir", "models/red_v1"]
```

- [ ] **Step 2: Verify the tests fail**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_actions.py -v`
Expected: FAIL — modules don't exist yet (`ImportError`).

- [ ] **Step 3: Implement `tui/actions/demos.py`**

Create `tui/actions/demos.py`:

```python
"""The 4 demo actions — each shells out to ``mjpython demo/menu.py <key>``."""
from __future__ import annotations

from tui.actions.base import Action


def _demo_argv(key: str):
    def _f(_values: dict) -> list[str]:
        return ["mjpython", "demo/menu.py", key]
    return _f


DEMOS = [
    Action(
        key=k,
        label=k,
        group="Demo",
        glyph_attr="DEMO",
        fields=(),
        requires_mjpython=True,
        build_argv=_demo_argv(k),
    )
    for k in ("hover", "waypoint", "takedown", "score-through-tag")
]
```

- [ ] **Step 4: Implement `tui/actions/train.py`**

Create `tui/actions/train.py`:

```python
"""Training-slot actions: train (single), train-team-red, train-team-blue, resume, resume-team."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _train_single_argv(v: dict) -> list[str]:
    out = ["python", "scripts/train_ppo.py"]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    if v.get("PRETRAIN"):
        out += ["--pretrain", f"{v['PRETRAIN']}/best_model"]
    return out


def _train_team_red_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--learner", "red_0", "--opponent", "beeline_blue",
    ]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    if v.get("WARM_START"):
        out += ["--warm-start", f"{v['WARM_START']}/best_model"]
    return out


def _train_team_blue_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--learner", "blue_0",
        "--opponent", f"frozen:{v['RED']}/best_model",
    ]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    return out


def _resume_argv(v: dict) -> list[str]:
    return [
        "python", "scripts/train_ppo.py",
        "--run-name", v["RUN_NAME"],
        "--resume", v["CHECKPOINT"],
    ]


def _resume_team_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--resume", v["CHECKPOINT"],
        "--run-name", v["RUN_NAME"],
    ]
    if v.get("LEARNER"):
        out += ["--learner", v["LEARNER"]]
    if v.get("OPPONENT"):
        out += ["--opponent", v["OPPONENT"]]
    return out


TRAIN = [
    Action(
        key="train",
        label="train single-agent",
        group="Train",
        glyph_attr="TRAIN",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="PRETRAIN", label="Pretrain from", kind=FieldKind.PICKER_MODELS),
        ),
        is_training=True,
        build_argv=_train_single_argv,
    ),
    Action(
        key="train-team-red",
        label="train-team-red",
        group="Train",
        glyph_attr="TRAIN_TEAM",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="WARM_START", label="Warm start from",
                      kind=FieldKind.PICKER_MODELS),
        ),
        is_training=True,
        build_argv=_train_team_red_argv,
    ),
    Action(
        key="train-team-blue",
        label="train-team-blue",
        group="Train",
        glyph_attr="TRAIN_TEAM",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="RED", label="Red opponent (frozen)",
                      kind=FieldKind.PICKER_MODELS, required=True),
        ),
        is_training=True,
        build_argv=_train_team_blue_argv,
    ),
    Action(
        key="resume",
        label="resume",
        group="Train",
        glyph_attr="RESUME",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="CHECKPOINT", label="Checkpoint",
                      kind=FieldKind.PICKER_CHECKPOINTS_IN_TRIAL, required=True,
                      depends_on=("RUN_NAME", "TRIAL")),
        ),
        is_training=True,
        build_argv=_resume_argv,
    ),
    Action(
        key="resume-team",
        label="resume-team",
        group="Train",
        glyph_attr="RESUME",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="CHECKPOINT", label="Checkpoint",
                      kind=FieldKind.PICKER_CHECKPOINTS_IN_TRIAL, required=True,
                      depends_on=("RUN_NAME", "TRIAL")),
            FieldSpec(name="LEARNER", label="Learner override (optional)",
                      kind=FieldKind.TEXT),
            FieldSpec(name="OPPONENT", label="Opponent spec override (optional)",
                      kind=FieldKind.TEXT),
        ),
        is_training=True,
        build_argv=_resume_team_argv,
    ),
]
```

- [ ] **Step 5: Implement `tui/actions/eval.py`**

Create `tui/actions/eval.py`:

```python
"""Eval actions: single-agent eval + team eval. GUI flag selects python vs mjpython."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _eval_argv(v: dict) -> list[str]:
    interp = "mjpython" if v.get("GUI") else "python"
    out = [
        interp, "scripts/eval_ppo.py",
        "--model", f"runs/{v['RUN_NAME']}/{v['TRIAL']}/best_model",
        "--episodes", str(int(v.get("EPISODES") or 10)),
    ]
    if not v.get("GUI"):
        out.append("--no-render")
    return out


def _eval_team_argv(v: dict) -> list[str]:
    interp = "mjpython" if v.get("GUI") else "python"
    out = [
        interp, "scripts/eval_team.py",
        "--red", v["RED"], "--blue", v["BLUE"],
        "--episodes", str(int(v.get("EPISODES") or (5 if v.get("GUI") else 100))),
    ]
    if v.get("GUI"):
        out.append("--gui")
    if v.get("DETERMINISTIC"):
        out.append("--deterministic")
    return out


EVAL = [
    Action(
        key="eval",
        label="eval",
        group="Eval",
        glyph_attr="EVAL",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="EPISODES", label="Episodes", kind=FieldKind.INT, default=10),
            FieldSpec(name="GUI", label="GUI viewer", kind=FieldKind.BOOL, default=True),
        ),
        requires_mjpython=True,  # when GUI=True; runner inspects argv[0]
        build_argv=_eval_argv,
    ),
    Action(
        key="eval-team",
        label="eval-team",
        group="Eval",
        glyph_attr="EVAL",
        fields=(
            FieldSpec(name="RED", label="Red model", kind=FieldKind.PICKER_MODELS, required=True),
            FieldSpec(name="BLUE", label="Blue model", kind=FieldKind.PICKER_MODELS, required=True),
            FieldSpec(name="EPISODES", label="Episodes", kind=FieldKind.INT, default=5),
            FieldSpec(name="GUI", label="GUI viewer", kind=FieldKind.BOOL, default=True),
            FieldSpec(name="DETERMINISTIC", label="Deterministic", kind=FieldKind.BOOL, default=True),
        ),
        requires_mjpython=True,
        build_argv=_eval_team_argv,
    ),
]
```

- [ ] **Step 6: Implement `tui/actions/manage.py`**

Create `tui/actions/manage.py`:

```python
"""Manage actions: tensorboard, promote, list-runs, lineage, repro."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _tensorboard_argv(v: dict) -> list[str]:
    logdir = f"runs/{v['RUN_NAME']}" if v.get("RUN_NAME") else "runs"
    return ["tensorboard", "--logdir", logdir]


def _promote_argv(v: dict) -> list[str]:
    return ["python", "scripts/promote.py",
            "--trial", f"runs/{v['RUN_NAME']}/{v['TRIAL']}"]


def _list_runs_argv(_v: dict) -> list[str]:
    return ["python", "scripts/list_runs.py"]


def _lineage_argv(v: dict) -> list[str]:
    return ["python", "scripts/lineage.py", f"runs/{v['RUN_NAME']}/{v['TRIAL']}"]


def _repro_argv(v: dict) -> list[str]:
    return ["python", "scripts/repro.py", "--model-dir", v["MODEL"]]


MANAGE = [
    Action(
        key="tensorboard",
        label="tensorboard",
        group="Manage",
        glyph_attr="TENSORBOARD",
        fields=(FieldSpec(name="RUN_NAME", label="Run (optional — all if blank)",
                          kind=FieldKind.PICKER_RUNS),),
        is_tensorboard=True,
        build_argv=_tensorboard_argv,
    ),
    Action(
        key="promote",
        label="promote",
        group="Manage",
        glyph_attr="PROMOTE",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
        ),
        build_argv=_promote_argv,
    ),
    Action(
        key="list-runs",
        label="list-runs",
        group="Manage",
        glyph_attr="LIST_RUNS",
        fields=(),
        build_argv=_list_runs_argv,
    ),
    Action(
        key="lineage",
        label="lineage",
        group="Manage",
        glyph_attr="LINEAGE",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
        ),
        build_argv=_lineage_argv,
    ),
    Action(
        key="repro",
        label="repro",
        group="Manage",
        glyph_attr="REPRO",
        fields=(FieldSpec(name="MODEL", label="Promoted model",
                          kind=FieldKind.PICKER_MODELS, required=True),),
        build_argv=_repro_argv,
    ),
]
```

- [ ] **Step 7: Implement `tui/actions/registry.py`**

Create `tui/actions/registry.py`:

```python
"""Single source of truth: ordered list of all actions."""
from __future__ import annotations

from tui.actions.demos import DEMOS
from tui.actions.eval import EVAL
from tui.actions.manage import MANAGE
from tui.actions.train import TRAIN

ACTIONS = [*DEMOS, *TRAIN, *EVAL, *MANAGE]
```

- [ ] **Step 8: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_actions.py -v`
Expected: all parametrized cases + 16 individual tests pass.

- [ ] **Step 9: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/actions/ tests/unit/test_tui_actions.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): action registry — 16 actions with argv builders

Four files grouping the catalog (demos, train, eval, manage) and a
registry that exposes a single ordered ACTIONS list. Each action
declares its fields and a build_argv callable. Table-driven tests
cover the argv-construction surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Progress tailer (`tui/process/progress.py`)

Reads `tui_progress.json` for the active training trial. Handles missing, partial (mid-rename), and unknown-schema files gracefully.

**Files:**
- Create: `tui/process/__init__.py` (empty)
- Create: `tui/process/progress.py`
- Create: `tests/unit/test_tui_progress_tail.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tui_progress_tail.py`:

```python
"""Unit tests for tui/process/progress.py — tailing tui_progress.json."""
from __future__ import annotations

import json
from pathlib import Path

from tui.process.progress import read_snapshot, ProgressSnapshot, SCHEMA_VERSION_SUPPORTED


def _write_snapshot(p: Path, **overrides) -> None:
    base = {
        "schema_version": 1,
        "ts": 1.0, "run_name": "x", "trial": "y",
        "kind": "single", "learner": None, "opponent": None,
        "step": 100, "total_steps": 1000, "fps": 100.0, "elapsed_sec": 1.0,
        "ep_rew_mean": 0.5, "ep_len_mean": 200.0,
        "best_so_far": {"reward": 0.5, "step": 100},
        "recent_rewards": [0.5],
    }
    base.update(overrides)
    p.write_text(json.dumps(base))


def test_read_snapshot_returns_dataclass(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    _write_snapshot(p)
    snap = read_snapshot(p)
    assert isinstance(snap, ProgressSnapshot)
    assert snap.step == 100
    assert snap.total_steps == 1000


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_snapshot(tmp_path / "absent.json") is None


def test_partial_json_returns_none_without_raising(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    p.write_text('{"schema_version": 1, "step": ')  # truncated mid-write
    assert read_snapshot(p) is None


def test_newer_schema_version_returns_none_gracefully(tmp_path: Path) -> None:
    p = tmp_path / "tui_progress.json"
    _write_snapshot(p, schema_version=SCHEMA_VERSION_SUPPORTED + 1)
    assert read_snapshot(p) is None
```

- [ ] **Step 2: Verify the tests fail**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_progress_tail.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `tui/process/progress.py`**

Create `tui/process/__init__.py` (empty).

Create `tui/process/progress.py`:

```python
"""Read snapshots of ``tui_progress.json`` written by TUIProgressCallback.

Tolerant to: missing file, partial JSON (a mid-write race despite the callback
using atomic os.replace — defensive), and unsupported newer schema versions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION_SUPPORTED = 1


@dataclass(frozen=True)
class ProgressSnapshot:
    schema_version: int
    ts: float
    run_name: str
    trial: str
    kind: str
    learner: str | None
    opponent: str | None
    step: int
    total_steps: int
    fps: float
    elapsed_sec: float
    ep_rew_mean: float | None
    ep_len_mean: float | None
    best_so_far: dict[str, Any] | None
    recent_rewards: list[float]


def read_snapshot(path: Path) -> ProgressSnapshot | None:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("schema_version") != SCHEMA_VERSION_SUPPORTED:
        return None
    try:
        return ProgressSnapshot(**data)
    except TypeError:
        return None
```

- [ ] **Step 4: Verify the tests pass**

Run: `conda run --no-capture-output -n uav python -m pytest tests/unit/test_tui_progress_tail.py -v`
Expected: 4 passed.

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/process/__init__.py tui/process/progress.py tests/unit/test_tui_progress_tail.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): progress snapshot reader

Reads tui_progress.json into a frozen dataclass; returns None for
missing, partial-JSON, or unsupported-schema files. The training
widget polls this every 500ms.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Process manager (`tui/process/manager.py`)

Two-slot subprocess manager. Tested via a simple smoke (not full unit) — Popen lifecycle is awkward to mock and the manager is small.

**Files:**
- Create: `tui/process/manager.py`

- [ ] **Step 1: Implement `tui/process/manager.py`**

Create `tui/process/manager.py`:

```python
"""Two-slot subprocess manager for the TUI.

Slots:
- ``training``: one of {train, train-team-*, resume, resume-team}. Mutex among
  themselves; running while ``aux`` runs is OK.
- ``aux``: one of {demos, eval, eval-team, tensorboard, promote, list-runs,
  lineage, repro}. Mutex among themselves; running while ``training`` runs is OK.

Output lines are captured into a ring buffer per slot for the log overlay.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Literal

SlotName = Literal["training", "aux"]
_RING = 1000


class ManagedProcess:
    def __init__(self, argv: list[str], env: dict[str, str] | None,
                 run_dir: Path | None = None) -> None:
        self.argv = argv
        self.run_dir = run_dir
        self._lines: deque[str] = deque(maxlen=_RING)
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, **(env or {})},
        )
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.append(line.rstrip("\n"))

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def returncode(self) -> int | None:
        return self._proc.returncode

    def tail(self, n: int = 200) -> list[str]:
        return list(self._lines)[-n:]

    def stop(self, *, escalate_after_sec: float = 5.0) -> None:
        if not self.is_alive():
            return
        self._proc.send_signal(signal.SIGINT)
        try:
            self._proc.wait(timeout=escalate_after_sec)
        except subprocess.TimeoutExpired:
            self._proc.terminate()


class ProcessManager:
    def __init__(self) -> None:
        self._slots: dict[SlotName, ManagedProcess | None] = {"training": None, "aux": None}

    def start(self, slot: SlotName, argv: list[str], *,
              env: dict[str, str] | None = None,
              run_dir: Path | None = None) -> ManagedProcess:
        current = self._slots[slot]
        if current is not None and current.is_alive():
            raise RuntimeError(f"slot {slot!r} is already running; stop it first")
        # Reap a dead process so a new one can start.
        proc = ManagedProcess(argv, env=env, run_dir=run_dir)
        self._slots[slot] = proc
        return proc

    def stop(self, slot: SlotName) -> None:
        p = self._slots[slot]
        if p is not None:
            p.stop()

    def is_running(self, slot: SlotName) -> bool:
        p = self._slots[slot]
        return p is not None and p.is_alive()

    def current(self, slot: SlotName) -> ManagedProcess | None:
        p = self._slots[slot]
        if p is None:
            return None
        return p if p.is_alive() else None

    def stop_all(self) -> None:
        for slot in self._slots:
            self.stop(slot)
```

- [ ] **Step 2: Smoke-test the manager**

Run: `conda run --no-capture-output -n uav python -c "
from tui.process.manager import ProcessManager
import time
pm = ProcessManager()
p = pm.start('aux', ['python', '-c', 'print(\"hi\"); import time; time.sleep(0.5)'])
time.sleep(1.0)
assert not pm.is_running('aux')
assert p.returncode() == 0
print('tail:', p.tail())
"`
Expected: prints `tail: ['hi']`.

- [ ] **Step 3: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/process/manager.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): two-slot subprocess manager

ManagedProcess wraps Popen + a 1000-line ring buffer; ProcessManager
exposes training and aux slots with start/stop/is_running. Training
and aux can run concurrently; within a slot, a new start fails if
one is already alive.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Textual CSS theme (`tui/app.tcss`)

**Files:**
- Create: `tui/app.tcss`

- [ ] **Step 1: Create `tui/app.tcss`**

Create `tui/app.tcss`:

```css
/* Drone-Quidditch TUI theme. Palette per spec §9. */

Screen {
    background: #0f1117;
    color: #e2e4ea;
}

Header {
    background: #161922;
    color: #a78bfa;
    height: 1;
    text-style: bold;
}

Footer {
    background: #161922;
    color: #7c69d6;
}

.panel {
    border: round #2a2f3d;
    background: #161922;
    padding: 0 1;
}

#actions {
    width: 30;
}

#form {
    width: 1fr;
}

#state {
    width: 36;
}

#training {
    border: heavy #a78bfa;
    height: 7;
}

.dim    { color: #5a6178; }
.accent { color: #a78bfa; }
.warn   { color: #e0a85a; }
.good   { color: #7ec27e; }
.red    { color: #c1432d; }
.blue   { color: #2d63c1; }

.tree-group {
    color: #a78bfa;
    text-style: bold;
}

.tree-disabled {
    color: #5a6178;
}

Button {
    background: #2a2f3d;
    color: #e2e4ea;
}

Button:hover {
    background: #a78bfa;
    color: #0f1117;
}

ProgressBar > .bar--bar {
    color: #7ec27e;
}
```

- [ ] **Step 2: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/app.tcss
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): CSS theme — violet accent, rounded panels

Three-pane layout sizing + palette per spec §9. The training widget
gets a heavy violet border for visual prominence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: ActionTree widget (`tui/widgets/action_tree.py`)

Left pane. Grouped list of actions with glyphs, keyboard navigation, and disabled rendering for the Train group while training is running.

**Files:**
- Create: `tui/widgets/__init__.py` (empty)
- Create: `tui/widgets/action_tree.py`

- [ ] **Step 1: Create `tui/widgets/__init__.py`**

Create `tui/widgets/__init__.py` (empty file).

- [ ] **Step 2: Implement `tui/widgets/action_tree.py`**

Create `tui/widgets/action_tree.py`:

```python
"""Left-pane action tree — grouped, keyboard-navigable, disable-aware."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from tui import theme
from tui.actions import ACTIONS
from tui.actions.base import Action

_GROUP_ORDER = ("Demo", "Train", "Eval", "Manage")


class ActionTree(Widget):
    """Renders ACTIONS grouped by `group`, with selection + disabled state."""

    cursor: reactive[int] = reactive(0)
    training_disabled: reactive[bool] = reactive(False)

    class Selected(Message):
        def __init__(self, action: Action) -> None:
            super().__init__()
            self.action = action

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="tree-scroll")

    def on_mount(self) -> None:
        self._refresh()

    def watch_cursor(self, _old: int, _new: int) -> None:
        self._refresh()
        self.post_message(self.Selected(self._flat()[self.cursor]))

    def watch_training_disabled(self, _old: bool, _new: bool) -> None:
        self._refresh()

    def key_down(self) -> None:
        flat = self._flat()
        if self.cursor < len(flat) - 1:
            self.cursor += 1

    def key_up(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def _flat(self) -> list[Action]:
        return [a for g in _GROUP_ORDER for a in ACTIONS if a.group == g]

    def _is_disabled(self, action: Action) -> bool:
        return self.training_disabled and action.is_training

    def _refresh(self) -> None:
        scroll = self.query_one("#tree-scroll", VerticalScroll)
        scroll.remove_children()
        flat = self._flat()
        ix = 0
        for group in _GROUP_ORDER:
            members = [a for a in ACTIONS if a.group == group]
            if not members:
                continue
            scroll.mount(Static(f"  {group}", classes="tree-group"))
            for a in members:
                glyph = getattr(theme.ACTIVE, a.glyph_attr, "·")
                lock = "  " + theme.ACTIVE.LOCK if self._is_disabled(a) else ""
                arrow = "▸" if ix == self.cursor else " "
                cls = "tree-disabled" if self._is_disabled(a) else ""
                scroll.mount(Static(f"  {arrow} {glyph}  {a.label}{lock}", classes=cls))
                ix += 1
```

- [ ] **Step 3: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/widgets/__init__.py tui/widgets/action_tree.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): action tree widget

Grouped list of actions (Demo / Train / Eval / Manage) with cursor
navigation, glyph rendering, and disabled state for the Train group
when training is active. Emits Selected messages on cursor changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: ActionForm widget (`tui/widgets/action_form.py`)

Centre pane. Dynamic per selected action.

**Files:**
- Create: `tui/widgets/action_form.py`

- [ ] **Step 1: Implement `tui/widgets/action_form.py`**

Create `tui/widgets/action_form.py`:

```python
"""Centre-pane form: renders a selected Action's fields, validates, builds argv."""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch

from tui import theme
from tui.actions.base import Action, FieldKind, FieldSpec
from tui.state import scan


class ActionForm(Widget):
    """Renders the currently-selected action's form fields and Run / Dry-run buttons."""

    class Run(Message):
        def __init__(self, action: Action, values: dict[str, Any]) -> None:
            super().__init__()
            self.action = action
            self.values = values

    class DryRun(Message):
        def __init__(self, action: Action, argv: list[str]) -> None:
            super().__init__()
            self.action = action
            self.argv = argv

    def __init__(self) -> None:
        super().__init__()
        self._action: Action | None = None
        self._values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="form-body")

    def set_action(self, action: Action) -> None:
        self._action = action
        self._values = {f.name: f.default for f in action.fields}
        self._render()

    def _render(self) -> None:
        body = self.query_one("#form-body", Vertical)
        body.remove_children()
        a = self._action
        if a is None:
            body.mount(Static("Select an action ↑↓", classes="dim"))
            return

        glyph = getattr(theme.ACTIVE, a.glyph_attr, "·")
        body.mount(Static(f"{glyph}  {a.label}", classes="accent"))
        body.mount(Static(""))  # spacer

        for f in a.fields:
            body.mount(Static(f.label + (" *" if f.required else "")))
            body.mount(self._build_widget(f))

        body.mount(Static(""))
        body.mount(Horizontal(
            Button("Run", id="btn-run", variant="success"),
            Button("Dry-run", id="btn-dryrun"),
        ))

    def _build_widget(self, f: FieldSpec) -> Widget:
        wid = f"field-{f.name}"
        if f.kind == FieldKind.TEXT:
            w = Input(value=str(f.default or ""), id=wid)
            return w
        if f.kind == FieldKind.INT:
            return Input(value=str(f.default or ""), id=wid, type="integer")
        if f.kind == FieldKind.BOOL:
            return Switch(value=bool(f.default), id=wid)
        # picker:* — fill choices from scan
        return Select(self._picker_choices(f), id=wid, prompt="(none)")

    def _picker_choices(self, f: FieldSpec) -> list[tuple[str, str]]:
        if f.kind == FieldKind.PICKER_MODELS:
            return [(m.alias or m.name, str(m.path)) for m in scan.promoted_models()]
        if f.kind == FieldKind.PICKER_RUNS:
            return [(n, n) for n in scan.run_names()]
        if f.kind == FieldKind.PICKER_TRIALS_IN_RUN:
            run = self._values.get("RUN_NAME") or ""
            return [(t.name, t.name) for t in scan.trials_in_run(run)] if run else []
        if f.kind == FieldKind.PICKER_CHECKPOINTS_IN_TRIAL:
            run = self._values.get("RUN_NAME") or ""
            trial = self._values.get("TRIAL") or ""
            if run and trial:
                return [
                    (c.path.with_suffix("").name, str(c.path.with_suffix("")))
                    for c in scan.checkpoints(run, trial)
                ]
            return []
        return []

    # ------------------------------------------------------------------
    # Event handlers — keep field state in self._values and refresh
    # dependent pickers when their source field changes.

    def on_input_changed(self, event: Input.Changed) -> None:
        field_name = event.input.id.removeprefix("field-")
        self._values[field_name] = event.value
        self._refresh_dependents_of(field_name)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        field_name = event.switch.id.removeprefix("field-")
        self._values[field_name] = event.value

    def on_select_changed(self, event: Select.Changed) -> None:
        field_name = event.select.id.removeprefix("field-")
        self._values[field_name] = event.value
        self._refresh_dependents_of(field_name)

    def _refresh_dependents_of(self, source: str) -> None:
        a = self._action
        if a is None:
            return
        for f in a.fields:
            if source in f.depends_on and f.kind.value.startswith("picker:"):
                try:
                    sel = self.query_one(f"#field-{f.name}", Select)
                except Exception:
                    continue
                sel.set_options(self._picker_choices(f))
                self._values[f.name] = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._action is None:
            return
        if event.button.id == "btn-run":
            if self._validate():
                self.post_message(self.Run(self._action, dict(self._values)))
        elif event.button.id == "btn-dryrun":
            argv = self._action.build_argv(self._values)
            self.post_message(self.DryRun(self._action, argv))

    def _validate(self) -> bool:
        for f in self._action.fields:  # type: ignore[union-attr]
            if f.required and not self._values.get(f.name):
                self.app.notify(f"required: {f.label}", severity="error")
                return False
        return True
```

- [ ] **Step 2: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/widgets/action_form.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): dynamic action form widget

Renders the selected action's fields (text / int / bool / pickers),
keeps a values dict, refreshes dependent pickers when upstream field
changes, validates required fields, and emits Run / DryRun messages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: StatePane widget (`tui/widgets/state_pane.py`)

Right pane. Lists promoted models + 10 most-recent trials globally. Live trials marked with `●`.

**Files:**
- Create: `tui/widgets/state_pane.py`

- [ ] **Step 1: Implement `tui/widgets/state_pane.py`**

Create `tui/widgets/state_pane.py`:

```python
"""Right-pane state: promoted models + recent trials. Refreshes every 5s."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from tui import theme
from tui.state import scan

_REFRESH_SEC = 5.0
_TRIALS_SHOWN = 10


class StatePane(Widget):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="state-scroll")

    def on_mount(self) -> None:
        self._refresh()
        self._timer: Timer = self.set_interval(_REFRESH_SEC, self._refresh)

    def _refresh(self) -> None:
        scroll = self.query_one("#state-scroll", VerticalScroll)
        scroll.remove_children()

        scroll.mount(Static("Promoted models", classes="accent"))
        models = scan.promoted_models()
        if not models:
            scroll.mount(Static("  (none)", classes="dim"))
        else:
            for m in models:
                label = f"  • {m.alias}" if m.alias else f"  • {m.name}"
                scroll.mount(Static(label))
                if m.alias:
                    scroll.mount(Static(f"    {m.name}", classes="dim"))

        scroll.mount(Static(""))
        scroll.mount(Static("Recent trials", classes="accent"))
        all_trials = []
        for run in scan.run_names():
            all_trials.extend(scan.trials_in_run(run))
        recent = sorted(all_trials, key=lambda t: t.name, reverse=True)[:_TRIALS_SHOWN]
        if not recent:
            scroll.mount(Static("  (none)", classes="dim"))
        else:
            for t in recent:
                live = f"  {theme.ACTIVE.LIVE} live" if t.is_live else ""
                scroll.mount(Static(f"  {t.run_name}"))
                scroll.mount(Static(f"    {t.name}{live}",
                                    classes="good" if t.is_live else "dim"))
```

- [ ] **Step 2: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/widgets/state_pane.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): state pane — promoted models + recent trials

Refreshes every 5s from tui.state.scan. Live trials marked with a
dot glyph based on tui_progress.json mtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: TrainingWidget (`tui/widgets/training_widget.py`)

Docked bottom panel. Reads `tui_progress.json` and renders the progress bar + stats.

**Files:**
- Create: `tui/widgets/training_widget.py`

- [ ] **Step 1: Implement `tui/widgets/training_widget.py`**

Create `tui/widgets/training_widget.py`:

```python
"""Docked-bottom training widget — reads the JSON status file every 500ms."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

from tui.process.progress import read_snapshot, ProgressSnapshot

_POLL_MS = 500


def _fmt_secs(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = max(hi - lo, 1e-9)
    return "".join(bars[1 + int((v - lo) / rng * 7)] for v in values)


class TrainingWidget(Widget):
    """Reads ``<run_dir>/tui_progress.json`` and renders progress + stats."""

    def __init__(self) -> None:
        super().__init__()
        self._run_dir: Path | None = None
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("", id="train-title"),
            ProgressBar(id="train-bar", show_eta=False, total=100),
            Static("", id="train-stats"),
            Static("", id="train-spark"),
            id="train-body",
        )

    def follow(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        if self._timer is None:
            self._timer = self.set_interval(_POLL_MS / 1000, self._refresh)
        self._refresh()

    def stop_following(self) -> None:
        self._run_dir = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.query_one("#train-title", Static).update("(no active training)")
        self.query_one("#train-stats", Static).update("")
        self.query_one("#train-spark", Static).update("")

    def _refresh(self) -> None:
        if self._run_dir is None:
            return
        snap = read_snapshot(self._run_dir / "tui_progress.json")
        if snap is None:
            self.query_one("#train-title", Static).update("waiting for first checkpoint…")
            return
        self._update_from(snap)

    def _update_from(self, s: ProgressSnapshot) -> None:
        title = f"⚑ Training — {s.run_name}/{s.trial}"
        if s.kind == "team":
            title += f"  ({s.learner} vs {s.opponent})"
        self.query_one("#train-title", Static).update(title)

        bar = self.query_one("#train-bar", ProgressBar)
        pct = 100 * s.step / max(s.total_steps, 1)
        bar.update(total=100, progress=pct)

        eta_sec = (s.elapsed_sec / max(s.step, 1)) * max(s.total_steps - s.step, 0)
        best = s.best_so_far or {"reward": float("nan"), "step": 0}
        rew = "n/a" if s.ep_rew_mean is None else f"{s.ep_rew_mean:.2f}"
        stats = (
            f"step {s.step:,} / {s.total_steps:,}   "
            f"ep_rew_mean {rew}   best {best.get('reward', 0):.2f} @ {best.get('step', 0):,}   "
            f"fps {s.fps:,.0f}   elapsed {_fmt_secs(s.elapsed_sec)}   "
            f"ETA {_fmt_secs(eta_sec)}"
        )
        self.query_one("#train-stats", Static).update(stats)
        self.query_one("#train-spark", Static).update(_sparkline(s.recent_rewards))
```

- [ ] **Step 2: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/widgets/training_widget.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): training widget — live progress + sparkline

Polls tui_progress.json every 500ms; renders a ProgressBar plus
step counts, ep_rew_mean, best-so-far, fps, elapsed, ETA, and a
unicode sparkline of recent rewards.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: LogOverlay widget (`tui/widgets/log_overlay.py`)

Full-screen modal — shows the last N lines of either slot's ring buffer.

**Files:**
- Create: `tui/widgets/log_overlay.py`

- [ ] **Step 1: Implement `tui/widgets/log_overlay.py`**

Create `tui/widgets/log_overlay.py`:

```python
"""Modal log overlay — last 200 lines of either subprocess slot.

Toggled by the `l` hotkey on the main app.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, RichLog


class LogOverlay(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "close"), ("l", "dismiss", "close")]

    def __init__(self, title: str, lines: list[str]) -> None:
        super().__init__()
        self._title = title
        self._lines = lines

    def compose(self) -> ComposeResult:
        log = RichLog(highlight=False, markup=False, wrap=True, id="log-body")
        yield Vertical(
            Static(self._title, classes="accent"),
            log,
            id="log-shell",
        )

    def on_mount(self) -> None:
        log = self.query_one("#log-body", RichLog)
        for line in self._lines:
            log.write(line)
```

- [ ] **Step 2: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/widgets/log_overlay.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): log overlay modal

ModalScreen rendering the tail of either subprocess slot's ring
buffer. Bound to the 'l' hotkey on the main app.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: Main `DroneSimApp` + flesh out `__main__.py`

The composition root. Wires everything: tree → form → process manager + training/aux slots; hotkeys; dry-run output; log overlay; quit-with-children prompt.

**Files:**
- Modify: `tui/__main__.py`
- Create: `tui/app.py`

- [ ] **Step 1: Implement `tui/app.py`**

Create `tui/app.py`:

```python
"""DroneSimApp — composes the three panes, training widget, and hotkeys."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from tui import theme
from tui.actions.base import Action
from tui.process.manager import ProcessManager
from tui.widgets.action_form import ActionForm
from tui.widgets.action_tree import ActionTree
from tui.widgets.log_overlay import LogOverlay
from tui.widgets.state_pane import StatePane
from tui.widgets.training_widget import TrainingWidget


class DroneSimApp(App):
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit_with_prompt", "quit"),
        ("ctrl+c", "stop_selected", "stop"),
        ("l", "show_logs", "logs"),
        ("t", "toggle_tensorboard", "tensorboard"),
    ]

    def __init__(self, *, initial_group: str | None = None) -> None:
        super().__init__()
        self._pm = ProcessManager()
        self._initial_group = initial_group

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            ActionTree(id="actions", classes="panel"),
            Vertical(
                ActionForm(id="form"),
                id="form-wrap",
                classes="panel",
            ),
            StatePane(id="state", classes="panel"),
        )
        yield TrainingWidget(id="training", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        # Hide training widget until a training subprocess is alive.
        self.query_one(TrainingWidget).stop_following()
        if self._initial_group:
            tree = self.query_one(ActionTree)
            # Naive: scan flat order, set cursor to first action whose group matches.
            for i, a in enumerate(tree._flat()):
                if a.group.lower() == self._initial_group.lower():
                    tree.cursor = i
                    break

    # ------------------------------------------------------------------
    # Tree → form coupling

    def on_action_tree_selected(self, message: ActionTree.Selected) -> None:
        self.query_one(ActionForm).set_action(message.action)

    # ------------------------------------------------------------------
    # Form → run / dry-run

    def on_action_form_run(self, message: ActionForm.Run) -> None:
        argv = message.action.build_argv(message.values)
        self._launch(message.action, argv)

    def on_action_form_dry_run(self, message: ActionForm.DryRun) -> None:
        rendered = " ".join(shlex.quote(s) for s in message.argv)
        self.push_screen(LogOverlay(title=f"Dry-run: {message.action.label}",
                                    lines=[rendered]))

    def _launch(self, action: Action, argv: list[str]) -> None:
        slot = "training" if action.is_training else "aux"
        if self._pm.is_running(slot):
            self.notify(f"{slot} slot is busy", severity="warning")
            return
        try:
            proc = self._pm.start(slot, argv)
        except Exception as e:  # noqa: BLE001
            self.notify(f"failed to start: {e}", severity="error")
            return
        self.notify(f"▶ {action.label} ({' '.join(argv[:3])} …)")
        if slot == "training":
            run_dir = self._guess_run_dir(action, argv)
            if run_dir is not None:
                self.query_one(TrainingWidget).follow(run_dir)
            self.query_one(ActionTree).training_disabled = True
        self.set_interval(1.0, lambda: self._poll_slot(slot, action))

    def _poll_slot(self, slot: str, action: Action) -> None:
        if self._pm.is_running(slot):
            return
        proc = self._pm._slots[slot]
        if proc is None:
            return
        rc = proc.returncode()
        sev = "information" if rc == 0 else "error"
        self.notify(f"{action.label} exited with code {rc} (press 'l' for logs)",
                    severity=sev)
        if slot == "training":
            self.query_one(TrainingWidget).stop_following()
            self.query_one(ActionTree).training_disabled = False

    def _guess_run_dir(self, action: Action, argv: list[str]) -> Path | None:
        """Heuristic: after launching, look for the newest trial under runs/."""
        from tui.state.scan import latest_trial
        # Poll the filesystem briefly — the run dir is created in the first
        # second of training. Caller is in the Textual event loop; a sync
        # sleep would block, so instead the training widget's first
        # refresh will surface the right dir via latest_trial once it lands.
        return latest_trial().path if latest_trial() is not None else None

    # ------------------------------------------------------------------
    # Hotkeys

    def action_stop_selected(self) -> None:
        # Stop the most-recent running slot — prefer aux if both alive.
        for slot in ("aux", "training"):
            if self._pm.is_running(slot):
                self._pm.stop(slot)
                self.notify(f"sent SIGINT to {slot}")
                return
        self.notify("nothing to stop")

    def action_show_logs(self) -> None:
        # Show whichever slot is alive — preference: aux first.
        for slot in ("aux", "training"):
            proc = self._pm.current(slot)
            if proc is not None:
                self.push_screen(LogOverlay(f"{slot} logs (last 200)", proc.tail(200)))
                return
        self.notify("no active subprocess")

    def action_toggle_tensorboard(self) -> None:
        # Quick-launch / kill TB without going through the form.
        proc = self._pm.current("aux")
        if proc is not None and "tensorboard" in proc.argv[0]:
            self._pm.stop("aux")
            self.notify("tensorboard stopped")
            return
        if self._pm.is_running("aux"):
            self.notify("aux slot busy — cannot start tensorboard", severity="warning")
            return
        try:
            self._pm.start("aux", ["tensorboard", "--logdir", "runs"])
            self.notify("tensorboard at http://localhost:6006")
        except Exception as e:  # noqa: BLE001
            self.notify(f"could not start tensorboard: {e}", severity="error")

    def action_quit_with_prompt(self) -> None:
        if self._pm.is_running("training"):
            self.notify(
                "Training still running — quitting leaves it alive in the background. "
                "Press Ctrl-C first to stop, then q again.",
                severity="warning",
            )
            return
        self._pm.stop_all()
        self.exit()
```

- [ ] **Step 2: Replace `tui/__main__.py` with the real entrypoint**

Replace `tui/__main__.py` contents with:

```python
"""Entrypoint for `python -m tui` and `make ui`."""
from __future__ import annotations

import argparse
import sys

from tui import theme
from tui.app import DroneSimApp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tui",
                                description="Drone-Quidditch interactive launcher")
    p.add_argument("--group", default=None,
                   help="Initial action group to focus (Demo | Train | Eval | Manage)")
    p.add_argument("--ascii", action="store_true",
                   help="Use ASCII fallback glyphs (no Nerd Font required)")
    args = p.parse_args(argv)

    if args.ascii:
        theme.use_ascii()

    DroneSimApp(initial_group=args.group).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Stage and commit (user runs)**

```bash
git -C "$PWD" add tui/app.py tui/__main__.py
git -C "$PWD" commit -m "$(cat <<'EOF'
feat(tui): DroneSimApp + entrypoint

Composes the three panes + docked training widget + footer + log
overlay. Wires tree→form selection, form→ProcessManager dispatch,
training slot lock-out, quit-with-running-training prompt, and the
't' / 'l' / 'ctrl-c' / 'q' hotkeys.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Strip the Makefile + add `ui` and `demo` targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Read the current Makefile end-to-end**

Read `Makefile` (206 lines) so the edits below land on the right lines.

- [ ] **Step 2: Replace the Makefile**

Rewrite `Makefile` (this is large enough that a full Write is cleanest). The new content keeps the header, conda resolution, `help`, `install`, `configs`, `clean`, `test*`, and `camera-test`, then adds `ui` and `demo`.

Replace the entire `Makefile` with:

```makefile
# Quidditch-Sim Makefile — infrastructure & test targets only.
#
# Day-to-day workflow (train / eval / resume / promote / lineage / tensorboard /
# repro / list-runs / demos) lives in the TUI launcher: `make ui` or
# `python -m tui`. The TUI shells out to scripts/*.py; those scripts remain
# directly callable for scripting / CI.

CONDA_ENV ?= uav

# Resolve the conda binary: prefer $CONDA_EXE (set by `conda init`), fall back to PATH.
CONDA := $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null))
ifeq ($(CONDA),)
$(error conda not found — activate a conda shell or set CONDA_EXE)
endif

CONDA_RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python

.PHONY: help ui demo test test-fast test-warm camera-test install configs clean

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "All day-to-day actions (train / eval / promote / …) are in the TUI: 'make ui'."

# ──────────────────────────────────────────────────────────────────────────────
# Interactive launcher

ui: ## 🎛  Interactive launcher (TUI dashboard)
	@$(PYTHON) -m tui

demo: ## 🎮 Open TUI on the Demo group
	@$(PYTHON) -m tui --group Demo

# ──────────────────────────────────────────────────────────────────────────────
# Tests

test: ## ✅ Run all tests (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit

test-warm: ## ✅ Warm-start preserves single-agent behavior  MODEL=<run-name>
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see TUI list-runs)"; exit 1; }; \
	 MODEL="$(MODEL)" $(PYTHON) -m pytest tests/integration/test_warm_start.py

CAM ?= grid
camera-test: ## 🎥 Render hover flight as 2x2 grid → mp4 (CAM=grid|fixed|north|east|south|west|top|fpv|tpv|port|starboard)
	@$(PYTHON) demo/camera_test.py --cam $(CAM)

# ──────────────────────────────────────────────────────────────────────────────
# Setup

install: ## 📦 Create/update the $(CONDA_ENV) conda env + populate config/ from templates
	@$(CONDA) env create -f environment.yml 2>/dev/null || $(CONDA) env update -f environment.yml --prune
	@$(MAKE) --no-print-directory configs
	@echo "Done. Verify with: make test"

configs: ## 🛠  Populate config/ from templates/ (idempotent — never overwrites)
	@mkdir -p config
	@for f in training camera; do \
	   if [ ! -f config/$$f.toml ]; then \
	     cp templates/$$f.toml config/$$f.toml; \
	     echo "config/$$f.toml << templates/$$f.toml."; \
	   else \
	     echo "config/$$f.toml already exists — not overwritten"; \
	   fi; \
	 done

clean: ## 🧹 Remove __pycache__ and .pyc files
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean."
```

- [ ] **Step 3: Verify `make help` shows only the kept targets**

Run: `make help`
Expected: lists `help`, `ui`, `demo`, `test`, `test-fast`, `test-warm`, `camera-test`, `install`, `configs`, `clean` — nothing else.

- [ ] **Step 4: Verify `make test` still passes**

Run: `make test`
Expected: all tests pass (existing + new).

- [ ] **Step 5: Stage and commit (user runs)**

```bash
git -C "$PWD" add Makefile
git -C "$PWD" commit -m "$(cat <<'EOF'
refactor(make): strip 15 day-to-day targets, add ui + demo

train / train-team-* / resume / resume-team / eval / eval-headless /
eval-team / tensorboard / promote / list-runs / lineage / repro
all moved to the TUI launcher ('make ui'). The variable-resolution
block (_LATEST_IN_RUN etc.) is gone — its logic lives in
tui/state/scan.py as the single source of truth. The underlying
scripts/*.py are still directly callable for scripting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: README + CLAUDE.md mentions

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README.md**

Read `README.md` to find the "Usage" / "Quick start" section.

Add (or insert near the top of the usage section) a one-liner:

```markdown
**Interactive launcher:** `make ui` opens a TUI dashboard for training, eval, demos, and run management. Pickers fill in model paths and trial timestamps from disk — no more typing long arguments.
```

- [ ] **Step 2: Update CLAUDE.md**

Read `CLAUDE.md`. Add a line under the project notes section (after the intro paragraph, before the brain section):

```markdown
## Workflow

Day-to-day actions (train / eval / resume / promote / lineage / tensorboard / repro / demos) live in the TUI launcher: `make ui` (or `python -m tui`). The Makefile keeps only infrastructure targets (install / configs / clean / test*).
```

- [ ] **Step 3: Stage and commit (user runs)**

```bash
git -C "$PWD" add README.md CLAUDE.md
git -C "$PWD" commit -m "$(cat <<'EOF'
docs: mention make ui in README and CLAUDE.md

Single-line mentions pointing readers to the TUI launcher as the
canonical entry point for day-to-day workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 25: End-to-end manual verification

Not a code change — a sequence of human checks before merging.

**No files. Manual verification.**

- [ ] **Step 1: Launch the TUI**

Run: `make ui`
Expected: three-pane dashboard renders; "Demo" group at the top of the action tree; cursor on first demo.

- [ ] **Step 2: Demo from the TUI**

Navigate to `hover`, press Enter on Run. The MuJoCo viewer should open. Close the viewer. Notice the TUI shows `▶ hover` running in the footer, then a green exit notification when the viewer closes.

- [ ] **Step 3: Eval picker flow**

Navigate to Eval → eval. Pick a run + trial from the pickers (you should see populated dropdowns drawn from `runs/`). Toggle GUI off; click Dry-run; press `l` to view the constructed argv in the log overlay. Press Esc to dismiss.

- [ ] **Step 4: Brief training run**

Navigate to Train → train single-agent. Set RUN_NAME to `tui_e2e`. Click Run. The training widget docks at the bottom and starts updating (sub-second after the first ~500 env steps).

- [ ] **Step 5: Concurrent eval during training**

While training continues, navigate to Manage → list-runs and Run. It should succeed (aux slot is free). The training widget keeps updating.

- [ ] **Step 6: Stop training**

Press Ctrl-C (the TUI's binding, not the terminal's). Notification confirms SIGINT sent. Training widget transitions to "(no active training)" once the subprocess exits.

- [ ] **Step 7: Tensorboard quick-launch**

Press `t`. Notification shows the URL. Press `t` again to stop.

- [ ] **Step 8: Quit with everything stopped**

Press `q`. The app exits cleanly. Confirm no orphan processes: `pgrep -fa "scripts/train_ppo\|tensorboard"` should return nothing.

- [ ] **Step 9: ASCII fallback smoke**

Run: `python -m tui --ascii`. Glyphs should be bracket-letter forms (`[D]`, `[T]`, …). Press `q` to exit.

- [ ] **Step 10: `make demo` opens TUI on the Demo group**

Run: `make demo`. The TUI opens with the cursor on the first demo entry. Press `q`.

- [ ] **Step 11: Clean up the smoke run**

```bash
rm -rf runs/tui_e2e
```

**No commit for this task.** If anything failed: open an issue or halt and triage before merging.

---

## Self-Review Notes

**Spec coverage check (against `docs/superpowers/specs/2026-05-11-tui-launcher-design.md`):**

- §1 architecture overview → Tasks 6-22 build the package; Task 23 strips Makefile.
- §2 16 actions catalog → Task 13.
- §3 structured progress callback (always-on, schema v1, atomic write) → Tasks 6, 7, 8. Task 7 wires the callback into `build_callbacks` and the team script; Task 8 refactors `train_ppo.py` to go through the same `build_callbacks` (single wire-up site). Unintended-but-aligned side effect: checkpoint `name_prefix` unifies to `"ppo"` for new runs; legacy `"ppo_hoop_"` files remain discoverable via the prefix-agnostic scan regex from Task 11.
- §4 two-slot manager (training-exclusive, aux-concurrent) → Task 15; quit-with-running policy → Task 22.
- §5 action field kinds (text/int/bool/4 pickers) → Tasks 12, 13, 18.
- §6 three-pane layout + docked training widget + idle/busy states → Tasks 17, 18, 19, 20, 22.
- §7 disk scanner → Task 11.
- §8 process manager (SIGINT first, escalate to TERM at 5s) → Task 15.
- §9 visual theme (palette, glyphs, --ascii) → Tasks 10, 16, 22.
- §10 Makefile strip + `ui`/`demo` targets → Task 23.
- §11 file inventory → Covered across Tasks 1-22; promote/list_runs/repro extracted in Tasks 2-4.
- §12 failure modes & tests → Tasks 2, 3, 4, 6, 11, 13, 14 cover the testable rows; manual rows checked in Task 25.
- §13 out-of-scope items → Not implemented (correct).

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in any task body. Every code step contains the actual code.

**Type consistency:** `Action`, `FieldSpec`, `FieldKind`, `ProgressSnapshot`, `ManagedProcess`, `ProcessManager.start(slot, argv, env=, run_dir=)` signatures match between definition (Tasks 12, 14, 15) and consumers (Tasks 13, 18, 20, 22).
