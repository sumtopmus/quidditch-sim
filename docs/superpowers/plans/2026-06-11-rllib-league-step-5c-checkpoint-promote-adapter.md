# RLlib League Step 5c — Checkpoint / Lineage / Promote Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the `.zip`-checkpoint assumption so the `dsim` inventory/lineage/promote/resume/list-runs surface and the `eval_team` entrypoint all work on RLlib **directory** checkpoints — keeping the per-run `.hydra/{config,meta}.yaml` convention, the W&B-artifact + `:prod`-alias registry, and recording snapshot/module provenance.

**Architecture:** A new `core/rllib_checkpoint.py` centralizes the RLlib checkpoint-directory mechanics (find the latest `checkpoint_<N>` dir under a run, locate a single module's sub-checkpoint, load an `RLModule`, detect format). `core/run_listing.py` and `core/promote.py` learn a directory branch alongside their existing `.zip` branch. `scripts/_artifact_io.py` gains `log_rllib_run_artifact` (logs the checkpoint **dir** as a W&B artifact named `<run_name>`, matching `promote`'s lookup) and `resolve_parent` returns the checkpoint dir for dir-artifacts. `scripts/train_rllib.py` logs that artifact post-fit so `promote` can alias it `:prod`. `scripts/eval_team.py` gains an RLlib branch that loads both mains from a checkpoint dir and runs `rllib.eval_battery.rollout_battery` (from **5b**) head-to-head. Promotion records which modules and which source checkpoint were promoted.

**Tech Stack:** Python 3.13, Ray RLlib new API stack (`RLModule.from_checkpoint`, `Algorithm.save().checkpoint.path`), Typer `dsim` CLI, Hydra, W&B artifacts, pytest. Run via `uv run` (sandbox disabled).

**Sequencing — execute AFTER Step 5b (and 5a).** The `eval_team` RLlib port (Task 6) consumes `rllib.eval_battery.rollout_battery(env, act_red, act_blue, …)` and `module_action_fn(module)`, both defined in **5b**. The checkpoint mechanics (Tasks 1–5, 7) are independent of 5a/5b and could land anytime; they are sequenced last because promotion is the *end* of a training campaign and benefits from the eval battery existing to judge "dominates prior prod."

---

## Orientation for the implementer

The `.zip` assumption is threaded through several files. Read these first:

- [core/run_listing.py](../../../core/run_listing.py) — `_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")` (line 19); `_latest_checkpoint` globs `checkpoints/*.zip` (line 98); `resolve_checkpoint` appends `.zip` (line 83) and errors "no .zip checkpoints" (line 89). Drives `dsim list-runs` and `dsim resume`. RLlib checkpoints are **directories** named `checkpoint_<N>`, so these find nothing for RLlib runs today.
- [core/promote.py](../../../core/promote.py) — `promote_run_dir` requires `run_dir / "best_model.zip"` (lines 87–88), copies it (line 107). The W&B side (alias `:prod` + `<run_name>`, `art.save()`, lines 97–102) and the `_wandb_metadata.json` pin (lines 124–132) are **format-agnostic** and reused as-is.
- [scripts/_artifact_io.py](../../../scripts/_artifact_io.py) — `log_run_artifact` (lines 239–294) is SB3-only (`best_model.zip` / `final_model.zip`, `art.add_file`). `resolve_parent` (lines 160–236) hard-codes `best_model.zip` at lines 220–222, 236. The `_WandbURI` / `_parse_wandb_uri` / `_resolve_default_entity_project` helpers are format-agnostic. `log_run_artifact`'s artifact is named `cfg.run_name` (line 275) — `promote._find_run_artifact` looks up `<run_name>:latest`, so the RLlib logger must use the same name.
- [scripts/train_rllib.py](../../../scripts/train_rllib.py) — the Tune entrypoint. `tuner.fit()` (line 60) returns nothing today; it does **not** log any artifact. Task 5 captures the fit results and logs the RLlib dir-artifact.
- [tests/rllib/test_smoke_train.py](../../../tests/rllib/test_smoke_train.py) — the canonical RLlib checkpoint API: `algo.save(str(dir)).checkpoint.path` (line 63) returns a checkpoint **dir**; `Algorithm.from_checkpoint(ckpt)` (line 138) restores; `algo.env_runner.module.keys()` lists module ids. The single-module loader (`RLModule.from_checkpoint`) is what Task 1 wraps.
- [core/eval_core.py](../../../core/eval_core.py) — `run_scenario` (lines 76–246) loads `PPO.load` (line 167) via `OpponentControlledEnv`; SB3-only. `TERMINAL_BUCKETS` + `_classify_terminal` are reused by **5b**'s battery, not changed here.
- [scripts/eval_team.py](../../../scripts/eval_team.py) — Hydra entrypoint delegating to `run_scenario`. Task 6 adds an RLlib branch.
- [tests/scripts/test_promote.py](../../../tests/scripts/test_promote.py) — `_make_run_dir` (lines 11–32) builds a fake run with `best_model.zip` + `.hydra/`; mocks `wandb.Api`. Task 3 adds a dir-checkpoint variant.
- [tests/scripts/test_log_run_artifact.py](../../../tests/scripts/test_log_run_artifact.py) — the `MagicMock` + `patch("wandb.Artifact")` pattern for testing artifact logging offline. Task 4 follows it.

**RLlib checkpoint directory layout** (new API stack), as written by `algo.save(dir)`:
```
<checkpoint_dir>/                       # e.g. runs/<run>/<ts>/tune/<trial>/checkpoint_000010/
├── algorithm_state.pkl
├── learner_group/
│   └── learner/
│       └── rl_module/
│           ├── main_red/               # one subdir per module
│           ├── main_blue/
│           ├── red_pop_v1/  ...        # frozen league snapshots
└── ...
```
A single module loads via `RLModule.from_checkpoint(<checkpoint_dir>/learner_group/learner/rl_module/<module_id>)`.

**Test/run commands** (repo root, sandbox disabled):
- `uv run python -m pytest tests/core/test_rllib_checkpoint.py -v`
- `uv run python -m pytest -m "not slow"` (fast) / `-m slow` (Ray/MuJoCo integration)

---

## File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `core/rllib_checkpoint.py` | Find latest checkpoint dir, module subpath, format detect, load one `RLModule` | **Create** |
| `core/run_listing.py` | Directory-checkpoint fallback in `_latest_checkpoint` / `resolve_checkpoint` | **Modify** |
| `core/promote.py` | Format-aware copy (zip OR checkpoint dir) + provenance | **Modify** |
| `scripts/_artifact_io.py` | `log_rllib_run_artifact` (dir-artifact) + dir-aware `resolve_parent` | **Modify** |
| `scripts/train_rllib.py` | Log the RLlib dir-artifact post-fit | **Modify** |
| `scripts/eval_team.py` | RLlib branch: load both mains, run `rollout_battery` | **Modify** |
| `tests/core/test_rllib_checkpoint.py` | Unit tests for the checkpoint helpers | **Create** |
| `tests/core/test_run_listing_rllib.py` | Dir-checkpoint discovery | **Create** |
| `tests/scripts/test_promote.py` | Dir-checkpoint promote variant | **Modify** |
| `tests/scripts/test_log_run_artifact.py` | `log_rllib_run_artifact` cases | **Modify** |
| `tests/rllib/test_smoke_train.py` | Slow: save → load single module → eval | **Modify** |

---

## Task 1: `core/rllib_checkpoint.py` — checkpoint-directory mechanics

Centralize the RLlib checkpoint-dir logic: find the newest `checkpoint_<N>` directory under a run, build a module's sub-checkpoint path, detect the format, and load a single `RLModule`. The dir-finding and path-building are pure (no Ray import); only `load_rl_module` imports Ray (lazily).

**Files:**
- Create: [core/rllib_checkpoint.py](../../../core/rllib_checkpoint.py)
- Test: [tests/core/test_rllib_checkpoint.py](../../../tests/core/test_rllib_checkpoint.py)

- [ ] **Step 1: Write the failing test (pure helpers)**

Create `tests/core/test_rllib_checkpoint.py`:

```python
"""Step-5c RLlib checkpoint-directory mechanics."""
from __future__ import annotations

from pathlib import Path

import core.rllib_checkpoint as RC


def _make_ckpt(root: Path, rel: str, modules=("main_red", "main_blue")) -> Path:
    ckpt = root / rel
    rl = ckpt / "learner_group" / "learner" / "rl_module"
    for m in modules:
        (rl / m).mkdir(parents=True)
    return ckpt


def test_find_latest_checkpoint_dir_picks_highest_ordinal(tmp_path):
    run = tmp_path / "runs" / "rllib_league" / "20260611_130631"
    _make_ckpt(run, "tune/trial_abc/checkpoint_000010")
    latest = _make_ckpt(run, "tune/trial_abc/checkpoint_000030")
    _make_ckpt(run, "tune/trial_abc/checkpoint_000020")
    assert RC.find_latest_checkpoint_dir(run) == latest.resolve()


def test_find_latest_checkpoint_dir_none_when_absent(tmp_path):
    assert RC.find_latest_checkpoint_dir(tmp_path) is None


def test_is_rllib_checkpoint_detects_module_layout(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010")
    assert RC.is_rllib_checkpoint(ckpt) is True
    assert RC.is_rllib_checkpoint(tmp_path / "nope") is False
    (tmp_path / "best_model.zip").write_bytes(b"x")
    assert RC.is_rllib_checkpoint(tmp_path / "best_model.zip") is False


def test_module_subpath(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010")
    p = RC.module_subpath(ckpt, "main_blue")
    assert p == ckpt / "learner_group" / "learner" / "rl_module" / "main_blue"
    assert p.is_dir()


def test_module_ids_lists_population(tmp_path):
    ckpt = _make_ckpt(tmp_path, "checkpoint_000010",
                      modules=("main_red", "main_blue", "blue_pop_v1"))
    assert RC.module_ids(ckpt) == {"main_red", "main_blue", "blue_pop_v1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_rllib_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.rllib_checkpoint'`.

- [ ] **Step 3: Implement the module**

Create `core/rllib_checkpoint.py`:

```python
"""RLlib directory-checkpoint mechanics (Step 5c).

RLlib new-stack checkpoints are DIRECTORIES, not SB3 .zip files. Tune writes
them under runs/<run>/<ts>/tune/<trial>/checkpoint_<N>/, with one module
sub-checkpoint per id under learner_group/learner/rl_module/<module_id>/.

The dir-finding / path-building / detection helpers are pure (pathlib + regex,
no Ray import) so run_listing / promote / inventory can use them without pulling
Ray. Only load_rl_module imports Ray, lazily.
"""
from __future__ import annotations

import re
from pathlib import Path

_CKPT_DIR_RE = re.compile(r"^checkpoint_(\d+)$")
_RL_MODULE_REL = ("learner_group", "learner", "rl_module")


def find_latest_checkpoint_dir(run_root: Path | str) -> Path | None:
    """The highest-ordinal checkpoint_<N> directory anywhere under `run_root`
    (Tune nests them under tune/<trial>/). None when there are none."""
    run_root = Path(run_root)
    if not run_root.exists():
        return None
    best: tuple[int, Path] | None = None
    for d in run_root.rglob("checkpoint_*"):
        if not d.is_dir():
            continue
        m = _CKPT_DIR_RE.match(d.name)
        if not m:
            continue
        ordinal = int(m.group(1))
        if best is None or ordinal > best[0]:
            best = (ordinal, d)
    return best[1].resolve() if best else None


def module_subpath(checkpoint_dir: Path | str, module_id: str) -> Path:
    """Path to a single module's sub-checkpoint inside a checkpoint dir."""
    return Path(checkpoint_dir).joinpath(*_RL_MODULE_REL, module_id)


def module_ids(checkpoint_dir: Path | str) -> set[str]:
    """Module ids present in a checkpoint dir (the league population)."""
    rl = Path(checkpoint_dir).joinpath(*_RL_MODULE_REL)
    if not rl.is_dir():
        return set()
    return {d.name for d in rl.iterdir() if d.is_dir()}


def is_rllib_checkpoint(path: Path | str) -> bool:
    """True when `path` is an RLlib checkpoint directory (has the rl_module
    layout). False for files (.zip) and non-checkpoint dirs."""
    p = Path(path)
    return p.is_dir() and p.joinpath(*_RL_MODULE_REL).is_dir()


def load_rl_module(checkpoint_dir: Path | str, module_id: str):
    """Load a single frozen RLModule for inference. Ray imported lazily.

    NOTE: the rl_module subpath layout is RLlib-version-specific. If this raises,
    verify the on-disk layout under <checkpoint_dir>/learner_group/learner/
    rl_module/ against the installed Ray version and adjust _RL_MODULE_REL.
    """
    from ray.rllib.core.rl_module.rl_module import RLModule

    sub = module_subpath(checkpoint_dir, module_id)
    if not sub.is_dir():
        raise FileNotFoundError(
            f"no module sub-checkpoint for {module_id!r} at {sub}")
    return RLModule.from_checkpoint(str(sub))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_rllib_checkpoint.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add core/rllib_checkpoint.py tests/core/test_rllib_checkpoint.py
git commit -m "feat(core): RLlib directory-checkpoint mechanics helper"
```

---

## Task 2: `run_listing` — discover directory checkpoints

Teach `_latest_checkpoint` and `resolve_checkpoint` to fall back to an RLlib `checkpoint_<N>` directory when no `.zip` checkpoint is found, so `dsim list-runs` and `dsim resume` surface RLlib runs.

**Files:**
- Modify: [core/run_listing.py](../../../core/run_listing.py)
- Test: [tests/core/test_run_listing_rllib.py](../../../tests/core/test_run_listing_rllib.py)

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_run_listing_rllib.py`:

```python
"""Step-5c: run_listing recognizes RLlib directory checkpoints."""
from __future__ import annotations

from pathlib import Path

from core.run_listing import list_runs, resolve_checkpoint


def _rllib_run(runs: Path, run_name: str, ts: str, ordinal: str) -> Path:
    ckpt = (runs / run_name / ts / "tune" / "trial_x"
            / f"checkpoint_{ordinal}")
    (ckpt / "learner_group" / "learner" / "rl_module" / "main_blue").mkdir(parents=True)
    return ckpt.resolve()


def test_list_runs_finds_rllib_checkpoint_dir(tmp_path):
    runs = tmp_path / "runs"
    latest = _rllib_run(runs, "rllib_league", "20260611_130631", "000030")
    _rllib_run(runs, "rllib_league", "20260611_130631", "000010")
    entries = list_runs(runs_dir=runs)
    assert len(entries) == 1
    assert entries[0].latest_checkpoint == latest


def test_resolve_checkpoint_returns_dir_when_no_zip(tmp_path):
    runs = tmp_path / "runs"
    latest = _rllib_run(runs, "rllib_league", "20260611_130631", "000030")
    trial_dir = runs / "rllib_league" / "20260611_130631"
    assert resolve_checkpoint(trial_dir) == latest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_run_listing_rllib.py -v`
Expected: FAIL — `_latest_checkpoint` only globs `checkpoints/*.zip`, returns None; `resolve_checkpoint` raises "no checkpoints/".

- [ ] **Step 3: Add the directory fallback**

In `core/run_listing.py`, add the import (after line 16):

```python
from core.rllib_checkpoint import find_latest_checkpoint_dir
```

Replace `resolve_checkpoint` (lines 78–90) so a missing `checkpoints/` dir or absent `.zip` falls back to an RLlib checkpoint dir:

```python
def resolve_checkpoint(trial_dir: Path, *, ckpt: str | None = None) -> Path:
    cks = Path(trial_dir) / "checkpoints"
    if ckpt is not None:
        p = cks / (ckpt if ckpt.endswith(".zip") else ckpt + ".zip")
        if not p.exists():
            raise FileNotFoundError(f"no such checkpoint: {p}")
        return p.resolve()
    p = _latest_checkpoint(trial_dir)
    if p is None:
        raise FileNotFoundError(
            f"no .zip or RLlib checkpoints under {trial_dir}")
    return p
```

Replace `_latest_checkpoint` (lines 93–105) to fall back to the RLlib dir:

```python
def _latest_checkpoint(trial_dir: Path) -> Path | None:
    cks = Path(trial_dir) / "checkpoints"
    best: tuple[int, Path] | None = None
    if cks.exists():
        for f in cks.glob("*.zip"):
            m = _CKPT_STEPS_RE.search(f.name)
            if not m:
                continue
            steps = int(m.group(1))
            if best is None or steps > best[0]:
                best = (steps, f)
    if best is not None:
        return best[1].resolve()
    # RLlib runs have no checkpoints/*.zip — find the directory checkpoint.
    return find_latest_checkpoint_dir(trial_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_run_listing_rllib.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm SB3 `.zip` discovery still works**

Run: `uv run python -m pytest tests/dsim/ -k "list_runs or resume or inventory" -v`
Expected: PASS — the `.zip` path is unchanged (the dir fallback only runs when no `.zip` is found).

- [ ] **Step 6: Commit**

```bash
git add core/run_listing.py tests/core/test_run_listing_rllib.py
git commit -m "feat(core): run_listing falls back to RLlib directory checkpoints"
```

---

## Task 3: `promote` — copy a checkpoint directory

Make `promote_run_dir` format-aware: when there's no `best_model.zip` but an RLlib checkpoint dir exists under the run, copy the checkpoint dir into `models/<run>/checkpoint/`, record `checkpoint_format: "rllib"` and the module ids in `_wandb_metadata.json`, and keep the existing W&B alias flow.

**Files:**
- Modify: [core/promote.py](../../../core/promote.py)
- Test: [tests/scripts/test_promote.py](../../../tests/scripts/test_promote.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_promote.py`:

```python
def _make_rllib_run_dir(tmp_path: Path, run_name: str) -> Path:
    """A completed RLlib run: no best_model.zip, a Tune checkpoint dir instead."""
    run_dir = tmp_path / "runs" / run_name / "20260611_130631"
    ckpt = (run_dir / "tune" / "trial_x" / "checkpoint_000030"
            / "learner_group" / "learner" / "rl_module")
    for m in ("main_red", "main_blue", "blue_pop_v1"):
        (ckpt / m).mkdir(parents=True)
    hydra = run_dir / ".hydra"
    hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(
        f"run_name: {run_name}\nwandb:\n  project: drone-quidditch\n")
    (hydra / "meta.yaml").write_text("git_hash: abc123\nparent_chain_total: 0\n")
    return run_dir


def test_promote_copies_rllib_checkpoint_dir(tmp_path: Path) -> None:
    import json
    from unittest.mock import MagicMock, patch
    from scripts.promote import promote_run_dir

    run_dir = _make_rllib_run_dir(tmp_path, "rllib_league_step5")
    models_root = tmp_path / "models"

    art = MagicMock(); art.version = "v0"; art.aliases = ["latest"]
    api = MagicMock(); api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="rllib_league_step5",
                        models_root=models_root)

    dest = models_root / "rllib_league_step5"
    # Checkpoint dir copied (one module subdir suffices to prove the tree).
    assert (dest / "checkpoint" / "learner_group" / "learner"
            / "rl_module" / "main_blue").is_dir()
    assert not (dest / "best_model.zip").exists()
    meta = json.loads((dest / "_wandb_metadata.json").read_text())
    assert meta["checkpoint_format"] == "rllib"
    assert "blue_pop_v1" in meta["module_ids"]
    assert "prod" in meta["aliases"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_promote.py::test_promote_copies_rllib_checkpoint_dir -v`
Expected: FAIL with `FileNotFoundError: ... best_model.zip not found`.

- [ ] **Step 3: Make `promote_run_dir` format-aware**

In `core/promote.py`, add the import (after line 28):

```python
from core.rllib_checkpoint import find_latest_checkpoint_dir, module_ids
```

Replace the source-resolution + copy section of `promote_run_dir` (lines 86–133) with a branch on format:

```python
    run_dir = Path(run_dir).resolve()
    zip_src = run_dir / "best_model.zip"
    rllib_ckpt = None if zip_src.exists() else find_latest_checkpoint_dir(run_dir)
    if not zip_src.exists() and rllib_ckpt is None:
        raise FileNotFoundError(
            f"{zip_src} not found and no RLlib checkpoint under {run_dir} — "
            "was eval triggered, or did training crash early?"
        )

    timestamp = run_dir.name
    entity, project = _resolve_entity_project(run_dir)
    art = _find_run_artifact(run_name, timestamp, entity=entity, project=project)

    aliases = list(art.aliases)
    for alias in ("prod", run_name):
        if alias not in aliases:
            aliases.append(alias)
    art.aliases = aliases
    art.save()

    dest = Path(models_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    checkpoint_format = "zip"
    promoted_modules: list[str] = []
    if zip_src.exists():
        shutil.copy2(zip_src, dest / "best_model.zip")
        copied.append("best_model.zip")
    else:
        # RLlib directory checkpoint: copy the whole tree into models/<run>/checkpoint/.
        checkpoint_format = "rllib"
        promoted_modules = sorted(module_ids(rllib_ckpt))
        ckpt_dest = dest / "checkpoint"
        if ckpt_dest.exists():
            shutil.rmtree(ckpt_dest)
        shutil.copytree(rllib_ckpt, ckpt_dest)
        copied.append("checkpoint/")
    hydra_src = run_dir / ".hydra"
    if hydra_src.exists():
        hydra_dest = dest / ".hydra"
        if hydra_dest.exists():
            shutil.rmtree(hydra_dest)
        shutil.copytree(hydra_src, hydra_dest)
        copied.append(".hydra/")
    src_doc = run_dir / "MODEL.md"
    if src_doc.exists():
        shutil.copy2(src_doc, dest / "MODEL.md")
        copied.append("MODEL.md")

    def _str_or_none(v):
        return v if isinstance(v, str) else None

    metadata = {
        "name":     run_name,
        "version":  art.version,
        "entity":   _str_or_none(getattr(art, "entity", None)),
        "project":  _str_or_none(getattr(art, "project", None)),
        "aliases":  list(art.aliases),
        "logged_by_run_id": f"{run_name}_{timestamp}",
        "checkpoint_format": checkpoint_format,
        "module_ids": promoted_modules,
    }
    (dest / "_wandb_metadata.json").write_text(json.dumps(metadata, indent=2))
    copied.append("_wandb_metadata.json")

    return PromoteResult(
        run_name=run_name,
        wandb_version=art.version,
        wandb_alias="prod",
        target_dir=dest,
        copied_files=copied,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/scripts/test_promote.py -v`
Expected: PASS — the new RLlib test plus all existing `.zip` promote tests (the zip branch is unchanged; existing metadata keys are preserved, with `checkpoint_format: "zip"` and empty `module_ids` added).

- [ ] **Step 5: Commit**

```bash
git add core/promote.py tests/scripts/test_promote.py
git commit -m "feat(promote): copy RLlib checkpoint dir + record provenance"
```

---

## Task 4: `_artifact_io` — log a directory artifact + resolve it

Add `log_rllib_run_artifact(run, run_dir, cfg, checkpoint_dir, parent_chain_total, best_eval_reward)` that logs the checkpoint **dir** (`art.add_dir(..., name="checkpoint")`) as a W&B artifact named `cfg.run_name` (so `promote._find_run_artifact` resolves it), and teach `resolve_parent` to return the checkpoint dir for dir-artifacts.

**Files:**
- Modify: [scripts/_artifact_io.py](../../../scripts/_artifact_io.py)
- Test: [tests/scripts/test_log_run_artifact.py](../../../tests/scripts/test_log_run_artifact.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_log_run_artifact.py`:

```python
def _make_rllib_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "runs" / "rllib_league_step5" / "20260611_130631"
    ckpt = run_dir / "tune" / "trial_x" / "checkpoint_000030"
    (ckpt / "learner_group" / "learner" / "rl_module" / "main_blue").mkdir(parents=True)
    hydra = run_dir / ".hydra"; hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text("run_name: rllib_league_step5\n")
    return run_dir, ckpt


def _rllib_cfg() -> "OmegaConf":
    return OmegaConf.create({
        "run_name": "rllib_league_step5",
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1},
        "env": {"learner_id": "red_0"},
        "init": {"mode": "scratch", "parent": None},
    })


def test_log_rllib_run_artifact_adds_checkpoint_dir(tmp_path: Path) -> None:
    from scripts._artifact_io import log_rllib_run_artifact

    run_dir, ckpt = _make_rllib_run(tmp_path)
    run = MagicMock(); run.disabled = False
    art = MagicMock()
    with patch("wandb.Artifact", return_value=art) as mock_art_cls:
        log_rllib_run_artifact(run=run, run_dir=run_dir, cfg=_rllib_cfg(),
                               checkpoint_dir=ckpt, parent_chain_total=0,
                               best_eval_reward=0.55)

    # Artifact named after run_name (so promote's <run_name>:latest lookup works).
    assert mock_art_cls.call_args.kwargs["name"] == "rllib_league_step5"
    assert mock_art_cls.call_args.kwargs["metadata"]["checkpoint_format"] == "rllib"
    # The checkpoint dir is added (not a single file).
    add_dir_names = [c.kwargs.get("name") for c in art.add_dir.call_args_list]
    assert "checkpoint" in add_dir_names
    aliases = run.log_artifact.call_args.kwargs.get("aliases")
    assert aliases == ["latest"]


def test_log_rllib_run_artifact_noop_when_run_disabled(tmp_path: Path) -> None:
    from scripts._artifact_io import log_rllib_run_artifact

    run_dir, ckpt = _make_rllib_run(tmp_path)
    run = MagicMock(); run.disabled = True
    with patch("wandb.Artifact") as mock_art_cls:
        log_rllib_run_artifact(run=run, run_dir=run_dir, cfg=_rllib_cfg(),
                               checkpoint_dir=ckpt, parent_chain_total=0,
                               best_eval_reward=None)
    mock_art_cls.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_log_run_artifact.py::test_log_rllib_run_artifact_adds_checkpoint_dir -v`
Expected: FAIL with `ImportError: cannot import name 'log_rllib_run_artifact'`.

- [ ] **Step 3: Add `log_rllib_run_artifact`**

Append to `scripts/_artifact_io.py`:

```python
def log_rllib_run_artifact(
    run: Any,
    run_dir: Path,
    cfg: Any,
    checkpoint_dir: Path,
    parent_chain_total: int,
    best_eval_reward: float | None,
) -> None:
    """Log an RLlib run's checkpoint DIRECTORY as a wandb artifact.

    Parallel to log_run_artifact (the SB3 .zip path) but adds the checkpoint dir
    (art.add_dir(..., name="checkpoint")) instead of a single best_model.zip.
    Named cfg.run_name + aliased :latest, so promote's <run_name>:latest lookup
    and the :prod aliasing flow work unchanged. No-op when run is None/disabled.
    """
    if run is None or getattr(run, "disabled", False):
        return
    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return
    art = wandb.Artifact(
        name=str(cfg.run_name),
        type="model",
        metadata={
            "obs_spec":           str(cfg.obs.name),
            "n_stack":            int(cfg.obs.n_stack),
            "learner_id":         cfg.env.get("learner_id"),
            "init_mode":          str(cfg.init.mode),
            "parent_uri":         cfg.init.parent,
            "parent_chain_total": int(parent_chain_total),
            "best_eval_reward":   best_eval_reward,
            "model_kind":         "rllib",
            "checkpoint_format":  "rllib",
        },
    )
    art.add_dir(str(checkpoint_dir), name="checkpoint")
    hydra_dir = run_dir / ".hydra"
    if hydra_dir.exists():
        art.add_dir(str(hydra_dir), name=".hydra")
    model_doc = run_dir / "MODEL.md"
    if model_doc.exists():
        art.add_file(str(model_doc), name="MODEL.md")
    run.log_artifact(art, aliases=["latest"])
```

- [ ] **Step 4: Make `resolve_parent` return the checkpoint dir for dir-artifacts**

In `resolve_parent`, the committed cache-hit block (lines 217–223) and the download return (lines 234–236) assume `best_model.zip`. Add a directory branch. Replace the cache-hit block:

```python
    # Cache hit on committed dir?
    committed = Path(models_root) / parsed.name
    meta = _committed_metadata(committed)
    if meta is not None and meta.get("version") == version and meta.get("name") == parsed.name:
        if metadata_only and (committed / ".hydra" / "config.yaml").exists():
            return committed
        # RLlib dir-artifact: the loadable path is the committed checkpoint/ dir.
        ckpt_dir = committed / "checkpoint"
        if ckpt_dir.is_dir():
            return ckpt_dir if not metadata_only else committed
        cp = committed / "best_model.zip"
        if cp.exists():
            return cp if not metadata_only else committed
        # Metadata pins but no payload — fall through to download.
```

And the full-download return (lines 234–236):

```python
    # Download into the gitignored cache (full).
    cache_dir = Path(models_root) / ".cache" / f"{parsed.name}_{version}"
    art.download(root=str(cache_dir))
    ckpt_dir = cache_dir / "checkpoint"
    if ckpt_dir.is_dir():
        return ckpt_dir
    return cache_dir / "best_model.zip"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/scripts/test_log_run_artifact.py -v`
Expected: PASS — the two new RLlib tests plus all existing SB3 tests (untouched).

- [ ] **Step 6: Commit**

```bash
git add scripts/_artifact_io.py tests/scripts/test_log_run_artifact.py
git commit -m "feat(artifact-io): log + resolve RLlib directory artifacts"
```

---

## Task 5: `train_rllib` — log the dir-artifact post-fit

After `tuner.fit()`, resolve the best checkpoint dir and log it as a W&B artifact (named `cfg.run_name`, aliased `:latest`) so `dsim promote` can later alias it `:prod` — mirroring the SB3 flow where training logs `:latest` and promote adds `:prod`.

**Files:**
- Modify: [scripts/train_rllib.py](../../../scripts/train_rllib.py)
- Test: [tests/scripts/test_train_rllib_artifact.py](../../../tests/scripts/test_train_rllib_artifact.py)

- [ ] **Step 1: Write the failing test (the post-fit logging helper, no Ray)**

Create `tests/scripts/test_train_rllib_artifact.py`:

```python
"""Step-5c: train_rllib logs the RLlib checkpoint dir as a :latest artifact."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _cfg() -> "OmegaConf":
    return OmegaConf.create({
        "run_name": "rllib_league_step5",
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1},
        "env": {"learner_id": "red_0"},
        "init": {"mode": "scratch", "parent": None},
        "tune": {"wandb": {"enabled": True, "project": "drone-quidditch"}},
    })


def test_log_best_checkpoint_logs_latest_artifact(tmp_path: Path) -> None:
    from scripts.train_rllib import _log_best_checkpoint

    run_dir = tmp_path / "runs" / "rllib_league_step5" / "20260611_130631"
    ckpt = run_dir / "tune" / "trial_x" / "checkpoint_000030"
    (ckpt / "learner_group" / "learner" / "rl_module" / "main_blue").mkdir(parents=True)
    (run_dir / ".hydra").mkdir(parents=True)

    fake_run = MagicMock(); fake_run.disabled = False
    with patch("scripts.train_rllib.wandb") as wb, \
         patch("scripts.train_rllib.log_rllib_run_artifact") as log_art:
        wb.init.return_value = fake_run
        _log_best_checkpoint(_cfg(), run_dir)

    log_art.assert_called_once()
    kw = log_art.call_args.kwargs
    assert Path(kw["checkpoint_dir"]).name == "checkpoint_000030"
    fake_run.finish.assert_called_once()


def test_log_best_checkpoint_noop_when_wandb_disabled(tmp_path: Path) -> None:
    from scripts.train_rllib import _log_best_checkpoint

    run_dir = tmp_path / "runs" / "x" / "20260611_130631"
    (run_dir / "tune" / "trial_x" / "checkpoint_000010"
     / "learner_group" / "learner" / "rl_module" / "main_red").mkdir(parents=True)
    cfg = _cfg(); cfg.tune.wandb.enabled = False
    with patch("scripts.train_rllib.log_rllib_run_artifact") as log_art:
        _log_best_checkpoint(cfg, run_dir)
    log_art.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_train_rllib_artifact.py -v`
Expected: FAIL with `ImportError: cannot import name '_log_best_checkpoint'`.

- [ ] **Step 3: Add the post-fit logging helper and call it**

In `scripts/train_rllib.py`, add imports (after line 22):

```python
import wandb

from core.rllib_checkpoint import find_latest_checkpoint_dir
from scripts._artifact_io import log_rllib_run_artifact
```

Add the helper (before `main`):

```python
def _log_best_checkpoint(cfg: DictConfig, run_dir: Path) -> None:
    """Log the run's latest RLlib checkpoint dir as `<run_name>:latest` so
    `dsim promote` can alias it `:prod`. A short standalone wandb run holds the
    artifact (Tune's per-trial runs aren't handed back). No-op when wandb off."""
    if not cfg.tune.wandb.enabled:
        return
    ckpt = find_latest_checkpoint_dir(run_dir)
    if ckpt is None:
        return
    run = wandb.init(project=cfg.tune.wandb.project, name=str(cfg.run_name),
                     job_type="checkpoint", reinit=True)
    try:
        log_rllib_run_artifact(
            run=run, run_dir=run_dir, cfg=cfg, checkpoint_dir=ckpt,
            parent_chain_total=0, best_eval_reward=None)
    finally:
        run.finish()
```

In `main`, after `tuner.fit()` (line 60), add:

```python
    tuner.fit()
    _log_best_checkpoint(cfg, storage_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/scripts/test_train_rllib_artifact.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_rllib.py tests/scripts/test_train_rllib_artifact.py
git commit -m "feat(train-rllib): log the checkpoint dir as :latest post-fit"
```

---

## Task 6: `eval_team` RLlib branch — load both mains, run the battery

Add an RLlib path to `scripts/eval_team.py`: when `cfg.learner.uri` resolves to an RLlib checkpoint dir, load `main_red` + `main_blue` via the Task-1 loader, build the env from the checkpoint's recorded obs blocks, run **5b**'s `rollout_battery` head-to-head, and print the clean battery summary. The SB3 path is unchanged.

**Depends on 5b:** `rllib.eval_battery.rollout_battery` and `module_action_fn`.

**Files:**
- Modify: [scripts/eval_team.py](../../../scripts/eval_team.py)
- Create: [core/rllib_eval.py](../../../core/rllib_eval.py) (the RLlib scenario runner — keeps `eval_team.py` thin)
- Test: [tests/rllib/test_smoke_train.py](../../../tests/rllib/test_smoke_train.py) (slow), [tests/core/test_rllib_eval.py](../../../tests/core/test_rllib_eval.py)

- [ ] **Step 1: Write the failing test (the obs-block reader, pure)**

Create `tests/core/test_rllib_eval.py`:

```python
"""Step-5c: RLlib eval scenario — env spec read from the checkpoint's .hydra."""
from __future__ import annotations

from pathlib import Path

from core.rllib_eval import _env_config_from_run


def test_env_config_from_run_reads_obs_blocks(tmp_path):
    run_dir = tmp_path / "runs" / "rllib_league_step5" / "20260611_130631"
    hydra = run_dir / ".hydra"; hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(
        "obs:\n  name: DUEL_V1_BODY\n  n_stack: 1\n"
        "  blocks: [ANG_VEL, ANG_POS]\n"
        "multiagent:\n  learner_id: red_0\n"
        "curriculum:\n  randomise_start: false\n  episode_seconds: 30.0\n"
        "  red_start_pos: [0.5, 0.0, 2.0]\n")
    ec = _env_config_from_run(run_dir)
    assert ec["obs_blocks"] == ["ANG_VEL", "ANG_POS"]
    assert ec["learner_id"] == "red_0"
    assert ec["team_cfg"]["randomise_red_start"] is False
    assert ec["reward_stack"] is None     # eval builds the default team stack
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_rllib_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.rllib_eval'`.

- [ ] **Step 3: Implement the RLlib scenario runner**

Create `core/rllib_eval.py`:

```python
"""RLlib head-to-head eval (Step 5c).

Loads main_red + main_blue from a checkpoint dir and runs the Step-5b eval
battery head-to-head on the team env. Used by scripts/eval_team.py's RLlib
branch. The env is rebuilt from the checkpoint's recorded .hydra/config.yaml so
obs blocks / team config match training; the reward stack is irrelevant to the
battery's outcome metrics, so eval lets the env build its default team stack.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _env_config_from_run(run_dir: Path) -> dict:
    """env_config for make_team_env, read from the run's .hydra/config.yaml."""
    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    obs = cfg.get("obs") or {}
    ma = cfg.get("multiagent") or {}
    cur = cfg.get("curriculum") or {}
    team_cfg: dict = {}
    if cur:
        team_cfg["randomise_red_start"] = bool(cur.get("randomise_start", True))
        team_cfg["episode_seconds"] = float(cur.get("episode_seconds", 30.0))
        rsp = cur.get("red_start_pos")
        if rsp is not None:
            team_cfg["red_start_pos"] = [float(v) for v in rsp]
    return {
        "learner_id": str(ma.get("learner_id", "red_0")),
        "obs_blocks": list(obs.get("blocks") or []),
        "team_cfg": team_cfg,
        "reward_stack": None,
        "league": None,
    }


def run_rllib_battery(checkpoint_dir: Path, run_dir: Path, *,
                      n_episodes: int, seed: int) -> dict:
    """Load both mains from the checkpoint and run the head-to-head battery."""
    from envs.quidditch.rllib_env import make_team_env
    from core.rllib_checkpoint import load_rl_module
    from rllib.eval_battery import rollout_battery, module_action_fn

    env = make_team_env(_env_config_from_run(run_dir))
    try:
        red = module_action_fn(load_rl_module(checkpoint_dir, "main_red"))
        blue = module_action_fn(load_rl_module(checkpoint_dir, "main_blue"))
        return rollout_battery(env, red, blue, n_episodes=n_episodes, seed=seed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
```

- [ ] **Step 4: Wire the RLlib branch into the entrypoint**

In `scripts/eval_team.py`, add the dispatch in `main` (after building `scenario`, around line 53). Detect an RLlib checkpoint and route to the battery:

```python
    from core.rllib_checkpoint import find_latest_checkpoint_dir, is_rllib_checkpoint
    from pathlib import Path as _Path

    learner_uri = str(learner.uri)
    ckpt = (_Path(learner_uri) if is_rllib_checkpoint(learner_uri)
            else find_latest_checkpoint_dir(_Path(learner_uri)))
    if ckpt is not None:
        from core.rllib_eval import run_rllib_battery
        run_dir = ckpt
        # Walk up to the run-ts dir that holds .hydra/ (checkpoint dirs nest
        # under tune/<trial>/).
        for parent in ckpt.parents:
            if (parent / ".hydra" / "config.yaml").exists():
                run_dir = parent
                break
        metrics = run_rllib_battery(
            ckpt, run_dir,
            n_episodes=int(cfg.eval.n_episodes), seed=int(cfg.eval.seed))
        _print_rllib_summary(metrics)
        return

    result = run_scenario(
        learner_uri=learner_uri,
        scenario=scenario,
        render=bool(cfg.eval.gui),
    )
    _print_summary(result)
```

Add the printer:

```python
def _print_rllib_summary(m: dict) -> None:
    print("\n=== RLlib head-to-head (main_red vs main_blue) ===")
    print(f"  red score-rate:        {m['eval_red_score_rate']:.2%}")
    print(f"  blue prevention-rate:  {m['eval_blue_prevention_rate']:.2%}")
    print(f"  take-down rate:        {m['eval_takedown_rate']:.2%}")
    print(f"  mean episode length:   {m['eval_mean_ep_len']:.1f}")
    print(f"  terminal buckets:")
    for key, v in sorted(m.items()):
        if key.startswith("eval_terminal_") and v:
            print(f"    {key[len('eval_terminal_'):]:24s} {int(v)}")
```

- [ ] **Step 5: Run the pure test to verify it passes**

Run: `uv run python -m pytest tests/core/test_rllib_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Write the slow end-to-end test (train → save → eval battery from disk)**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_eval_team_battery_loads_from_checkpoint_dir(tmp_path):
    """Train a tiny league, save a checkpoint, then load both mains from the
    on-disk checkpoint dir and run the battery — the Step-5c eval path."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project
    from core.rllib_checkpoint import find_latest_checkpoint_dir, load_rl_module
    from rllib.eval_battery import rollout_battery, module_action_fn
    from envs.quidditch.rllib_env import make_team_env

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()
        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    found = find_latest_checkpoint_dir(tmp_path)
    assert found is not None
    env = make_team_env({"learner_id": "red_0",
                         "obs_blocks": list(cfg.obs.blocks),
                         "team_cfg": {"randomise_red_start": True,
                                      "episode_seconds": 1.0},
                         "reward_stack": None})
    try:
        red = module_action_fn(load_rl_module(found, "main_red"))
        blue = module_action_fn(load_rl_module(found, "main_blue"))
        m = rollout_battery(env, red, blue, n_episodes=2, seed=0)
    finally:
        env.close()
    assert 0.0 <= m["eval_red_score_rate"] <= 1.0
    assert abs(m["eval_red_score_rate"] + m["eval_blue_prevention_rate"] - 1.0) < 1e-6
```

- [ ] **Step 7: Run the slow test**

Run: `uv run python -m pytest tests/rllib/test_smoke_train.py::test_eval_team_battery_loads_from_checkpoint_dir -v -m slow`
Expected: PASS. If `load_rl_module` raises on the sub-path, adjust `_RL_MODULE_REL` in `core/rllib_checkpoint.py` (Task 1) to the installed Ray version's on-disk layout — this slow test is the validation gate for that path.

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_team.py core/rllib_eval.py tests/core/test_rllib_eval.py tests/rllib/test_smoke_train.py
git commit -m "feat(eval-team): RLlib branch loads both mains, runs the battery"
```

---

## Task 7: Record snapshot/module provenance for lineage

The migration spec says league lineage "becomes snapshot ancestry within the league rather than a linear ladder." A full snapshot-DAG walker is out of scope for this milestone; this task records the load-bearing provenance — **which modules and which source checkpoint were promoted** — in the committed `_wandb_metadata.json` (already added in Task 3) and surfaces it via `dsim describe-run`, so a promoted RLlib model's contents are auditable. The cross-run `init.parent` walkers (`core/lineage.py`) already no-op correctly on directory paths (their `.is_file()` normalization simply doesn't trigger), so they need no change.

**Files:**
- Modify: [dsim/commands/describe_run.py](../../../dsim/commands/describe_run.py)
- Test: [tests/dsim/test_cli_describe_run.py](../../../tests/dsim/test_cli_describe_run.py)

- [ ] **Step 1: Inspect the current `describe-run` output path**

Run: `uv run dsim describe-run --help`
Then read [dsim/commands/describe_run.py](../../../dsim/commands/describe_run.py) and [tests/dsim/test_cli_describe_run.py](../../../tests/dsim/test_cli_describe_run.py) to confirm how a committed model dir's `_wandb_metadata.json` is currently surfaced (it reads `MODEL.md` / run context).

- [ ] **Step 2: Write the failing test**

Append to `tests/dsim/test_cli_describe_run.py` (mirror the file's existing CLI-invocation pattern — typically `typer.testing.CliRunner` against `dsim.cli.app`):

```python
def test_describe_run_surfaces_rllib_module_provenance(tmp_path):
    import json
    from typer.testing import CliRunner
    from dsim.cli import app

    model_dir = tmp_path / "models" / "rllib_league_step5"
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(
        "run_name: rllib_league_step5\nobs:\n  name: DUEL_V1_BODY\n  n_stack: 1\n")
    (model_dir / "_wandb_metadata.json").write_text(json.dumps({
        "name": "rllib_league_step5", "version": "v0",
        "checkpoint_format": "rllib",
        "module_ids": ["main_red", "main_blue", "blue_pop_v1"],
        "aliases": ["prod", "rllib_league_step5"],
    }))

    result = CliRunner().invoke(app, ["describe-run", str(model_dir)])
    assert result.exit_code == 0
    assert "rllib" in result.stdout
    assert "blue_pop_v1" in result.stdout
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/dsim/test_cli_describe_run.py::test_describe_run_surfaces_rllib_module_provenance -v`
Expected: FAIL — `describe-run` does not currently print `checkpoint_format` / `module_ids`.

- [ ] **Step 4: Surface the provenance**

In `dsim/commands/describe_run.py`, where the committed model's metadata is rendered, read `_wandb_metadata.json` and, when `checkpoint_format == "rllib"`, print the format and the module ids. Add (adapting to the file's existing print/echo style):

```python
import json as _json

def _print_rllib_provenance(model_dir) -> None:
    meta_path = model_dir / "_wandb_metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = _json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return
    if meta.get("checkpoint_format") != "rllib":
        return
    typer.echo(f"checkpoint format: rllib")
    mods = meta.get("module_ids") or []
    if mods:
        typer.echo(f"modules:           {', '.join(mods)}")
```

Call `_print_rllib_provenance(model_dir)` from the command's render path (alongside the existing metadata output). Match the module's existing `typer.echo`/import conventions.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/dsim/test_cli_describe_run.py -v`
Expected: PASS (new test + existing describe-run tests).

- [ ] **Step 6: Commit**

```bash
git add dsim/commands/describe_run.py tests/dsim/test_cli_describe_run.py
git commit -m "feat(dsim): describe-run surfaces RLlib checkpoint module provenance"
```

> **Scoped-down decision (no silent cap):** Full league snapshot-DAG lineage (per-snapshot parent/iteration/eval edges, a `dsim lineage --tree` view) is **deferred** to a follow-up. This task records the promoted checkpoint's module set as provenance — enough to audit what a `:prod` RLlib model contains — but does not reconstruct the within-run snapshot ancestry tree. Note this explicitly in the brain when closing Step 5.

---

## Task 8: Full-suite verification

- [ ] **Step 1: Fast suite**

Run: `uv run python -m pytest -m "not slow"`
Expected: PASS — all prior fast tests plus the new `test_rllib_checkpoint.py`, `test_run_listing_rllib.py`, `test_rllib_eval.py`, `test_train_rllib_artifact.py`, and the promote/artifact/describe-run additions.

- [ ] **Step 2: Slow suite**

Run: `uv run python -m pytest -m slow`
Expected: PASS — existing RLlib smoke tests plus `test_eval_team_battery_loads_from_checkpoint_dir`. (5 macOS-render tests may fail headless — environmental.)

- [ ] **Step 3: Manual smoke of the dsim surface against a real RLlib run** (user-run)

Once a real `+experiment=rllib_league_step5` run has produced `runs/rllib_league_step5/<ts>/tune/<trial>/checkpoint_<N>/`:
```bash
uv run dsim list-runs --filter rllib_league_step5     # shows the dir checkpoint
uv run dsim promote runs/rllib_league_step5/<ts>      # copies checkpoint/ + aliases :prod
uv run python -m scripts.eval_team +eval_team=default \
    learner.uri=models/rllib_league_step5/checkpoint eval.n_episodes=20
```
Expected: `list-runs` shows the checkpoint dir; `promote` writes `models/rllib_league_step5/checkpoint/` + `_wandb_metadata.json` with `checkpoint_format: "rllib"`; `eval_team` prints the head-to-head battery summary.

- [ ] **Step 4: Stop for review**

Ready to merge into develop (`--no-ff`) **after** the user confirms the manual smoke above on a real run. Do not merge before that confirmation (behavior verified before committing the merge).

---

## Self-review checklist (run before handing off)

1. **Spec coverage (component #5 / B4):**
   - "RLlib checkpoints are directories, not .zip" → `core/rllib_checkpoint.py` (Task 1) + the directory branches in `run_listing` (Task 2), `promote` (Task 3), `_artifact_io` (Task 4). ✅
   - "Re-point `dsim promote`/`lineage`/`inventory`" → promote (Task 3), list-runs/resume (Task 2), describe-run provenance (Task 7). `inventory` keys on `.hydra/config.yaml` (not `.zip`), so it already discovers RLlib model dirs — verified in the explorer pass, no change needed. ✅
   - "Keep `.hydra/{config,meta}.yaml` + W&B-artifact + `:prod` alias" → promote reuses the format-agnostic alias flow; `log_rllib_run_artifact` keeps the `<run_name>` naming so `:prod` aliasing works (Tasks 3–5). ✅
   - "Port `eval_team` to load RLModule checkpoints, head-to-head on the ParallelEnv (no OCE)" → Task 6 (`core/rllib_eval.py` + the entrypoint branch, reusing 5b's `rollout_battery`). ✅
   - "Lineage edges become snapshot ancestry" → **partially**: module provenance recorded + surfaced (Task 7); full snapshot-DAG walker explicitly deferred and flagged (no silent cap). ⚠️ documented
   - B4 "promote a main_* to :prod when it dominates the battery" → the battery (5b) + the RLlib promote/artifact path (Tasks 3–5) provide the mechanics; the *dominance comparison* gate is a human/CLI judgment on the battery output, not automated here. Noted.
2. **Type consistency:** `find_latest_checkpoint_dir(run_root) -> Path|None`, `module_subpath(dir, id) -> Path`, `module_ids(dir) -> set[str]`, `is_rllib_checkpoint(path) -> bool`, `load_rl_module(dir, id) -> RLModule` (Task 1) used identically in Tasks 2/3/4/6. `log_rllib_run_artifact(run, run_dir, cfg, checkpoint_dir, parent_chain_total, best_eval_reward)` (Task 4) called with matching kwargs in Task 5. `rollout_battery` / `module_action_fn` signatures match 5b. ✅
3. **No placeholders:** every code step shows complete code; commands have expected output. The `_RL_MODULE_REL` subpath layout is flagged (Tasks 1, 6) as the one Ray-version-specific spot, gated by the Task-6 slow test. ✅
4. **Backward compatibility:** every `.zip` branch is preserved — `run_listing` (zip first, dir fallback), `promote` (zip branch unchanged), `_artifact_io` (`log_run_artifact` untouched; `resolve_parent` adds a dir branch ahead of the zip return). Existing SB3 tests stay green (Tasks 2/3/4 re-run them). ✅
```
