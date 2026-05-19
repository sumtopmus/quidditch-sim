# ML Infra Part 3 — Slice 1 (Engine + CLI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the data layer and CLI surface for the eval framework + checkpoint UX: importable `core/` services (`inventory`, `obs_compat`, `eval_core`, `eval_report`), Hydra migration of `eval_team.py` + `eval_ppo.py`, new `eval_battery.py` entrypoint with local Markdown/CSV/JSONL reports, and a unified `dsim` Typer CLI for inspection/dispatch. Reshape the Makefile to chores-only (plus `make train` for one-off muscle memory). Land this PR before Slice 2 (TUI) starts.

**Architecture:** `core/` holds importable Python functions; `scripts/` and `dsim/` are thin CLIs that call them. Hydra owns composable runs (training/eval/battery/sweep); Typer owns inspection/dispatch (inventory/lineage/promote/describe-run/resume/sweep). No business logic in CLI files. Inventory and obs-preflight are pure read-only — no W&B network calls in the default path. Battery dispatch is in-process (one `wandb.init` per battery, scenarios run sequentially).

**Tech Stack:** Python 3.11+, Hydra 1.3+, OmegaConf, Stable-Baselines3, Typer + Rich (new), pytest, MuJoCo, W&B SDK.

**Spec:** [docs/superpowers/specs/2026-05-18-ml-infra-part-3-design.md](docs/superpowers/specs/2026-05-18-ml-infra-part-3-design.md)

**Branch:** `feature/ml-infra-part-3` (worktree off develop, already created).

**Out of scope for this plan:** The Slice 2 TUI controller — that ships as a separate plan (`2026-05-18-ml-infra-part-3-slice-2-tui.md`) after this PR merges into develop.

---

## Repository conventions (read before starting)

- Conda env is `uav`. Activate before running anything: `conda activate uav`. The Makefile's `$(PYTHON)` already wraps with `conda run --no-capture-output -n uav python`.
- All commits GPG-signed by default (`git config commit.gpgsign true` is set).
- Commit type taxonomy: `feat:` (new), `refactor:` (move/restructure, no behavior change), `fix:` (bug), `test:` (tests only), `docs:` (docs only), `build:` (deps / packaging), `chore:` (true housekeeping).
- Test layout mirrors source: `tests/<package>/<module>/test_<name>.py`.
- Slow / integration tests gated by `@pytest.mark.slow`; `make test-fast` skips them.
- `WANDB_MODE=disabled` is set in `tests/conftest.py` so the suite stays offline.
- Single-agent canary: `pytest tests/scripts/test_train_smoke_wandb_disabled.py -v` plus the integration canary `tests/integration/test_scoring_canary.py` (asserts `step 434 / reward 7.3837`). Team canary: `tests/envs/quidditch/test_team_env_canary.py`. Both must remain byte-identical through Slice 1.

## Task index

1. Add Typer + dsim package scaffolding (build/scaffolding)
2. Lift `_load_run_context` → `core/run_context.py`
3. `core/inventory.py` — `ModelInfo` + `inventory()` + `load_model_doc()`
4. `core/obs_compat.py` — `preflight()` (no model weights loaded)
5. Refactor `scripts/lineage.py` → importable `core/lineage.py`
6. Refactor `scripts/promote.py` → importable `core/promote.py`
7. `core/run_listing.py` — `list_runs()`
8. Extract `core/eval_core.py` from `scripts/eval_team.py`
9. Hydra config groups for eval (`conf/eval`, `conf/learner`, `conf/eval_team`, `conf/eval_ppo`)
10. Hydra migration: `scripts/eval_team.py`
11. Hydra migration: `scripts/eval_ppo.py`
12. `core/eval_report.py` — MD + CSV + JSONL writer
13. `scripts/eval_battery.py` + `conf/eval_battery/{default,quick,ladder,scripted_only}.yaml`
14. `dsim` subcommands: read-only (inventory, obs-preflight, obs-specs, describe-run)
15. `dsim` subcommands: dispatch (lineage, list-runs, resume, promote, sweep)
16. `scripts/train.py` — obs-preflight warn-then-raise guard rail
17. Makefile reshape (chores-only + `make train`)
18. Docs (README + CLAUDE.md + brain/index.md update)

---

## Task 1: Typer + dsim package scaffolding

**Files:**
- Create: `dsim/__init__.py`
- Create: `dsim/__main__.py`
- Create: `dsim/cli.py`
- Create: `dsim/commands/__init__.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `tests/dsim/__init__.py`
- Create: `tests/dsim/test_cli_help_smoke.py`

- [ ] **Step 1: Add typer to requirements.txt**

Append to `requirements.txt`:

```
# CLI framework for dsim (inspection/dispatch).  Pulls in `rich` transitively;
# we also import rich directly for table rendering.
typer>=0.12,<1.0
```

`rich` is already in requirements (used by SB3 progress bars + our `ResumeProgressCallback`), so no duplicate.

- [ ] **Step 2: Add `dsim = "dsim.cli:app"` script entry point to pyproject.toml**

Read current `pyproject.toml` (it currently has only `[tool.pytest.ini_options]`). Replace with:

```toml
[project]
name = "drone-sim"
version = "0.1.0"
description = "MuJoCo-based RL project: drones playing simplified Quidditch."
requires-python = ">=3.11"

[project.scripts]
dsim = "dsim.cli:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["core*", "envs*", "scripts*", "dsim*", "demo*", "tui*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: integration tests — real episode loops, full canaries, warm-start. `make test-fast` skips these via -m 'not slow'.",
]
```

- [ ] **Step 3: Write the failing CLI smoke test**

Create `tests/dsim/__init__.py` as empty file. Then create `tests/dsim/test_cli_help_smoke.py`:

```python
"""Smoke test: dsim CLI mounts and --help renders without raising.

If a subcommand's import fails at module level (e.g. missing import,
typer decorator typo), this catches it before deployment.
"""
from __future__ import annotations

from typer.testing import CliRunner

from dsim.cli import app


def test_dsim_help_renders_without_error() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    # The top-level help should mention at least one subcommand.
    assert "Usage:" in result.output


def test_dsim_module_invocation_works() -> None:
    """`python -m dsim --help` must also work, not just the installed entry point."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "dsim", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
```

- [ ] **Step 4: Run the test, confirm it fails**

```bash
conda run -n uav python -m pytest tests/dsim/test_cli_help_smoke.py -v
```

Expected: `ModuleNotFoundError: No module named 'dsim'` (or similar).

- [ ] **Step 5: Create the dsim package**

`dsim/__init__.py` — empty.

`dsim/__main__.py`:

```python
"""`python -m dsim` entry point — delegates to the Typer app in dsim.cli."""
from dsim.cli import app

if __name__ == "__main__":
    app()
```

`dsim/cli.py`:

```python
"""Top-level Typer app for dsim.

Subcommands are registered here as they're implemented (see dsim/commands/).
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="dsim",
    help="Drone-sim inspection and dispatch CLI.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


# Subcommands are added in later tasks.  Each subcommand module exposes a
# `register(app)` function (or a Typer sub-app object) and is wired here.


def main() -> None:
    """Entry point for `dsim` script (defined in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
```

`dsim/commands/__init__.py` — empty.

- [ ] **Step 6: Install the package and re-run the smoke test**

```bash
conda run -n uav pip install -e .
conda run -n uav python -m pytest tests/dsim/test_cli_help_smoke.py -v
```

Expected: PASS.

- [ ] **Step 7: Verify `dsim` binary is on PATH**

```bash
conda run -n uav dsim --help
```

Expected: prints `Usage: dsim [OPTIONS] COMMAND [ARGS]...` and exits 0.

- [ ] **Step 8: Run the full test suite as a regression gate**

```bash
make test-fast
```

Expected: PASS (no existing test should change).

- [ ] **Step 9: Commit**

```bash
git add requirements.txt pyproject.toml dsim/ tests/dsim/
git commit -m "build(dsim): scaffolding for Typer CLI package

Add typer>=0.12 to requirements; expose dsim as a project script and
\`python -m dsim\` entry point; empty subcommand directory + smoke test
that catches subcommand import failures before deployment.

Subcommands land in later tasks of the Slice 1 plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Lift `_load_run_context` to `core/run_context.py`

The model-doc generator's `_load_run_context` function (at `scripts/_render_model_doc.py:21`) loads `.hydra/{config,meta,hydra}.yaml` + `_wandb_metadata.json` into one dict. The inventory (Task 3) and obs-compat preflight (Task 4) both need this loader; lift it into `core/` so they don't import from `scripts/`.

**Files:**
- Create: `core/run_context.py`
- Modify: `scripts/_render_model_doc.py` (re-import from core/)
- Create: `tests/core/test_run_context.py`

- [ ] **Step 1: Write the failing test**

`tests/core/test_run_context.py`:

```python
"""Behavior contract for core.run_context.load_run_context.

Re-uses the same fixtures as tests/scripts/test_render_model_doc.py; the
old `scripts._render_model_doc._load_run_context` re-imports from core so
both call sites get the same loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _write_hydra_dir(run_dir: Path, *, cfg: dict, meta: dict | None = None) -> None:
    hdir = run_dir / ".hydra"
    hdir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), hdir / "config.yaml")
    if meta is not None:
        OmegaConf.save(OmegaConf.create(meta), hdir / "meta.yaml")


def test_load_run_context_reads_hydra_config(tmp_path: Path) -> None:
    from core.run_context import load_run_context

    _write_hydra_dir(
        tmp_path,
        cfg={"run_name": "x", "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3}},
        meta={"final_steps": 1_000_000, "parent_chain_total": 5_000_000},
    )
    ctx = load_run_context(tmp_path)

    assert ctx["cfg"]["run_name"] == "x"
    assert ctx["cfg"]["obs"]["name"] == "DUEL_V2_WORLD"
    assert ctx["meta"]["final_steps"] == 1_000_000


def test_load_run_context_missing_config_raises(tmp_path: Path) -> None:
    from core.run_context import load_run_context
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_run_context(tmp_path)


def test_load_run_context_optional_fields_default_to_none(tmp_path: Path) -> None:
    from core.run_context import load_run_context
    _write_hydra_dir(tmp_path, cfg={"run_name": "x"})
    ctx = load_run_context(tmp_path)
    assert ctx["meta"] is None
    assert ctx.get("wandb_meta") is None


def test_scripts_render_model_doc_uses_core_loader() -> None:
    """The legacy `_load_run_context` symbol on scripts._render_model_doc
    must still work — it's now a re-import from core.run_context."""
    from scripts import _render_model_doc as legacy
    from core.run_context import load_run_context
    assert legacy._load_run_context is load_run_context
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
conda run -n uav python -m pytest tests/core/test_run_context.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.run_context'`.

- [ ] **Step 3: Create `core/run_context.py`**

Copy the body of `_load_run_context` from `scripts/_render_model_doc.py:21` (read that function in full first; it's ~40 lines including the hydra.yaml + _wandb_metadata.json loads with their interpolation-handling).

```python
"""Loader for a run's persisted context: .hydra/{config,meta,hydra}.yaml +
_wandb_metadata.json.

Public API:
    load_run_context(run_dir) -> dict

Used by:
    - scripts._render_model_doc (model-doc rendering; re-imports this)
    - core.inventory             (per-model row construction)
    - core.obs_compat            (parent obs spec lookup without loading weights)

No filesystem writes, no wandb calls.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def load_run_context(run_dir: Path) -> dict[str, Any]:
    """Gather config + meta + hydra-choices + wandb-meta into one dict.

    Required: `.hydra/config.yaml`.  All other inputs optional; missing ones
    surface as `None` in the returned ctx.
    """
    hdir = Path(run_dir) / ".hydra"
    cfg_path = hdir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"required input missing: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    meta_path = hdir / "meta.yaml"
    meta = (
        OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
        if meta_path.exists() else None
    )

    hydra_yaml_path = hdir / "hydra.yaml"
    # resolve=False: hydra.yaml carries interpolations like `${run_name}` that
    # reference the parent cfg; resolving them here would crash.
    hydra_meta = (
        OmegaConf.to_container(OmegaConf.load(hydra_yaml_path), resolve=False)
        if hydra_yaml_path.exists() else None
    )

    wandb_meta_path = Path(run_dir) / "_wandb_metadata.json"
    wandb_meta: dict[str, Any] | None = None
    if wandb_meta_path.exists():
        try:
            wandb_meta = json.loads(wandb_meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not read %s: %s", wandb_meta_path, e)
            wandb_meta = None

    return {
        "run_dir": Path(run_dir),
        "cfg": cfg,
        "meta": meta,
        "hydra_meta": hydra_meta,
        "wandb_meta": wandb_meta,
    }


# Legacy alias: scripts._render_model_doc historically named this
# `_load_run_context`; keep the leading-underscore name working.
_load_run_context = load_run_context
```

- [ ] **Step 4: Update `scripts/_render_model_doc.py` to re-import from core**

Find the existing `_load_run_context` function at line 21 and replace lines 21-60 (the whole function) with a re-import at the module top:

```python
# At the top of the file's imports section, add:
from core.run_context import load_run_context as _load_run_context  # noqa: F401  re-export for back-compat
```

Then delete the original `def _load_run_context(...)` body. All other references (`ctx = _load_run_context(run_dir)` in `render_model_doc`) keep working because the name is still in scope.

- [ ] **Step 5: Run the tests, confirm pass**

```bash
conda run -n uav python -m pytest tests/core/test_run_context.py tests/scripts/test_render_model_doc.py -v
```

Expected: PASS for both (no behavior change in the model-doc renderer).

- [ ] **Step 6: Run model-doc on a real promoted model as a sanity check**

```bash
conda run -n uav python -m scripts.render_model_doc --run-dir models/ppo_hoop_blue_4_20260511_202612
```

Expected: prints `wrote .../MODEL.md (NNN bytes)` and exits 0; the rendered MODEL.md should be byte-identical to the one on disk (diff against `git show develop:models/ppo_hoop_blue_4_20260511_202612/MODEL.md`).

- [ ] **Step 7: Commit**

```bash
git add core/run_context.py scripts/_render_model_doc.py tests/core/test_run_context.py
git commit -m "refactor(run-context): lift _load_run_context to core/run_context.py

The model-doc renderer's run-context loader is now importable from
core/, so core.inventory and core.obs_compat can use it without
importing scripts/_render_model_doc (which is properly private).

scripts/_render_model_doc.py re-imports the symbol under its legacy
name (_load_run_context) — zero behavior change for callers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `core/inventory.py` — `ModelInfo` + `inventory()` + `load_model_doc()`

**Files:**
- Create: `core/inventory.py`
- Create: `tests/core/test_inventory.py`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_inventory.py`:

```python
"""Behavior contract for core.inventory.

inventory() walks models/<name>/ (vendored) and optionally models/.cache/<name>/
(downloaded), reads each model's .hydra/config.yaml + .hydra/meta.yaml +
_wandb_metadata.json via core.run_context.load_run_context, and returns a
list[ModelInfo] sorted by short_name then trial timestamp descending.

Legacy migrated models (.hydra/config.yaml hand-written by
scripts/migrate_legacy_models.py) are handled the same way; their obs_spec
field may be missing or sparse, which inventory() reports as obs_spec="?"
(string sentinel, not None — keeps the table renderer simple).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _make_model_dir(
    root: Path,
    name: str,
    *,
    obs_name: str = "DUEL_V2_WORLD",
    n_stack: int = 3,
    parent: str | None = None,
    final_steps: int = 1_000_000,
    chain_total: int = 1_000_000,
    wandb_alias: str | None = "prod",
    wandb_version: str | None = "v0",
    has_model_doc: bool = False,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "best_model.zip").write_bytes(b"\x50\x4b\x03\x04stub-zip")
    h = d / ".hydra"
    h.mkdir()
    cfg = {
        "run_name": name.rsplit("_", 2)[0],
        "obs": {"name": obs_name, "n_stack": n_stack},
        "init": {"mode": "scratch" if parent is None else "pretrain",
                 "parent": parent} if parent is not None else {"mode": "scratch"},
    }
    OmegaConf.save(OmegaConf.create(cfg), h / "config.yaml")
    OmegaConf.save(OmegaConf.create({
        "final_steps": final_steps,
        "parent_chain_total": chain_total,
    }), h / "meta.yaml")
    if wandb_alias is not None:
        import json
        (d / "_wandb_metadata.json").write_text(json.dumps({
            "name": name.rsplit("_", 2)[0],
            "alias": wandb_alias,
            "version": wandb_version,
        }))
    if has_model_doc:
        (d / "MODEL.md").write_text("# MODEL: " + name + "\n")
    return d


def test_inventory_lists_vendored_models(tmp_path: Path) -> None:
    from core.inventory import inventory

    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir, "ppo_hoop_red_1_20260506_103058",
                    obs_name="DUEL_V1_BODY", n_stack=1)

    rows = inventory(models_dir=models_dir)

    names = [r.name for r in rows]
    assert "ppo_hoop_blue_4_20260511_202612" in names
    assert "ppo_hoop_red_1_20260506_103058" in names
    assert len(rows) == 2


def test_inventory_short_name_strips_timestamp(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    rows = inventory(models_dir=models_dir)
    assert rows[0].short_name == "blue_4"


def test_inventory_excludes_cache_by_default(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir / ".cache", "ppo_hoop_blue_5_v3")

    rows = inventory(models_dir=models_dir)
    assert all(r.source == "vendored" for r in rows)
    assert len(rows) == 1


def test_inventory_include_cache_picks_up_downloaded(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir / ".cache", "ppo_hoop_blue_5_v3")

    rows = inventory(models_dir=models_dir, include_cache=True)
    sources = {r.source for r in rows}
    assert sources == {"vendored", "cache"}


def test_inventory_carries_parent_chain_and_wandb_metadata(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_5_20260514_120000",
                    parent="wandb://ppo_hoop_blue_4:prod",
                    chain_total=15_000_000,
                    wandb_alias="prod", wandb_version="v3")
    rows = inventory(models_dir=models_dir)
    r = rows[0]
    assert r.parent == "wandb://ppo_hoop_blue_4:prod"
    assert r.parent_chain_total == 15_000_000
    assert r.wandb_alias == "prod"
    assert r.wandb_version == "v3"


def test_inventory_detects_model_doc_presence(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "with_doc", has_model_doc=True)
    _make_model_dir(models_dir, "without_doc", has_model_doc=False)
    rows = {r.name: r for r in inventory(models_dir=models_dir)}
    assert rows["with_doc"].has_model_doc is True
    assert rows["without_doc"].has_model_doc is False


def test_inventory_skips_dirs_without_hydra_config(tmp_path: Path) -> None:
    """A models/<x>/ dir with no .hydra/config.yaml is not a model — skip it."""
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "real_model")
    (models_dir / "junk").mkdir()
    (models_dir / "junk" / "README.txt").write_text("not a model")
    rows = inventory(models_dir=models_dir)
    assert [r.name for r in rows] == ["real_model"]


def test_inventory_sorted_by_short_name_then_timestamp_desc(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260501_000000")
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir, "ppo_hoop_red_1_20260506_103058")
    rows = inventory(models_dir=models_dir)
    # blue_4 entries come first (sort by short_name), latest timestamp first.
    assert rows[0].name == "ppo_hoop_blue_4_20260511_202612"
    assert rows[1].name == "ppo_hoop_blue_4_20260501_000000"
    assert rows[2].name == "ppo_hoop_red_1_20260506_103058"


def test_load_model_doc_reads_when_present(tmp_path: Path) -> None:
    from core.inventory import inventory, load_model_doc
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_x", has_model_doc=True)
    [info] = inventory(models_dir=models_dir)
    assert load_model_doc(info) == "# MODEL: ppo_hoop_blue_4_x\n"


def test_load_model_doc_returns_none_when_absent(tmp_path: Path) -> None:
    from core.inventory import inventory, load_model_doc
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_x", has_model_doc=False)
    [info] = inventory(models_dir=models_dir)
    assert load_model_doc(info) is None
```

- [ ] **Step 2: Run, confirm fail**

```bash
conda run -n uav python -m pytest tests/core/test_inventory.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.inventory'`.

- [ ] **Step 3: Implement `core/inventory.py`**

```python
"""Read-only model catalog.

Walks `models/*/` (committed-vendored) and optionally `models/.cache/*/`
(wandb downloads), reads each model's `.hydra/{config,meta}.yaml` +
`_wandb_metadata.json` via `core.run_context.load_run_context`, and returns
a sorted list of `ModelInfo` rows.

Public API:
    inventory(models_dir=Path("models"), include_cache=False) -> list[ModelInfo]
    load_model_doc(info) -> str | None
    load_run_context(info) -> dict        # re-export for callers that want the raw dict

No W&B network calls; offline-survivable.  Used by `dsim inventory`, the TUI
inventory pane, and `core.obs_compat.preflight` (parent lookup).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.run_context import load_run_context as _load_run_context

MODELS_DIR = Path("models")
_CACHE_SUBDIR = ".cache"


@dataclass(frozen=True)
class ModelInfo:
    name: str                  # e.g. "ppo_hoop_blue_4_20260511_202612"
    short_name: str            # e.g. "blue_4"
    obs_spec: str              # e.g. "DUEL_V2_WORLD"; "?" when missing
    n_stack: int               # 1 if not declared
    parent: str | None         # init.parent URI or path; None for scratch
    parent_chain_total: int    # 0 when meta.yaml missing
    final_steps: int | None
    source: Literal["vendored", "cache"]
    path: Path                 # absolute path to the model dir
    wandb_alias: str | None    # "prod" / "<run_name>" / None
    wandb_version: str | None  # "v<N>" pinned in _wandb_metadata.json
    has_model_doc: bool        # True if MODEL.md exists on disk


def inventory(
    models_dir: Path = MODELS_DIR,
    include_cache: bool = False,
) -> list[ModelInfo]:
    """Enumerate vendored (and optionally cached) promoted models."""
    models_dir = Path(models_dir)
    rows: list[ModelInfo] = []

    if models_dir.exists():
        rows.extend(_scan(models_dir, source="vendored", skip_dirs={_CACHE_SUBDIR}))

    if include_cache:
        cache_dir = models_dir / _CACHE_SUBDIR
        if cache_dir.exists():
            rows.extend(_scan(cache_dir, source="cache"))

    # Sort: short_name asc, then trial timestamp suffix desc within each short_name.
    def _sort_key(r: ModelInfo) -> tuple[str, str]:
        # Trailing timestamp like "_20260511_202612" sorts as a desc tiebreaker
        # if we negate via lexicographic reversal.  Simpler: invert by using
        # a tuple of (short_name, -timestamp-as-int) when parseable.
        ts = _trail_timestamp_int(r.name)
        return (r.short_name, _invert_int(ts))

    rows.sort(key=_sort_key)
    return rows


def load_model_doc(info: ModelInfo) -> str | None:
    """Return the contents of `<model_dir>/MODEL.md`, or None if absent."""
    p = info.path / "MODEL.md"
    if not p.exists():
        return None
    return p.read_text()


def load_run_context(info: ModelInfo) -> dict[str, Any]:
    """Re-load the full `core.run_context.load_run_context` dict for an info row."""
    return _load_run_context(info.path)


# --- internals --------------------------------------------------------------

def _scan(
    base: Path,
    *,
    source: Literal["vendored", "cache"],
    skip_dirs: set[str] | None = None,
) -> list[ModelInfo]:
    skip_dirs = skip_dirs or set()
    out: list[ModelInfo] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in skip_dirs:
            continue
        if not (d / ".hydra" / "config.yaml").exists():
            continue
        try:
            ctx = _load_run_context(d)
        except Exception:
            continue
        out.append(_row_from_ctx(d, ctx, source=source))
    return out


def _row_from_ctx(
    d: Path,
    ctx: dict[str, Any],
    *,
    source: Literal["vendored", "cache"],
) -> ModelInfo:
    cfg = ctx["cfg"]
    meta = ctx.get("meta") or {}
    wandb_meta = ctx.get("wandb_meta") or {}

    obs_block = cfg.get("obs") or {}
    obs_spec = str(obs_block.get("name", "?")) if obs_block else "?"
    n_stack = int(obs_block.get("n_stack", 1)) if obs_block else 1

    init_block = cfg.get("init") or {}
    parent = init_block.get("parent")
    parent = str(parent) if parent else None

    return ModelInfo(
        name=d.name,
        short_name=_short_name(d.name),
        obs_spec=obs_spec,
        n_stack=n_stack,
        parent=parent,
        parent_chain_total=int(meta.get("parent_chain_total", 0) or 0),
        final_steps=(int(meta["final_steps"]) if meta.get("final_steps") is not None else None),
        source=source,
        path=d.resolve(),
        wandb_alias=wandb_meta.get("alias"),
        wandb_version=wandb_meta.get("version"),
        has_model_doc=(d / "MODEL.md").exists(),
    )


def _short_name(full: str) -> str:
    """`ppo_hoop_blue_4_20260511_202612` → `blue_4` (drops `ppo_hoop_` prefix
    and `_YYYYMMDD_HHMMSS` trailing timestamp).
    """
    s = full
    if s.startswith("ppo_hoop_"):
        s = s[len("ppo_hoop_"):]
    # Strip a trailing `_YYYYMMDD_HHMMSS` if present.
    parts = s.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit() \
            and len(parts[2]) == 6 and parts[2].isdigit():
        s = parts[0]
    return s


def _trail_timestamp_int(name: str) -> int:
    """Concatenated `YYYYMMDDHHMMSS` int, or 0 if not parseable."""
    parts = name.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit() \
            and len(parts[2]) == 6 and parts[2].isdigit():
        return int(parts[1] + parts[2])
    return 0


def _invert_int(n: int) -> int:
    return -n  # negative makes "larger ts = smaller key" → desc within group
```

- [ ] **Step 4: Run the tests, fix any failures**

```bash
conda run -n uav python -m pytest tests/core/test_inventory.py -v
```

Expected: PASS.

- [ ] **Step 5: Smoke against real `models/`**

```bash
conda run -n uav python -c "from core.inventory import inventory
for r in inventory():
    print(f'{r.short_name:14s} {r.obs_spec:20s} n_stack={r.n_stack} parent_chain_total={r.parent_chain_total:>10}  {r.name}')"
```

Expected: 7 rows (blue_1, blue_4, fixed_start ×2, rand_start ×2, red_1) printed in short_name asc, ts desc order. Spot-check that `blue_4` shows `DUEL_V2_WORLD n_stack=3` and `red_1` shows `DUEL_V1_BODY n_stack=1`.

- [ ] **Step 6: Run full test suite**

```bash
make test-fast
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add core/inventory.py tests/core/test_inventory.py
git commit -m "feat(inventory): core.inventory.inventory() — read-only model catalog

Returns list[ModelInfo] for vendored (default) and optionally cached
models, reading each .hydra/{config,meta}.yaml + _wandb_metadata.json
via core.run_context.load_run_context.

Sorted by short_name asc, then trial timestamp suffix desc.  No W&B
network calls — offline-survivable.  Will back the dsim inventory CLI,
the TUI inventory pane, and core.obs_compat.preflight's parent lookup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `core/obs_compat.py` — `preflight()` (no model weights loaded)

The existing `check_obs_compat` in `scripts/_train_common.py:193` reads from `run_info.toml` via `read_obs_spec` (legacy) and `sys.exit(2)`s on mismatch with a printed diff. We need a *function-shaped* version that:
- Reads the parent's obs spec from `.hydra/config.yaml` (Hydra-era) — not `run_info.toml`
- Returns a structured `PreflightReport` instead of printing + exiting
- Optionally accepts wandb:// URIs and downloads only `.hydra/` from the artifact
- Does NOT load model weights

**Files:**
- Create: `core/obs_compat.py`
- Modify: `scripts/_train_common.py` (factor `_is_compat` + `_render_diff` into core so both paths share the diff logic)
- Create: `tests/core/test_obs_compat.py`
- Modify: `scripts/_artifact_io.py` — add `metadata_only` kwarg to `resolve_parent` so wandb-URI preflight can fetch `.hydra/` without weights

- [ ] **Step 1: Write the failing test**

`tests/core/test_obs_compat.py`:

```python
"""Behavior contract for core.obs_compat.preflight.

preflight(parent_uri, child_obs_name, child_n_stack) -> PreflightReport
- Resolves parent_uri to its obs spec WITHOUT loading the model
- Returns a structured report (no print, no sys.exit)
- compatible=False ⇒ check_obs_compat would strict-raise
- surgery_required=True ⇒ would need init.mode=warm_start
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _make_parent(tmp_path: Path, *, obs_name: str, n_stack: int) -> Path:
    d = tmp_path / "parent"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "parent",
        "obs": {"name": obs_name, "n_stack": n_stack},
    }), d / ".hydra" / "config.yaml")
    return d


def test_preflight_matched_specs_returns_compatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    assert report.compatible is True
    assert report.surgery_required is False
    assert report.parent_spec_name == "DUEL_V2_WORLD"
    assert report.child_spec_name == "DUEL_V2_WORLD"


def test_preflight_n_stack_mismatch_is_incompatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=1)
    assert report.compatible is False
    assert report.surgery_required is True
    assert report.parent_n_stack == 3
    assert report.child_n_stack == 1


def test_preflight_spec_change_is_surgery_required(tmp_path: Path) -> None:
    """Different obs specs (e.g. V1_BODY → V2_WORLD) need warm_start."""
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V1_BODY", n_stack=1)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    assert report.compatible is False
    assert report.surgery_required is True
    assert any(d.status in ("frame_changed", "removed", "added") for d in report.diff)


def test_preflight_handles_v3_body_ego(tmp_path: Path) -> None:
    """DUEL_V3_BODY_EGO is a recognized spec (added 2026-05-18)."""
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V3_BODY_EGO", n_stack=1)
    report = preflight(str(parent), child_obs_name="DUEL_V3_BODY_EGO", child_n_stack=1)
    assert report.compatible is True


def test_preflight_unknown_parent_spec_raises(tmp_path: Path) -> None:
    import pytest
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="NO_SUCH_SPEC", n_stack=1)
    with pytest.raises(KeyError, match="NO_SUCH_SPEC"):
        preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)


def test_preflight_missing_parent_config_raises(tmp_path: Path) -> None:
    import pytest
    from core.obs_compat import preflight
    with pytest.raises(FileNotFoundError):
        preflight(str(tmp_path / "does-not-exist"),
                  child_obs_name="DUEL_V2_WORLD", child_n_stack=3)


def test_preflight_diff_columns_aligned_for_compatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    # All blocks should be matched.
    assert all(d.status == "matched" for d in report.diff)
```

- [ ] **Step 2: Run, confirm fail**

```bash
conda run -n uav python -m pytest tests/core/test_obs_compat.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.obs_compat'`.

- [ ] **Step 3: Implement `core/obs_compat.py`**

```python
"""Obs-spec compatibility preflight — function-shaped, no sys.exit, no print.

Resolves a parent run's obs spec via .hydra/config.yaml (Hydra-era) WITHOUT
loading the model weights, and runs the existing _is_compat logic from
scripts/_train_common.py against the child spec.  Returns a structured
PreflightReport so callers (dsim obs-preflight CLI, the TUI experiment
picker, scripts/train.py's pretrain guard rail) can decide how to render.

Public API:
    preflight(parent_uri, child_obs_name, child_n_stack=1) -> PreflightReport

For wandb:// URIs, downloads ONLY .hydra/ from the artifact (not the
weights) via core.obs_compat._download_parent_hydra (uses
scripts._artifact_io.resolve_parent with metadata_only=True).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf

from envs.quidditch.obs_spec import SPEC_BY_NAME, ObsSpec
from core.run_context import load_run_context

DiffStatus = Literal["matched", "frame_changed", "removed", "added"]


@dataclass(frozen=True)
class ObsBlockDiff:
    block: str
    dim: int
    parent_frame: str | None
    child_frame: str | None
    status: DiffStatus


@dataclass(frozen=True)
class PreflightReport:
    compatible: bool                # would NOT strict-raise
    surgery_required: bool          # init.mode=warm_start would be needed
    diff: list[ObsBlockDiff]
    parent_spec_name: str
    child_spec_name: str
    parent_n_stack: int
    child_n_stack: int


def preflight(
    parent_uri: str,
    child_obs_name: str,
    child_n_stack: int = 1,
) -> PreflightReport:
    """Run obs-compat check between parent_uri and (child_obs_name, child_n_stack).

    parent_uri may be:
      - a local path to a run dir (containing .hydra/config.yaml)
      - a local path to a best_model.zip (the parent dir is used)
      - a wandb:// URI (only .hydra/ is downloaded, not weights)
    """
    parent_dir = _resolve_parent_dir_metadata_only(parent_uri)
    ctx = load_run_context(parent_dir)
    parent_obs = (ctx["cfg"].get("obs") or {})
    if not parent_obs:
        raise FileNotFoundError(
            f"parent {parent_dir}/.hydra/config.yaml has no `obs` block — "
            "cannot preflight"
        )

    parent_spec_name = str(parent_obs["name"])
    parent_n_stack = int(parent_obs.get("n_stack", 1))

    parent_spec = SPEC_BY_NAME[parent_spec_name]   # KeyError on unknown
    child_spec = SPEC_BY_NAME[child_obs_name]

    diff = _build_diff(parent_spec, child_spec)
    n_stack_ok = parent_n_stack == child_n_stack
    all_matched = all(d.status == "matched" for d in diff)
    compatible = n_stack_ok and all_matched
    surgery_required = not compatible  # warm_start handles either kind of mismatch

    return PreflightReport(
        compatible=compatible,
        surgery_required=surgery_required,
        diff=diff,
        parent_spec_name=parent_spec_name,
        child_spec_name=child_obs_name,
        parent_n_stack=parent_n_stack,
        child_n_stack=child_n_stack,
    )


# --- internals --------------------------------------------------------------

def _resolve_parent_dir_metadata_only(parent_uri: str) -> Path:
    """Return the parent run dir, downloading only .hydra/ for wandb URIs."""
    if parent_uri.startswith(("wandb://", "wandb-artifact://")):
        from scripts._artifact_io import resolve_parent
        # metadata_only=True: download .hydra/ and _wandb_metadata.json only;
        # do NOT fetch best_model.zip.  Added to resolve_parent in Step 6.
        return resolve_parent(parent_uri, metadata_only=True)

    p = Path(parent_uri)
    if p.is_file():
        p = p.parent
    if not (p / ".hydra" / "config.yaml").exists():
        raise FileNotFoundError(f"no .hydra/config.yaml under {p}")
    return p


def _build_diff(parent: ObsSpec, child: ObsSpec) -> list[ObsBlockDiff]:
    """Column-by-column block alignment between two specs."""
    out: list[ObsBlockDiff] = []
    p_by_name = {b.name: b for b in parent.blocks}
    c_by_name = {b.name: b for b in child.blocks}

    seen: set[str] = set()
    for cb in child.blocks:
        pb = p_by_name.get(cb.name)
        if pb is None:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=None, child_frame=cb.frame,
                status="added",
            ))
            continue
        seen.add(cb.name)
        if pb.dim != cb.dim:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=pb.frame, child_frame=cb.frame,
                status="frame_changed",
            ))
        elif pb.frame != cb.frame:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=pb.frame, child_frame=cb.frame,
                status="frame_changed",
            ))
        else:
            out.append(ObsBlockDiff(
                block=cb.name, dim=cb.dim,
                parent_frame=pb.frame, child_frame=cb.frame,
                status="matched",
            ))
    for pb in parent.blocks:
        if pb.name in seen:
            continue
        if pb.name not in c_by_name:
            out.append(ObsBlockDiff(
                block=pb.name, dim=pb.dim,
                parent_frame=pb.frame, child_frame=None,
                status="removed",
            ))
    return out
```

- [ ] **Step 4: Extend `scripts/_artifact_io.py:resolve_parent` with `metadata_only` kwarg**

Read the current `resolve_parent` body (`scripts/_artifact_io.py:160-205`). Then modify the signature and the download path:

```python
def resolve_parent(
    uri_or_path: str | Path,
    models_root: Path = Path("models"),
    *,
    metadata_only: bool = False,
) -> Path:
    """Resolve a parent reference to a local Path.

    When metadata_only=True:
      - For filesystem paths, returns the parent run dir (containing .hydra/).
      - For wandb URIs, downloads ONLY .hydra/ + _wandb_metadata.json (not
        best_model.zip) into models/.cache/<name>_v<N>_meta/ and returns
        the run dir.

    When metadata_only=False (default), the existing behavior: download/
    return the full checkpoint path (best_model.zip).
    """
    # ... existing logic for non-wandb paths ...
    s = str(uri_or_path)
    if not _is_wandb_uri(s):
        p = Path(s)
        if metadata_only and p.is_file():
            return p.parent
        return p

    # ... wandb path: existing alias→version resolution ...
    # When metadata_only=True: skip best_model.zip download; use
    # art.file('.hydra/config.yaml').download(root=meta_cache) (or
    # equivalent: download just the .hydra/ subtree).  Return the dir.

    # ... existing committed-cache hit path ...
    # If metadata_only=True and the committed dir has .hydra/, return committed
    # dir even if best_model.zip is missing.

    # ... existing download fallback ...
    # If metadata_only=True, use a separate cache subdir (e.g. <name>_v<N>_meta/)
    # and download only the .hydra/ subtree via art.download(path_prefix=".hydra/")
    # (wandb's Artifact.download supports a `path_prefix` filter).
```

The full diff is mechanical — read the existing function in detail, then thread `metadata_only` through to:
1. The "is filesystem path" branch (return parent dir if a file path is passed).
2. The "wandb URI, committed cache hit" branch (return committed dir even when `best_model.zip` is missing).
3. The "wandb URI, download" branch (use `art.download(path_prefix=".hydra/")` into a separate cache subdir).

Write a unit test alongside in `tests/scripts/test_artifact_resolve.py` (extend the existing file if present, else create new) asserting `resolve_parent(filesystem_path, metadata_only=True)` returns the parent dir, and a wandb-URI test using monkeypatched `wandb.Api`.

- [ ] **Step 5: Run obs-compat tests**

```bash
conda run -n uav python -m pytest tests/core/test_obs_compat.py tests/scripts/test_artifact_resolve.py -v
```

Expected: PASS.

- [ ] **Step 6: Smoke against real models**

```bash
conda run -n uav python -c "
from core.obs_compat import preflight
# blue_4 is DUEL_V2_WORLD n_stack=3.  Trying to feed it to a child that
# expects the same spec should be compatible.
r = preflight('models/ppo_hoop_blue_4_20260511_202612', 'DUEL_V2_WORLD', 3)
print('compatible:', r.compatible, 'surgery_required:', r.surgery_required)
# Trying to feed a V1_BODY parent into a V2_WORLD child should require surgery.
r = preflight('models/ppo_hoop_red_1_20260506_103058', 'DUEL_V2_WORLD', 3)
print('compatible:', r.compatible, 'surgery_required:', r.surgery_required)
print('diff:')
for d in r.diff:
    print(' ', d.status, d.block, d.dim, d.parent_frame, '->', d.child_frame)
"
```

Expected: first call prints `compatible: True surgery_required: False`; second call prints `compatible: False surgery_required: True` with a non-empty diff.

- [ ] **Step 7: Commit**

```bash
git add core/obs_compat.py scripts/_artifact_io.py tests/core/test_obs_compat.py tests/scripts/test_artifact_resolve.py
git commit -m "feat(obs-compat): core.obs_compat.preflight() — read-only diff

Function-shaped obs-compat: reads parent's obs spec from .hydra/config.yaml
(Hydra-era, not the legacy run_info.toml path) and returns a structured
PreflightReport (no print, no sys.exit).  For wandb:// URIs, downloads
only .hydra/ via resolve_parent(metadata_only=True) — does not fetch
best_model.zip.

resolve_parent gains metadata_only kwarg.  Used by dsim obs-preflight,
the TUI experiment picker badge, and scripts/train.py's warn-then-raise
guard rail (added in Task 16).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Refactor `scripts/lineage.py` → importable `core/lineage.py`

The current `scripts/lineage.py` mixes CLI dispatch with the actual walker logic. Lift the walkers into `core/lineage.py` so `dsim lineage` can call them directly.

**Files:**
- Create: `core/lineage.py`
- Modify: `scripts/lineage.py` (becomes thin CLI wrapper)
- Create: `tests/core/test_lineage.py`

- [ ] **Step 1: Write the failing test**

`tests/core/test_lineage.py`:

```python
"""Behavior contract for core.lineage walkers.

walk_chain_local(start_path) -> list[LineageNode]
  Reads _wandb_metadata.json chain in models/<run>/ (offline-survivable).

walk_chain_wandb(target_uri) -> list[LineageNode]
  Uses wandb.Api().artifact().logged_by().used_artifacts().  Network required.
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf


def _make_chain(tmp_path: Path, names: list[str], parents: list[str | None]) -> None:
    """Build a synthetic model chain where models[i]'s parent points to models[i-1]."""
    models = tmp_path / "models"
    for name, parent in zip(names, parents):
        d = models / name
        (d / ".hydra").mkdir(parents=True)
        cfg = {"run_name": name, "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3}}
        if parent is not None:
            cfg["init"] = {"mode": "pretrain", "parent": parent}
        OmegaConf.save(OmegaConf.create(cfg), d / ".hydra" / "config.yaml")
        OmegaConf.save(OmegaConf.create({"final_steps": 1_000_000}),
                       d / ".hydra" / "meta.yaml")


def test_walk_chain_local_walks_back_via_parent_field(tmp_path: Path) -> None:
    from core.lineage import walk_chain_local
    _make_chain(
        tmp_path,
        names=["A", "B", "C"],
        parents=[None, str(tmp_path / "models" / "A"), str(tmp_path / "models" / "B")],
    )
    chain = walk_chain_local(tmp_path / "models" / "C")
    assert [n.name for n in chain] == ["C", "B", "A"]


def test_walk_chain_local_stops_at_scratch(tmp_path: Path) -> None:
    from core.lineage import walk_chain_local
    _make_chain(tmp_path, names=["A", "B"], parents=[None, str(tmp_path / "models" / "A")])
    chain = walk_chain_local(tmp_path / "models" / "B")
    assert chain[-1].parent is None
    assert chain[-1].name == "A"


def test_walk_chain_local_handles_missing_intermediate(tmp_path: Path) -> None:
    """If a parent path doesn't exist, the chain truncates with a sentinel."""
    from core.lineage import walk_chain_local
    _make_chain(
        tmp_path,
        names=["B"],
        parents=[str(tmp_path / "models" / "VANISHED_PARENT")],
    )
    chain = walk_chain_local(tmp_path / "models" / "B")
    assert chain[0].name == "B"
    assert chain[-1].truncated is True
```

- [ ] **Step 2: Run, confirm fail**

```bash
conda run -n uav python -m pytest tests/core/test_lineage.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `core/lineage.py`**

```python
"""Walk a run's pretrain ancestry — importable walkers + URI parsing helpers.

Two walkers, same return shape:

  walk_chain_local(start)  -> list[LineageNode]
    Reads parent links from each ancestor's .hydra/config.yaml `init.parent`.
    Offline-survivable; truncates if an intermediate path is missing.

  walk_chain_wandb(target) -> list[LineageNode]
    Uses wandb.Api().artifact().logged_by().used_artifacts() for the native
    artifact DAG.  Richer (sees un-vendored intermediates) but needs
    network + credentials.

CLI dispatch lives in scripts/lineage.py (a thin wrapper) and
dsim/commands/lineage.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class LineageNode:
    name: str
    path: Path | None        # local path if known
    parent: str | None       # parent URI/path (None ⇒ scratch)
    final_steps: int | None
    parent_chain_total: int | None
    truncated: bool = False  # True when chain was cut due to missing intermediate


def walk_chain_local(start_path: Path | str) -> list[LineageNode]:
    """Walk back via init.parent in each .hydra/config.yaml, ending at scratch."""
    start = Path(start_path)
    if start.is_file():
        start = start.parent

    chain: list[LineageNode] = []
    cursor: Path | None = start
    safety = 64

    while cursor is not None and safety > 0:
        safety -= 1
        node = _node_from_path(cursor)
        if node is None:
            # Truncated chain: previous step had a parent we can't follow.
            if chain:
                last = chain[-1]
                chain[-1] = LineageNode(
                    name=last.name, path=last.path, parent=last.parent,
                    final_steps=last.final_steps,
                    parent_chain_total=last.parent_chain_total,
                    truncated=True,
                )
            break
        chain.append(node)
        cursor = _next_parent_path(node)
    return chain


def walk_chain_wandb(target_uri: str) -> list[LineageNode]:
    """Walk an artifact DAG via wandb.Api.

    The implementation mirrors scripts/lineage.py:walk_chain_wandb exactly;
    moved here verbatim plus the LineageNode wrapping at the end.
    """
    # Implementation: read scripts/lineage.py's existing walk_chain_wandb
    # body and adapt it to yield LineageNode rows.  The original is ~80 lines;
    # it qualifies the URI via _WandbURI.for_api, hits art.logged_by(), then
    # walks art.used_artifacts() iteratively, building one row per step.
    raise NotImplementedError("port from scripts/lineage.py:walk_chain_wandb")


def _node_from_path(p: Path) -> LineageNode | None:
    cfg_path = p / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    init = cfg.get("init") or {}
    parent = init.get("parent")
    parent = str(parent) if parent else None
    meta_path = p / ".hydra" / "meta.yaml"
    meta = (OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
            if meta_path.exists() else None) or {}
    return LineageNode(
        name=str(cfg.get("run_name", p.name)),
        path=p.resolve(),
        parent=parent,
        final_steps=meta.get("final_steps"),
        parent_chain_total=meta.get("parent_chain_total"),
        truncated=False,
    )


def _next_parent_path(node: LineageNode) -> Path | None:
    if node.parent is None:
        return None
    if node.parent.startswith(("wandb://", "wandb-artifact://")):
        # Walker A doesn't follow wandb URIs; caller should switch to walker B.
        return None
    p = Path(node.parent)
    if p.is_file():
        p = p.parent
    return p if p.exists() else None
```

- [ ] **Step 4: Port `walk_chain_wandb` body from `scripts/lineage.py`**

Read `scripts/lineage.py:walk_chain_wandb` (the existing function — around lines 100-200). Copy its body into `core/lineage.py:walk_chain_wandb`, adapting the return shape to `list[LineageNode]`. The existing `_resolve_uri_to_local_name`, `_WandbURI`-related helpers should move with it (or be imported from `scripts/_artifact_io.py` where they likely live).

- [ ] **Step 5: Thin `scripts/lineage.py` to be a CLI wrapper**

Replace `scripts/lineage.py` body with:

```python
"""CLI wrapper around core.lineage walkers.

For new code, prefer:
    dsim lineage --target X [--local|--both]
This script remains for back-compat with `make lineage` (removed in
Task 17) and the `python -m scripts.lineage` invocation pattern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.lineage import walk_chain_local, walk_chain_wandb


def _render(chain) -> str:
    lines = []
    for i, node in enumerate(chain):
        indent = "  " * i
        suffix = "  (truncated)" if node.truncated else ""
        lines.append(f"{indent}{node.name}  steps={node.final_steps}  "
                     f"chain_total={node.parent_chain_total}{suffix}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", help="Filesystem path or wandb:// URI")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="Walker A only")
    g.add_argument("--both", action="store_true", help="Both walkers side-by-side")
    args = p.parse_args()

    if args.local:
        chain = walk_chain_local(args.target)
        print(_render(chain))
        return 0

    if args.both:
        print("--- local ---")
        print(_render(walk_chain_local(args.target)))
        print("\n--- wandb ---")
        try:
            print(_render(walk_chain_wandb(args.target)))
        except Exception as e:
            print(f"(wandb walker failed: {e})", file=sys.stderr)
        return 0

    # Default: wandb if URI / network, else local.
    if args.target.startswith(("wandb://", "wandb-artifact://")):
        try:
            print(_render(walk_chain_wandb(args.target)))
            return 0
        except Exception as e:
            print(f"(wandb failed: {e}; falling back to local)", file=sys.stderr)
    print(_render(walk_chain_local(args.target)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run lineage tests + a real-data smoke**

```bash
conda run -n uav python -m pytest tests/core/test_lineage.py -v
conda run -n uav python -m scripts.lineage --local models/ppo_hoop_blue_4_20260511_202612
```

Expected: tests pass; smoke prints a chain ending at `rand_start_20260505_174509` or earlier.

- [ ] **Step 7: Commit**

```bash
git add core/lineage.py scripts/lineage.py tests/core/test_lineage.py
git commit -m "refactor(lineage): lift walkers into core.lineage; scripts/lineage.py thin wrapper

core.lineage exposes walk_chain_local() and walk_chain_wandb() as
importable functions returning list[LineageNode].  dsim lineage (Task 15)
imports these directly; scripts/lineage.py becomes a thin argparse
wrapper for back-compat with the old python -m scripts.lineage form.

No behavior change in either walker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Refactor `scripts/promote.py` → importable `core/promote.py`

**Files:**
- Create: `core/promote.py`
- Modify: `scripts/promote.py` (becomes thin CLI wrapper)
- Modify: `tests/scripts/test_promote.py` (adapt imports)

- [ ] **Step 1: Read current `scripts/promote.py`** — note `_resolve_run_name`, `_resolve_entity_project`, `main`, and the wandb-alias + repo-copy + `_wandb_metadata.json` write sequence.

- [ ] **Step 2: Create `core/promote.py` with `promote_run()` function**

```python
"""Promote a training run's best model to canonical / vendored status.

Two-step (unchanged from scripts/promote.py):

  1. Wandb side: alias the artifact this run logged with `prod` + `<run_name>`,
     save() to persist.
  2. Repo side: copy best_model.zip + .hydra/ + MODEL.md (if present) into
     models/<run_name>/, write _wandb_metadata.json pinning the IMMUTABLE
     version (`v3`, not `prod`).

Public API:
    promote_run(run_dir, *, alias="prod") -> PromoteResult

The script `scripts/promote.py` becomes a thin CLI wrapper around this.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf


@dataclass(frozen=True)
class PromoteResult:
    run_name: str
    wandb_version: str
    wandb_alias: str
    target_dir: Path
    copied_files: list[str]


def promote_run(run_dir: Path, *, alias: str = "prod") -> PromoteResult:
    run_dir = Path(run_dir)
    run_name = _resolve_run_name(run_dir)

    # Copy the entire body of the existing scripts/promote.py main(), adapted
    # to take run_dir + alias as args (instead of reading argparse) and to
    # return PromoteResult instead of printing.  The W&B side
    # (artifact.aliases.append + artifact.save()) and the repo side (shutil
    # copy + _wandb_metadata.json write) are unchanged.
    #
    # The MODEL.md copy step (already in develop after PR #12) is preserved.

    # ... full body ported from scripts/promote.py ...
    raise NotImplementedError("port from scripts/promote.py:main")


def _resolve_run_name(run_dir: Path) -> str:
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return str(cfg.run_name)
```

The bulk is a mechanical port. The full body lives in the current `scripts/promote.py:main()`. Copy lines 50-150 (or wherever the alias + copy logic is) into the body of `promote_run`, parameterizing on `run_dir` and `alias`. Return a `PromoteResult` instead of printing.

- [ ] **Step 3: Update `scripts/promote.py` to be a thin wrapper**

```python
"""CLI wrapper around core.promote.promote_run.

For new code, prefer:
    dsim promote <run-name> [--alias prod]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.promote import promote_run


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Path to runs/<name>/<ts>/")
    p.add_argument("--alias", default="prod", help="W&B alias to set (default: prod)")
    args = p.parse_args()
    result = promote_run(args.run_dir, alias=args.alias)
    print(f"promoted {result.run_name} ({result.wandb_version}, alias={result.wandb_alias})")
    print(f"  copied {len(result.copied_files)} files into {result.target_dir}")
    for f in result.copied_files:
        print(f"    {f}")
    print("git add models/ && git commit -m 'model: promote ...' to vendor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run promote tests + smoke**

```bash
conda run -n uav python -m pytest tests/scripts/test_promote.py -v
```

Expected: PASS (the existing test exercises the W&B side via mocks; it should keep passing because the logic moved verbatim).

Optional smoke (only if you have a real run + wandb creds):

```bash
conda run -n uav python -m scripts.promote runs/<some_unpromoted_run>/<ts>
```

- [ ] **Step 5: Commit**

```bash
git add core/promote.py scripts/promote.py tests/scripts/test_promote.py
git commit -m "refactor(promote): lift promote_run() into core.promote

core.promote.promote_run(run_dir, alias='prod') is the importable shape;
scripts/promote.py is now a thin argparse wrapper.  Used by dsim promote
(Task 15) and the TUI Promote task (Slice 2).

Behavior unchanged: same W&B alias + repo copy + _wandb_metadata.json
write sequence, same MODEL.md copy step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `core/run_listing.py` — `list_runs()` for recent runs

The TUI's Resume / Promote tasks (Slice 2) and the `dsim resume` / `dsim list-runs` CLIs (Task 15) need to enumerate recent runs under `runs/`, resolve the latest trial per run, and find the latest checkpoint inside each trial.

**Files:**
- Create: `core/run_listing.py`
- Create: `tests/core/test_run_listing.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Behavior contract for core.run_listing.

list_runs(runs_dir=Path("runs")) -> list[RunEntry]
  One entry per `runs/<run_name>/`, with latest_trial = the lex-max
  YYYYMMDD_HHMMSS subdir, plus latest_checkpoint = the highest-step .zip
  inside <latest_trial>/checkpoints/.

resolve_trial(run_name, *, trial=None, runs_dir=Path("runs")) -> Path
  Helper: returns runs/<run_name>/<trial>/ for a given trial id, or the
  latest trial if trial is None.

resolve_checkpoint(trial_dir, *, ckpt=None) -> Path
  Helper: returns the requested checkpoint zip, or the highest-step zip
  if ckpt is None.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _seed_run(runs_dir: Path, run: str, trials: list[str], *,
              checkpoints_per_trial: dict[str, list[int]] | None = None) -> None:
    checkpoints_per_trial = checkpoints_per_trial or {}
    for trial in trials:
        td = runs_dir / run / trial
        td.mkdir(parents=True)
        (td / ".hydra").mkdir()
        (td / ".hydra" / "config.yaml").write_text(f"run_name: {run}\n")
        for step in checkpoints_per_trial.get(trial, []):
            cks = td / "checkpoints"
            cks.mkdir(exist_ok=True)
            (cks / f"ppo_hoop_{step}_steps.zip").write_bytes(b"\x50\x4b\x03\x04stub")


def test_list_runs_one_entry_per_run_dir(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "ppo_hoop_blue_5", ["20260514_120000", "20260515_010000"])
    _seed_run(tmp_path, "ppo_hoop_red_2", ["20260516_020000"])
    rows = list_runs(runs_dir=tmp_path)
    names = sorted([r.run_name for r in rows])
    assert names == ["ppo_hoop_blue_5", "ppo_hoop_red_2"]


def test_list_runs_latest_trial_is_lex_max(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000",
                                   "20260514_120000"])
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_trial.name == "20260515_080000"


def test_list_runs_latest_checkpoint_is_highest_step(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"],
              checkpoints_per_trial={"20260514_120000": [50_000, 200_000, 100_000]})
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_checkpoint is not None
    assert "200000" in row.latest_checkpoint.name


def test_list_runs_no_checkpoints_returns_none_checkpoint(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"])
    [row] = list_runs(runs_dir=tmp_path)
    assert row.latest_checkpoint is None


def test_list_runs_filter_by_run_name(tmp_path: Path) -> None:
    from core.run_listing import list_runs
    _seed_run(tmp_path, "blue_5", ["20260514_120000"])
    _seed_run(tmp_path, "red_2", ["20260516_120000"])
    rows = list_runs(runs_dir=tmp_path, run_filter="blue_5")
    assert [r.run_name for r in rows] == ["blue_5"]


def test_resolve_trial_picks_latest_when_none(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000"])
    p = resolve_trial("blue_5", trial=None, runs_dir=tmp_path)
    assert p.name == "20260515_080000"


def test_resolve_trial_with_explicit_id(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    _seed_run(tmp_path, "blue_5", ["20260513_120000", "20260515_080000"])
    p = resolve_trial("blue_5", trial="20260513_120000", runs_dir=tmp_path)
    assert p.name == "20260513_120000"


def test_resolve_trial_unknown_raises(tmp_path: Path) -> None:
    from core.run_listing import resolve_trial
    with pytest.raises(FileNotFoundError):
        resolve_trial("nope", runs_dir=tmp_path)


def test_resolve_checkpoint_highest_step(tmp_path: Path) -> None:
    from core.run_listing import resolve_checkpoint
    trial = tmp_path / "trial"
    cks = trial / "checkpoints"
    cks.mkdir(parents=True)
    for s in (50_000, 200_000, 100_000):
        (cks / f"ppo_hoop_{s}_steps.zip").write_bytes(b"stub")
    p = resolve_checkpoint(trial)
    assert "200000" in p.name


def test_resolve_checkpoint_explicit(tmp_path: Path) -> None:
    from core.run_listing import resolve_checkpoint
    trial = tmp_path / "trial"
    cks = trial / "checkpoints"
    cks.mkdir(parents=True)
    (cks / "ppo_hoop_50000_steps.zip").write_bytes(b"stub")
    p = resolve_checkpoint(trial, ckpt="ppo_hoop_50000_steps")
    assert p.name == "ppo_hoop_50000_steps.zip"
```

- [ ] **Step 2: Implement `core/run_listing.py`**

```python
"""Enumerate `runs/` for resume / promote / list-runs workflows.

Each `runs/<run_name>/<YYYYMMDD_HHMMSS>/` is a trial; per the Hydra Part 1
convention.  `list_runs` returns one row per `<run_name>` with its latest
trial and that trial's latest checkpoint resolved.

Public API:
    list_runs(runs_dir=Path("runs"), run_filter=None) -> list[RunEntry]
    resolve_trial(run_name, trial=None, runs_dir=Path("runs")) -> Path
    resolve_checkpoint(trial_dir, ckpt=None) -> Path
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path("runs")
_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class RunEntry:
    run_name: str
    run_dir: Path
    latest_trial: Path
    latest_checkpoint: Path | None


def list_runs(runs_dir: Path = RUNS_DIR, run_filter: str | None = None) -> list[RunEntry]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    out: list[RunEntry] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if run_filter and run_filter not in d.name:
            continue
        trials = sorted([t for t in d.iterdir() if t.is_dir()])
        if not trials:
            continue
        latest = trials[-1]
        ckpt = _latest_checkpoint(latest)
        out.append(RunEntry(
            run_name=d.name, run_dir=d.resolve(),
            latest_trial=latest.resolve(),
            latest_checkpoint=ckpt.resolve() if ckpt else None,
        ))
    return out


def resolve_trial(
    run_name: str,
    *,
    trial: str | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    run_dir = Path(runs_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"no such run: {run_dir}")
    if trial is not None:
        td = run_dir / trial
        if not td.exists():
            raise FileNotFoundError(f"no such trial: {td}")
        return td.resolve()
    trials = sorted([t for t in run_dir.iterdir() if t.is_dir()])
    if not trials:
        raise FileNotFoundError(f"no trials under {run_dir}")
    return trials[-1].resolve()


def resolve_checkpoint(trial_dir: Path, *, ckpt: str | None = None) -> Path:
    cks = Path(trial_dir) / "checkpoints"
    if not cks.exists():
        raise FileNotFoundError(f"no checkpoints/ under {trial_dir}")
    if ckpt is not None:
        p = cks / (ckpt if ckpt.endswith(".zip") else ckpt + ".zip")
        if not p.exists():
            raise FileNotFoundError(f"no such checkpoint: {p}")
        return p.resolve()
    p = _latest_checkpoint(trial_dir)
    if p is None:
        raise FileNotFoundError(f"no .zip checkpoints under {cks}")
    return p


def _latest_checkpoint(trial_dir: Path) -> Path | None:
    cks = Path(trial_dir) / "checkpoints"
    if not cks.exists():
        return None
    best: tuple[int, Path] | None = None
    for f in cks.glob("*.zip"):
        m = _CKPT_STEPS_RE.search(f.name)
        if not m:
            continue
        steps = int(m.group(1))
        if best is None or steps > best[0]:
            best = (steps, f)
    return best[1] if best else None
```

- [ ] **Step 3: Run, fix until green**

```bash
conda run -n uav python -m pytest tests/core/test_run_listing.py -v
```

Expected: PASS.

- [ ] **Step 4: Smoke**

```bash
conda run -n uav python -c "
from core.run_listing import list_runs
for r in list_runs():
    ck = r.latest_checkpoint.name if r.latest_checkpoint else '(none)'
    print(f'{r.run_name:40s} latest_trial={r.latest_trial.name}  ckpt={ck}')
"
```

Expected: one row per `runs/<name>/`; latest_trial is the most recent timestamped subdir; latest_checkpoint resolves to the highest-step zip.

- [ ] **Step 5: Commit**

```bash
git add core/run_listing.py tests/core/test_run_listing.py
git commit -m "feat(run-listing): core.run_listing — list/resume/checkpoint helpers

list_runs() enumerates runs/<name>/, picking the latest trial per run
and that trial's highest-step checkpoint.  resolve_trial() and
resolve_checkpoint() handle the 'latest by default, explicit by flag'
ergonomics used by dsim resume / list-runs (Task 15) and the TUI
Resume task (Slice 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Extract `core/eval_core.py` from `scripts/eval_team.py`

The per-episode loop, terminal-cause bucket counters, take-down counter, and OCE plumbing in `scripts/eval_team.py` need to become an importable function so the Hydra-migrated `eval_team.py` (Task 10) and the new `eval_battery.py` (Task 13) can share one engine.

**Files:**
- Create: `core/eval_core.py`
- Create: `tests/core/test_eval_core.py`

- [ ] **Step 1: Read the current `scripts/eval_team.py`** (it's ~256 lines). Identify:
  - The per-episode loop and what info it tracks.
  - The terminal-cause buckets (`drone_drone_crash`, `red_floor`, `blue_floor`, `red_wall`, `blue_wall`, `red_oob`, `blue_oob`, `timeout`, `score`).
  - The OCE / FrameStackWrapper plumbing in learner mode.
  - The `from_spec()` opponent factory call.
  - The crash-aftermath handling.

- [ ] **Step 2: Write the failing test**

`tests/core/test_eval_core.py`:

```python
"""Behavior contract for core.eval_core.run_scenario.

A small integration test: run 1 episode against zero_red (immediate
crash), assert ScenarioResult has the expected shape and bucket counts.
Slow tests run a real loop; quick ones use a stub policy.

Marked @pytest.mark.slow because it touches the MuJoCo env.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_run_scenario_produces_episode_results() -> None:
    from core.eval_core import ScenarioSpec, run_scenario

    spec = ScenarioSpec(
        opponent="zero_red",
        opponent_model_path=None,
        randomise_start=False,
        n_episodes=1,
        crash_aftermath_seconds=0.0,
        deterministic=True,
        learner_id="blue_0",
        seed=42,
    )

    # Use beeline_blue as the "learner" (it's a scripted policy, so we can run
    # without loading weights — pass a sentinel that eval_core knows to use
    # for scripted "learners" in tests).  Real callers always pass a model URI.
    result = run_scenario(
        learner_uri="scripted:beeline_blue",
        scenario=spec,
        render=False,
    )

    assert result.scenario == spec
    assert len(result.episodes) == 1
    ep = result.episodes[0]
    assert ep.length > 0
    assert ep.terminal_cause in {
        "drone_drone_crash", "red_floor", "blue_floor",
        "red_wall", "blue_wall", "red_oob", "blue_oob",
        "timeout", "score",
    }
    # Aggregates
    assert 0.0 <= result.win_rate <= 1.0
    assert isinstance(result.terminal_cause_counts, dict)
```

- [ ] **Step 3: Implement `core/eval_core.py`**

```python
"""Eval engine: run a learner through N episodes of one scenario.

The interactive eval_team.py CLI (Task 10) and the eval_battery.py engine
(Task 13) both go through run_scenario().  Pure function (no print, no
sys.exit); returns a ScenarioResult with per-episode detail + aggregates.

Implementation notes:
  - Uses envs.quidditch.team_env.QuidditchTeamEnv directly.
  - For non-scripted learners, loads the PPO via PPO.load(learner_uri); the
    obs spec + n_stack are read from the learner's .hydra/config.yaml via
    core.run_context.load_run_context.
  - learner_id + learner_spec are first-class config fields (per the
    feature/blue-v7-body-ego refactor at de34230); team_env builds the
    right obs per-agent.  OCE is a pure pass-through.
  - Crash-aftermath handling matches the existing _step_aftermath path in
    scripts/eval_team.py (info dicts carry drone_drone_crash=True so the
    bucket classifier doesn't mis-route to timeout).
  - For `opponent='scripted:<name>'` learners (test only), uses the
    same scripted policies from envs.quidditch.opponents.from_spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

OpponentKind = Literal["scripted", "frozen"]
TERMINAL_BUCKETS = (
    "drone_drone_crash",
    "red_floor", "blue_floor",
    "red_wall",  "blue_wall",
    "red_oob",   "blue_oob",
    "timeout",
    "score",
)


@dataclass(frozen=True)
class ScenarioSpec:
    opponent: str                  # "beeline_red" / "intercepter_red:lookahead=0.5" / "frozen" / ...
    opponent_model_path: str | None  # required when opponent == "frozen"
    randomise_start: bool
    n_episodes: int
    crash_aftermath_seconds: float = 0.0
    deterministic: bool = True
    learner_id: str = "blue_0"
    seed: int = 0


@dataclass(frozen=True)
class EpisodeResult:
    length: int
    reward_learner: float
    reward_opponent: float
    terminal_cause: str            # one of TERMINAL_BUCKETS
    take_down_fired: bool
    score_at_episode_end: int | None  # learner side scored (1), opponent side (-1), neither (None)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioSpec
    episodes: list[EpisodeResult]
    win_rate: float
    mean_reward_learner: float
    mean_reward_opponent: float
    take_down_rate: float
    terminal_cause_counts: dict[str, int]
    mean_episode_length: float


def run_scenario(
    learner_uri: str,
    scenario: ScenarioSpec,
    *,
    render: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> ScenarioResult:
    """Run `scenario.n_episodes` episodes; aggregate into a ScenarioResult.

    `learner_uri` may be:
      - filesystem path to a best_model.zip
      - wandb:// URI (resolves via scripts._artifact_io.resolve_parent)
      - "scripted:<spec>" for tests (e.g. "scripted:beeline_blue")
    """
    # Implementation — port from scripts/eval_team.py:
    #   1. Resolve learner: PPO.load(weights) + read its .hydra/config.yaml
    #      to get learner_spec + n_stack.  Or, for "scripted:<spec>", use
    #      envs.quidditch.opponents.from_spec(<spec>) as the learner policy.
    #   2. Build QuidditchTeamEnv with:
    #        cfg.learner_id = scenario.learner_id
    #        cfg.learner_spec = <learner's obs spec name>
    #        cfg.randomise_start = scenario.randomise_start
    #        cfg.crash_aftermath_seconds = scenario.crash_aftermath_seconds
    #   3. Resolve opponent: if scenario.opponent == "frozen", build a
    #      FrozenPolicyOpponent(scenario.opponent_model_path); else
    #      from_spec(scenario.opponent).
    #   4. Wrap learner side through OCE if learner_id != "red_0"; team_env
    #      now builds per-agent obs natively (post feature/blue-v7-body-ego).
    #      Frame-stack via the learner's n_stack.
    #   5. For each episode:
    #        - episode counters start at 0
    #        - while not done: step the env, accumulate rewards, count
    #          take-down trigger (info["take_down_fired"])
    #        - on done, classify terminal_cause from the merged info dict
    #          (matches scripts/eval_team.py:_classify_terminal exactly)
    #        - append EpisodeResult
    #   6. Aggregate into ScenarioResult.
    raise NotImplementedError("port from scripts/eval_team.py main()")
```

The port is mechanical but careful — the existing `scripts/eval_team.py:main()` has the full loop. Copy its body, parameterizing on `ScenarioSpec` instead of argparse args. Replace the print-based summary at the end with the return of a `ScenarioResult`.

The `_classify_terminal` helper (mapping the merged info dict → bucket string) MUST be shared between the old and new code paths; lift it into `core/eval_core.py` and have it return one of the `TERMINAL_BUCKETS` strings.

- [ ] **Step 4: Run, fix until green**

```bash
conda run -n uav python -m pytest tests/core/test_eval_core.py -v -m slow
```

Expected: PASS. The test runs one episode against `zero_red` with `beeline_blue` as the scripted-learner — should terminate quickly (probably as `blue_oob` or `timeout`).

- [ ] **Step 5: Verify the existing eval_team.py canary still passes**

We haven't migrated `scripts/eval_team.py` yet (Task 10); it should still work unchanged. Run the existing smoke:

```bash
conda run -n uav python -m scripts.eval_team --red beeline_red --blue beeline_blue --episodes 2 --no-gui
```

Expected: prints two-episode summary; same as before this task.

- [ ] **Step 6: Commit**

```bash
git add core/eval_core.py tests/core/test_eval_core.py
git commit -m "feat(eval-core): core.eval_core.run_scenario — extracted from eval_team.py

Pure-function eval engine: takes (learner_uri, ScenarioSpec) and returns
ScenarioResult with per-episode EpisodeResults + aggregates.  Aligned
with feature/blue-v7-body-ego's first-class learner_id/learner_spec
env_factory plumbing.

Used by:
  - scripts/eval_team.py (Hydra migration, Task 10)
  - scripts/eval_battery.py (new, Task 13)

Old scripts/eval_team.py path still works unchanged for this commit; it
gets rewritten in Task 10.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Hydra config groups for eval

**Files (all created):**
- `conf/eval/default.yaml`
- `conf/learner/blue.yaml`
- `conf/learner/red.yaml`
- `conf/eval_team/default.yaml`
- `conf/eval_ppo/default.yaml`

- [ ] **Step 1: Create `conf/eval/default.yaml`** — shared eval params

```yaml
# Shared eval params across eval_team, eval_ppo, and eval_battery.
n_episodes: 5
deterministic: true
crash_aftermath_seconds: 0.0
gui: false
randomise_start: false
seed: 0
```

- [ ] **Step 2: Create `conf/learner/blue.yaml` and `conf/learner/red.yaml`**

`conf/learner/blue.yaml`:

```yaml
# Identify the learner side.  uri is required at CLI time; obs spec + n_stack
# are resolved from the URI's .hydra/config.yaml at run time (via
# core.run_context.load_run_context).
id: blue_0
uri: ???
```

`conf/learner/red.yaml`:

```yaml
id: red_0
uri: ???
```

- [ ] **Step 3: Create `conf/eval_team/default.yaml`**

```yaml
# Hydra entrypoint config for scripts/eval_team.py.  Composes:
#   - eval/default        — shared params
#   - learner/blue|red    — picked at CLI: `learner=blue learner.uri=…`
#   - opponent/<choice>   — picked at CLI: `opponent=beeline_red`
defaults:
  - /eval: default
  - /learner: ???           # required: pick `learner=blue` or `learner=red`
  - /opponent: ???          # required: pick `opponent=beeline_red` etc.
  - _self_

# Resolved logging — eval_team writes a one-line summary to stdout and
# (when wandb is enabled) logs aggregate metrics under `eval/*`.
log_per_episode: false
```

- [ ] **Step 4: Create `conf/eval_ppo/default.yaml`**

```yaml
# Hydra entrypoint config for scripts/eval_ppo.py (single-agent).
defaults:
  - /eval: default
  - _self_

# Single-agent eval doesn't take a learner group or an opponent — just the
# model URI as `model_uri`.
model_uri: ???
n_episodes: 10              # default: more than the team-eval default (5),
                            # matches the historical eval_ppo default.
```

- [ ] **Step 5: Verify Hydra can compose these**

```bash
conda run -n uav python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
cfg_dir = str(Path('conf').resolve())
with initialize_config_dir(version_base=None, config_dir=cfg_dir):
    cfg = compose(
        config_name='config',
        overrides=['+eval_team=default', 'learner=blue',
                   '+learner.uri=models/ppo_hoop_blue_4_x/best_model',
                   'opponent=beeline_red'],
    )
    print(cfg.eval_team)
"
```

Expected: prints a resolved config object with eval/learner/opponent fields populated. (Will error on missing `+learner.uri` if you forget the `+` prefix — that's expected.)

- [ ] **Step 6: Commit**

```bash
git add conf/eval/ conf/learner/ conf/eval_team/ conf/eval_ppo/
git commit -m "feat(conf): Hydra config groups for eval (eval/, learner/, eval_team/, eval_ppo/)

Shared eval params in /eval, side selection in /learner, top-level
composition for the two entrypoints (eval_team, eval_ppo) in their
own groups.  Opponent group is reused from the existing /opponent
tree (added in Hydra Part 1).

These YAMLs are consumed in Tasks 10 + 11 (the script migrations).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Hydra migration — `scripts/eval_team.py`

**Files:**
- Modify: `scripts/eval_team.py` (full rewrite — argparse → @hydra.main)
- Modify: `tests/scripts/test_eval_team.py` (if exists; else create)
- Modify: `Makefile` (REMOVED in Task 17 — keep the existing target for now to preserve compatibility through Tasks 10-16)

- [ ] **Step 1: Replace `scripts/eval_team.py` with a Hydra entrypoint**

```python
"""Head-to-head eval — Hydra entrypoint.

Usage (Hydra):
    python -m scripts.eval_team +eval_team=default \\
        learner=blue learner.uri=models/ppo_hoop_blue_4_*/best_model \\
        opponent=beeline_red \\
        eval.gui=true eval.crash_aftermath_seconds=3.0 eval.n_episodes=5

The argparse surface (--learner/--learner-frame-stack/--blue/--red/--gui/
--crash-aftermath-seconds/--episodes/--deterministic/--randomise-start)
is gone; all those flags are now Hydra overrides on the config groups.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
from omegaconf import DictConfig

from core.eval_core import ScenarioSpec, run_scenario


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    et = cfg.eval_team
    learner = et.learner
    opp = et.opponent

    # Translate the Hydra cfg into a ScenarioSpec for core.eval_core.
    scenario = ScenarioSpec(
        opponent=_opponent_spec_from_cfg(opp),
        opponent_model_path=getattr(opp, "model_path", None),
        randomise_start=bool(et.eval.randomise_start),
        n_episodes=int(et.eval.n_episodes),
        crash_aftermath_seconds=float(et.eval.crash_aftermath_seconds),
        deterministic=bool(et.eval.deterministic),
        learner_id=str(learner.id),
        seed=int(et.eval.seed),
    )

    result = run_scenario(
        learner_uri=str(learner.uri),
        scenario=scenario,
        render=bool(et.eval.gui),
    )

    _print_summary(result)


def _opponent_spec_from_cfg(opp_cfg: DictConfig) -> str:
    """Map the /opponent config-group choice back to a from_spec()-style string.

    The /opponent group ships with one YAML per scripted opponent (e.g.
    `conf/opponent/beeline_red.yaml`) plus `frozen.yaml`.  Each YAML
    declares a `spec` field that's the canonical from_spec() string;
    we just forward it.
    """
    return str(opp_cfg.spec)


def _print_summary(result) -> None:
    print(f"\n=== {result.scenario.opponent}  (n={len(result.episodes)}) ===")
    print(f"  win_rate:               {result.win_rate:.2%}")
    print(f"  mean reward (learner):  {result.mean_reward_learner:+.3f}")
    print(f"  mean reward (opponent): {result.mean_reward_opponent:+.3f}")
    print(f"  take-down rate:         {result.take_down_rate:.2%}")
    print(f"  mean episode length:    {result.mean_episode_length:.1f}")
    print(f"  terminal buckets:")
    for cause, n in sorted(result.terminal_cause_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {cause:24s} {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify each `conf/opponent/*.yaml` has a `spec:` field**

```bash
cd "$REPO_ROOT" && for f in conf/opponent/*.yaml; do echo "--- $f ---"; cat "$f"; done
```

If any opponent YAML doesn't carry a `spec` field, add one (e.g. `spec: beeline_red`). If the existing /opponent group uses `_target_` instantiation instead of a `spec` string, adapt `_opponent_spec_from_cfg` to construct the string from the cfg fields (e.g. `f"{cfg.kind}_{cfg.side}"`). The point is: `core.eval_core.run_scenario` takes a string opponent spec; the cfg layer maps to it.

- [ ] **Step 3: Write a smoke test**

`tests/scripts/test_eval_team_hydra.py`:

```python
"""Smoke: eval_team Hydra entrypoint composes + runs 1 episode."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_team_hydra_runs_one_episode(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_team",
         "+eval_team=default",
         "learner=blue", "learner.uri=scripted:beeline_blue",
         "opponent=beeline_red",
         "eval.n_episodes=1",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "win_rate:" in result.stdout
```

(The test uses `scripted:beeline_blue` as the learner — a sentinel only `core.eval_core.run_scenario` accepts in tests so we don't need to load weights. The implementation in Task 8 added this path.)

- [ ] **Step 4: Run the test**

```bash
conda run -n uav python -m pytest tests/scripts/test_eval_team_hydra.py -v -m slow
```

Expected: PASS.

- [ ] **Step 5: Run the existing canary suite as a regression gate**

```bash
make test
```

Expected: same green as before this task (the single-agent + team canaries should be unchanged — we only touched `scripts/eval_team.py`, which isn't in any canary path).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_team.py tests/scripts/test_eval_team_hydra.py conf/opponent/*.yaml
git commit -m "feat(eval-team): Hydra migration — scripts/eval_team.py via @hydra.main

The argparse surface is gone.  Compose via /eval_team=default + learner
+ opponent + /eval overrides.  Body is a thin wrapper around
core.eval_core.run_scenario; results print as before but the engine is
shared with the new scripts/eval_battery.py (Task 13).

Existing /opponent config-group YAMLs gain a `spec:` field if they
didn't carry one; that's the string fed to from_spec inside eval_core.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Hydra migration — `scripts/eval_ppo.py`

Same shape as Task 10 but for single-agent.

**Files:**
- Modify: `scripts/eval_ppo.py` (full rewrite — argparse → @hydra.main)
- Create: `tests/scripts/test_eval_ppo_hydra.py`

- [ ] **Step 1: Replace `scripts/eval_ppo.py` with a Hydra entrypoint**

```python
"""Single-agent eval — Hydra entrypoint.

Usage:
    python -m scripts.eval_ppo +eval_ppo=default \\
        model_uri=models/ppo_hoop_rand_start_20260505_174509/best_model \\
        eval.gui=true eval.n_episodes=10
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO

from envs.quidditch.simple_env import QuidditchSimpleEnv
from scripts._artifact_io import resolve_parent
from scripts._train_common import _maybe_apply_kmp_dupes  # type: ignore[attr-defined]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ep = cfg.eval_ppo

    model_path = resolve_parent(ep.model_uri)
    model = PPO.load(model_path, device="cpu")

    env = QuidditchSimpleEnv(
        # … existing env construction from the old eval_ppo.py; the env factory
        # call in develop should already align with the Hydra env config under
        # cfg.env, which scripts/eval_ppo.py can compose into via +env=default
        # if needed.
    )

    rewards = []
    for _ in range(int(ep.eval.n_episodes)):
        obs, _ = env.reset(seed=int(ep.eval.seed))
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=bool(ep.eval.deterministic))
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += float(r)
            done = bool(terminated or truncated)
        rewards.append(ep_reward)

    print(f"n={len(rewards)}  mean={np.mean(rewards):+.4f}  std={np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
```

The exact env construction must mirror what `scripts/eval_ppo.py` does on develop today — copy that logic verbatim. The point of this task is to flip the entrypoint from argparse to Hydra and route the model URI through `resolve_parent` so wandb:// URIs work.

- [ ] **Step 2: Write a smoke test**

`tests/scripts/test_eval_ppo_hydra.py`:

```python
"""Smoke: eval_ppo Hydra entrypoint composes + runs 1 episode."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_ppo_hydra_runs_one_episode(tmp_path) -> None:
    # rand_start is the cheapest single-agent promoted model.
    model = "models/ppo_hoop_rand_start_20260505_174509/best_model"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_ppo",
         "+eval_ppo=default",
         f"eval_ppo.model_uri={model}",
         "eval_ppo.eval.n_episodes=1",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "mean=" in result.stdout
```

- [ ] **Step 3: Run + canary**

```bash
conda run -n uav python -m pytest tests/scripts/test_eval_ppo_hydra.py -v -m slow
make test
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_ppo.py tests/scripts/test_eval_ppo_hydra.py
git commit -m "feat(eval-ppo): Hydra migration — scripts/eval_ppo.py via @hydra.main

Single-agent counterpart of eval_team's migration in Task 10.  Same
shape: argparse out, @hydra.main in, model_uri routed through
resolve_parent so wandb:// URIs work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `core/eval_report.py` — MD + CSV + JSONL writer

**Files:**
- Create: `core/eval_report.py`
- Create: `tests/core/test_eval_report.py`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_eval_report.py`:

```python
"""Behavior contract for core.eval_report.write_report.

Given a list[ScenarioResult] + a candidate metadata dict, writes:
  - summary.md       Markdown with candidate header + one row per scenario
  - results.csv      Flat CSV; one row per scenario; all aggregates as columns
  - per_episode.jsonl  One JSON object per episode

All three files live under <output_dir>/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from core.eval_core import EpisodeResult, ScenarioResult, ScenarioSpec


def _make_scenario_result(opp: str, n: int = 3, win_rate: float = 0.5) -> ScenarioResult:
    spec = ScenarioSpec(
        opponent=opp, opponent_model_path=None, randomise_start=True,
        n_episodes=n, deterministic=True, learner_id="blue_0", seed=0,
    )
    episodes = [
        EpisodeResult(length=500 + i, reward_learner=1.0 + i,
                      reward_opponent=-0.5,
                      terminal_cause="drone_drone_crash" if i == 0 else "timeout",
                      take_down_fired=(i == 0),
                      score_at_episode_end=None)
        for i in range(n)
    ]
    return ScenarioResult(
        scenario=spec, episodes=episodes,
        win_rate=win_rate, mean_reward_learner=2.0, mean_reward_opponent=-0.5,
        take_down_rate=1.0 / n,
        terminal_cause_counts={"drone_drone_crash": 1, "timeout": n - 1},
        mean_episode_length=500 + (n - 1) / 2,
    )


def test_write_report_creates_all_three_files(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {
        "name": "ppo_hoop_blue_4_20260511_202612",
        "short_name": "blue_4",
        "obs_spec": "DUEL_V2_WORLD",
        "n_stack": 3,
        "parent_chain_total": 30_007_296,
        "source": "vendored",
        "wandb_alias": "prod",
    }
    results = [
        _make_scenario_result("beeline_red"),
        _make_scenario_result("intercepter_red:lookahead=0.5"),
    ]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "per_episode.jsonl").exists()


def test_summary_md_includes_candidate_header_and_one_row_per_scenario(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("beeline_red"),
               _make_scenario_result("zero_red")]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    md = (tmp_path / "summary.md").read_text()
    assert "DUEL_V2_WORLD" in md
    assert "beeline_red" in md
    assert "zero_red" in md
    # Header table row count: each scenario row in the summary table.
    # Check by counting "| beeline_red" + "| zero_red".
    assert md.count("| beeline_red") == 1
    assert md.count("| zero_red") == 1


def test_results_csv_one_row_per_scenario(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("a"), _make_scenario_result("b"),
               _make_scenario_result("c")]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["opponent"] for r in rows} == {"a", "b", "c"}


def test_per_episode_jsonl_one_object_per_episode(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("beeline_red", n=3),
               _make_scenario_result("zero_red", n=2)]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    lines = (tmp_path / "per_episode.jsonl").read_text().splitlines()
    assert len(lines) == 5
    rows = [json.loads(line) for line in lines]
    assert {r["scenario"] for r in rows} == {"beeline_red", "zero_red"}
    assert all("terminal_cause" in r for r in rows)
```

- [ ] **Step 2: Implement `core/eval_report.py`**

```python
"""Battery report writer: Markdown + CSV + JSONL.

Public API:
    write_report(scenario_results, output_dir, candidate)
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.eval_core import ScenarioResult


def write_report(
    scenario_results: list[ScenarioResult],
    output_dir: Path,
    candidate: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_md(scenario_results, output_dir / "summary.md", candidate)
    _write_results_csv(scenario_results, output_dir / "results.csv")
    _write_per_episode_jsonl(scenario_results, output_dir / "per_episode.jsonl")


def _write_summary_md(results: list[ScenarioResult], path: Path, candidate: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Eval battery — {candidate['short_name']}")
    lines.append("")
    lines.append(f"**Run:** `{candidate['name']}`  ·  "
                 f"**Obs:** `{candidate['obs_spec']}` × n_stack={candidate['n_stack']}")
    lines.append(f"**Chain total:** {candidate['parent_chain_total']:,} steps  ·  "
                 f"**Source:** {candidate['source']}  ·  "
                 f"**Alias:** `{candidate.get('wandb_alias') or '—'}`")
    lines.append("")
    lines.append("## Per-scenario summary")
    lines.append("")
    lines.append("| Opponent | Start | n_eps | win% | mean R (lrn) | mean R (opp) | take-down% | mean ep len |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        start = "random" if r.scenario.randomise_start else "fixed"
        lines.append(
            f"| {r.scenario.opponent} | {start} | {len(r.episodes)} | "
            f"{r.win_rate * 100:.1f}% | {r.mean_reward_learner:+.3f} | "
            f"{r.mean_reward_opponent:+.3f} | {r.take_down_rate * 100:.1f}% | "
            f"{r.mean_episode_length:.1f} |"
        )
    lines.append("")
    lines.append("## Terminal-cause breakdown")
    lines.append("")
    for r in results:
        start = "random" if r.scenario.randomise_start else "fixed"
        lines.append(f"### {r.scenario.opponent} ({start} start)")
        lines.append("")
        for cause, n in sorted(r.terminal_cause_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{cause}**: {n}")
        lines.append("")
    path.write_text("\n".join(lines))


def _write_results_csv(results: list[ScenarioResult], path: Path) -> None:
    fields = [
        "opponent", "opponent_model_path", "randomise_start", "n_episodes",
        "deterministic", "crash_aftermath_seconds", "seed",
        "win_rate", "mean_reward_learner", "mean_reward_opponent",
        "take_down_rate", "mean_episode_length",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {**asdict(r.scenario),
                   "win_rate": r.win_rate,
                   "mean_reward_learner": r.mean_reward_learner,
                   "mean_reward_opponent": r.mean_reward_opponent,
                   "take_down_rate": r.take_down_rate,
                   "mean_episode_length": r.mean_episode_length}
            w.writerow({k: row.get(k) for k in fields})


def _write_per_episode_jsonl(results: list[ScenarioResult], path: Path) -> None:
    with path.open("w") as f:
        for r in results:
            for i, ep in enumerate(r.episodes):
                obj = {
                    "scenario": r.scenario.opponent,
                    "randomise_start": r.scenario.randomise_start,
                    "ep_idx": i,
                    "length": ep.length,
                    "reward_learner": ep.reward_learner,
                    "reward_opponent": ep.reward_opponent,
                    "terminal_cause": ep.terminal_cause,
                    "take_down_fired": ep.take_down_fired,
                    "score_at_episode_end": ep.score_at_episode_end,
                }
                f.write(json.dumps(obj) + "\n")
```

- [ ] **Step 3: Run + commit**

```bash
conda run -n uav python -m pytest tests/core/test_eval_report.py -v
```

Expected: PASS.

```bash
git add core/eval_report.py tests/core/test_eval_report.py
git commit -m "feat(eval-report): core.eval_report.write_report — MD + CSV + JSONL

Local-only battery report writer (per spec section 1.7).  Three files:
summary.md (Markdown with candidate header + per-scenario table +
terminal-cause breakdown), results.csv (one row per scenario, all
aggregates as columns; grep-friendly), per_episode.jsonl (one JSON
object per episode; for ad-hoc re-aggregation).

Used by scripts/eval_battery.py (Task 13).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: `scripts/eval_battery.py` + `conf/eval_battery/*.yaml`

**Files:**
- Create: `conf/eval_battery/default.yaml`
- Create: `conf/eval_battery/quick.yaml`
- Create: `conf/eval_battery/ladder.yaml`
- Create: `conf/eval_battery/scripted_only.yaml`
- Create: `scripts/eval_battery.py`
- Create: `tests/scripts/test_eval_battery.py`

- [ ] **Step 1: Create battery YAMLs**

`conf/eval_battery/default.yaml`:

```yaml
defaults:
  - _self_
candidate: ???              # required: wandb://… or models/… or runs/…
output_dir: null            # default = ${hydra.run.dir}/eval_report
seed: 0
scenarios:
  - opponent: beeline_red
    opponent_model_path: null
    randomise_start: false
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: beeline_red
    opponent_model_path: null
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: intercepter_red:lookahead=0.5
    opponent_model_path: null
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: zero_red
    opponent_model_path: null
    randomise_start: false
    n_episodes: 5
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: frozen
    opponent_model_path: models/ppo_hoop_red_1_20260506_103058/best_model
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
```

`conf/eval_battery/quick.yaml`:

```yaml
defaults:
  - _self_
candidate: ???
output_dir: null
seed: 0
scenarios:
  - opponent: beeline_red
    opponent_model_path: null
    randomise_start: true
    n_episodes: 3
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: zero_red
    opponent_model_path: null
    randomise_start: false
    n_episodes: 3
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: frozen
    opponent_model_path: models/ppo_hoop_red_1_20260506_103058/best_model
    randomise_start: true
    n_episodes: 3
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
```

`conf/eval_battery/ladder.yaml`:

```yaml
defaults:
  - _self_
candidate: ???
output_dir: null
seed: 0
scenarios:
  - opponent: frozen
    opponent_model_path: models/ppo_hoop_red_1_20260506_103058/best_model
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  # Add more frozen rungs as the ladder grows; this list is intentionally
  # short for now — extend when red_2 / blue_5 / blue_6 / blue_7 promote.
```

`conf/eval_battery/scripted_only.yaml`:

```yaml
defaults:
  - _self_
candidate: ???
output_dir: null
seed: 0
scenarios:
  - opponent: zero_red
    opponent_model_path: null
    randomise_start: false
    n_episodes: 5
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: beeline_red
    opponent_model_path: null
    randomise_start: false
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: beeline_red
    opponent_model_path: null
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
  - opponent: intercepter_red:lookahead=0.5
    opponent_model_path: null
    randomise_start: true
    n_episodes: 10
    crash_aftermath_seconds: 0.0
    deterministic: true
    learner_id: blue_0
```

- [ ] **Step 2: Implement `scripts/eval_battery.py`**

```python
"""Run a candidate through a battery of scenarios; emit local Markdown/CSV/JSONL report.

Usage:
    python -m scripts.eval_battery +eval_battery=default \\
        candidate=models/ppo_hoop_blue_4_20260511_202612/best_model

    python -m scripts.eval_battery +eval_battery=quick \\
        candidate=wandb://ppo_hoop_blue_4:prod

Writes:
    <hydra.run.dir>/eval_report/{summary.md,results.csv,per_episode.jsonl}
unless `output_dir` is set explicitly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from core.eval_core import ScenarioSpec, run_scenario
from core.eval_report import write_report
from core.inventory import inventory
from core.run_context import load_run_context
from scripts._artifact_io import resolve_parent
from scripts._wandb_init import init_wandb  # type: ignore


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    eb = cfg.eval_battery
    candidate_uri = str(eb.candidate)

    # Load candidate metadata for the report header.  Uses the same
    # load_run_context loader as the inventory, so the field names match
    # MODEL.md exactly.
    candidate_dir = resolve_parent(candidate_uri, metadata_only=True)
    ctx = load_run_context(candidate_dir)
    candidate_meta = _candidate_summary_from_ctx(candidate_dir, ctx)

    # Single wandb run for the whole battery.
    init_wandb(cfg, mode_override="eval_battery") if False else _noop_wandb()
    # See note below about init_wandb signature compatibility.

    scenario_results = []
    for s in eb.scenarios:
        spec = ScenarioSpec(
            opponent=str(s.opponent),
            opponent_model_path=(str(s.opponent_model_path) if s.opponent_model_path else None),
            randomise_start=bool(s.randomise_start),
            n_episodes=int(s.n_episodes),
            crash_aftermath_seconds=float(s.crash_aftermath_seconds),
            deterministic=bool(s.deterministic),
            learner_id=str(s.learner_id),
            seed=int(eb.seed),
        )
        result = run_scenario(learner_uri=candidate_uri, scenario=spec, render=False)
        scenario_results.append(result)
        _log_scenario_to_wandb(result)

    out_dir = Path(eb.output_dir) if eb.output_dir else Path(hydra_run_dir()) / "eval_report"
    write_report(scenario_results, output_dir=out_dir, candidate=candidate_meta)
    print(f"\nReport written to {out_dir}/")
    print(f"  summary.md       — Markdown summary")
    print(f"  results.csv      — one row per scenario")
    print(f"  per_episode.jsonl — per-episode detail")


def _candidate_summary_from_ctx(d: Path, ctx) -> dict:
    cfg = ctx["cfg"]
    meta = ctx.get("meta") or {}
    wandb_meta = ctx.get("wandb_meta") or {}
    return {
        "name": d.name,
        "short_name": _short_name(d.name),
        "obs_spec": str((cfg.get("obs") or {}).get("name", "?")),
        "n_stack": int((cfg.get("obs") or {}).get("n_stack", 1)),
        "parent_chain_total": int(meta.get("parent_chain_total", 0) or 0),
        "source": "vendored" if "/.cache/" not in str(d) else "cache",
        "wandb_alias": wandb_meta.get("alias"),
    }


def _short_name(full: str) -> str:
    if full.startswith("ppo_hoop_"):
        full = full[len("ppo_hoop_"):]
    parts = full.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        full = parts[0]
    return full


def _log_scenario_to_wandb(result) -> None:
    if wandb.run is None:
        return
    wandb.log({
        f"battery/{result.scenario.opponent}/win_rate": result.win_rate,
        f"battery/{result.scenario.opponent}/mean_R_learner": result.mean_reward_learner,
        f"battery/{result.scenario.opponent}/take_down_rate": result.take_down_rate,
    })


def _noop_wandb() -> None:
    """Placeholder; replace with the real init_wandb call once you confirm
    it accepts a `mode_override` kwarg.  If not, extend init_wandb to
    accept a tag-set parameter so 'eval_battery' propagates as a tag."""
    pass


def hydra_run_dir() -> str:
    """Return Hydra's `runtime.output_dir` for the active run."""
    from hydra.core.hydra_config import HydraConfig
    return HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
```

The `init_wandb` integration needs a brief audit: read `scripts/_wandb_init.py` to see what its signature is (the brain says it's the single call site for `wandb.init` and that it derives tags from cfg). Extend it (if needed) to accept a `mode_tag: str | None = None` kwarg so the battery can tag its run as `mode=eval_battery`. If `init_wandb` already does the right thing via the cfg, just compose `cfg.wandb.tags = [..., "eval_battery"]` in the battery YAML and the call site stays clean.

- [ ] **Step 3: Write the test**

```python
"""Smoke: eval_battery composes quick.yaml + runs 1-episode scenarios."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_eval_battery_quick_produces_report(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_battery",
         "+eval_battery=quick",
         "candidate=models/ppo_hoop_blue_4_20260511_202612/best_model",
         f"hydra.run.dir={tmp_path}/run",
         "eval_battery.scenarios.0.n_episodes=1",
         "eval_battery.scenarios.1.n_episodes=1",
         "eval_battery.scenarios.2.n_episodes=1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "eval_report" / "summary.md").exists()
    assert (tmp_path / "run" / "eval_report" / "results.csv").exists()
    assert (tmp_path / "run" / "eval_report" / "per_episode.jsonl").exists()
```

- [ ] **Step 4: Run + commit**

```bash
conda run -n uav python -m pytest tests/scripts/test_eval_battery.py -v -m slow
make test  # canary regression gate
```

```bash
git add scripts/eval_battery.py conf/eval_battery/ tests/scripts/test_eval_battery.py scripts/_wandb_init.py
git commit -m "feat(eval-battery): scripts/eval_battery.py + conf/eval_battery/*.yaml

New Hydra entrypoint: runs a candidate through N scenarios (one process,
one wandb.init), writes a local Markdown/CSV/JSONL report under
<hydra.run.dir>/eval_report/.

Four presets: default (5 scenarios × 10 eps), quick (3 × 3), ladder
(vs each frozen rung), scripted_only.  Battery doesn't log a model
artifact — it's tagged mode=eval_battery so it stays searchable in
W&B without polluting the artifact registry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `dsim` subcommands — read-only (inventory, obs-preflight, obs-specs, describe-run)

**Files:**
- Create: `dsim/commands/inventory.py`
- Create: `dsim/commands/obs_preflight.py`
- Create: `dsim/commands/obs_specs.py`
- Create: `dsim/commands/describe_run.py`
- Modify: `dsim/cli.py` (register subcommands)
- Create: `tests/dsim/test_inventory_cli.py`
- Create: `tests/dsim/test_obs_preflight_cli.py`
- Create: `tests/dsim/test_obs_specs_cli.py`
- Create: `tests/dsim/test_describe_run_cli.py`

- [ ] **Step 1: Implement `dsim/commands/inventory.py`**

```python
"""dsim inventory — list promoted models with obs spec, parent, chain total."""
from __future__ import annotations

import json as json_module
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.inventory import inventory

app = typer.Typer(help="List promoted models.")


@app.callback(invoke_without_command=True)
def main(
    json_out: bool = typer.Option(False, "--json", help="Output as JSON instead of a table"),
    include_cache: bool = typer.Option(False, "--include-cache",
                                       help="Include models/.cache/ entries"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir",
                                    help="Override the models/ directory"),
) -> None:
    rows = inventory(models_dir=models_dir, include_cache=include_cache)
    if json_out:
        out = [
            {
                "name": r.name, "short_name": r.short_name,
                "obs_spec": r.obs_spec, "n_stack": r.n_stack,
                "parent": r.parent, "parent_chain_total": r.parent_chain_total,
                "final_steps": r.final_steps, "source": r.source,
                "path": str(r.path),
                "wandb_alias": r.wandb_alias, "wandb_version": r.wandb_version,
                "has_model_doc": r.has_model_doc,
            }
            for r in rows
        ]
        typer.echo(json_module.dumps(out, indent=2))
        return

    table = Table(title=f"Inventory ({len(rows)} models)", show_lines=False)
    table.add_column("Short name")
    table.add_column("Obs spec")
    table.add_column("n_stack", justify="right")
    table.add_column("Chain steps", justify="right")
    table.add_column("Source")
    table.add_column("W&B")
    table.add_column("Doc")
    for r in rows:
        wandb_str = f"{r.wandb_alias}:{r.wandb_version}" if r.wandb_alias else "—"
        table.add_row(
            r.short_name, r.obs_spec, str(r.n_stack),
            f"{r.parent_chain_total:,}", r.source,
            wandb_str, "✓" if r.has_model_doc else "—",
        )
    Console().print(table)
```

- [ ] **Step 2: Implement `dsim/commands/obs_preflight.py`**

```python
"""dsim obs-preflight — check parent ↔ child obs-spec compat without loading weights.

Exit codes:
    0 — compatible (no surgery needed)
    1 — incompatible but surgery would resolve (warm_start required)
    2 — incompatible AND the diff suggests something else is wrong
       (e.g. unknown spec name, missing parent .hydra/)
"""
from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from core.obs_compat import preflight

app = typer.Typer(help="Obs-spec compatibility preflight (no model load).")


@app.callback(invoke_without_command=True)
def main(
    parent: str = typer.Option(..., "--parent",
                               help="Parent: filesystem path or wandb:// URI"),
    child_obs: str = typer.Option(..., "--child-obs",
                                  help="Child obs spec name (e.g. DUEL_V2_WORLD)"),
    child_n_stack: int = typer.Option(1, "--child-n-stack",
                                      help="Child frame-stack depth"),
) -> None:
    try:
        report = preflight(parent, child_obs, child_n_stack)
    except (FileNotFoundError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)

    cons = Console()
    cons.print(
        f"parent: [bold]{report.parent_spec_name}[/] × n_stack={report.parent_n_stack}  →  "
        f"child: [bold]{report.child_spec_name}[/] × n_stack={report.child_n_stack}"
    )

    table = Table(show_lines=False)
    table.add_column("Block")
    table.add_column("Dim", justify="right")
    table.add_column("Parent frame")
    table.add_column("Child frame")
    table.add_column("Status")
    glyph = {"matched": "✅", "frame_changed": "⚠️ ",
             "removed": "❌ removed", "added": "❌ added"}
    for d in report.diff:
        table.add_row(d.block, str(d.dim),
                      d.parent_frame or "—",
                      d.child_frame or "—",
                      glyph.get(d.status, d.status))
    cons.print(table)

    if report.compatible:
        cons.print("[green]✓ compatible[/] — init.mode=pretrain will load cleanly.")
        raise typer.Exit(code=0)
    cons.print("[yellow]⚠ surgery required[/] — set init.mode=warm_start.")
    raise typer.Exit(code=1)
```

- [ ] **Step 3: Implement `dsim/commands/obs_specs.py`**

```python
"""dsim obs-specs — pretty-print every entry in SPEC_BY_NAME."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from envs.quidditch.obs_spec import SPEC_BY_NAME

app = typer.Typer(help="Pretty-print the canonical obs spec catalog.")


@app.callback(invoke_without_command=True)
def main() -> None:
    cons = Console()
    for name, spec in SPEC_BY_NAME.items():
        cons.print(f"\n[bold]{name}[/] ({spec.dim}-d)")
        t = Table(show_lines=False)
        t.add_column("Slot")
        t.add_column("Block")
        t.add_column("Dim", justify="right")
        t.add_column("Frame")
        t.add_column("Notes")
        off = 0
        for b in spec.blocks:
            t.add_row(f"{off}:{off + b.dim}", b.name, str(b.dim),
                      b.frame or "—", b.notes or "")
            off += b.dim
        cons.print(t)
```

- [ ] **Step 4: Implement `dsim/commands/describe_run.py`**

```python
"""dsim describe-run — display the existing MODEL.md for a run.

Does NOT regenerate (per design decision 2026-05-18).  If MODEL.md is
absent, exit 2 with a pointer to scripts.render_model_doc.
"""
from __future__ import annotations

from pathlib import Path

import typer

from core.run_listing import resolve_trial

app = typer.Typer(help="Display the existing MODEL.md for a run.")


@app.callback(invoke_without_command=True)
def main(
    run_name: str = typer.Argument(..., help="Run name (e.g. ppo_hoop_blue_5)"),
    trial: str | None = typer.Option(None, "--trial",
                                     help="Specific trial; default: latest"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
) -> None:
    # Prefer models/<run_name>/MODEL.md (promoted); fall back to runs/.
    model_md = models_dir / run_name / "MODEL.md"
    if model_md.exists():
        typer.echo(model_md.read_text())
        return

    try:
        trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)

    candidate = trial_dir / "MODEL.md"
    if not candidate.exists():
        typer.echo(
            f"error: no MODEL.md under {trial_dir}\n"
            f"  generate one: python -m scripts.render_model_doc --run-dir {trial_dir}",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(candidate.read_text())
```

- [ ] **Step 5: Register the four subcommands in `dsim/cli.py`**

Edit `dsim/cli.py` to add (between the `app = typer.Typer(...)` declaration and the `def main()` function):

```python
from dsim.commands import (
    inventory as _inventory_cmd,
    obs_preflight as _obs_preflight_cmd,
    obs_specs as _obs_specs_cmd,
    describe_run as _describe_run_cmd,
)

app.add_typer(_inventory_cmd.app, name="inventory")
app.add_typer(_obs_preflight_cmd.app, name="obs-preflight")
app.add_typer(_obs_specs_cmd.app, name="obs-specs")
app.add_typer(_describe_run_cmd.app, name="describe-run")
```

- [ ] **Step 6: Write the CLI tests**

`tests/dsim/test_inventory_cli.py`:

```python
"""Smoke: dsim inventory renders + --json round-trips."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dsim.cli import app


def test_inventory_table_renders(tmp_path: Path) -> None:
    # Use a synthetic models/ dir so we don't depend on real promoted models.
    from omegaconf import OmegaConf
    d = tmp_path / "models" / "ppo_hoop_blue_4_20260511_202612"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "ppo_hoop_blue_4",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
        "init": {"mode": "scratch"},
    }), d / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create({"final_steps": 1, "parent_chain_total": 1}),
                   d / ".hydra" / "meta.yaml")

    runner = CliRunner()
    result = runner.invoke(app, ["inventory", "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0, result.output
    assert "blue_4" in result.output
    assert "DUEL_V2_WORLD" in result.output


def test_inventory_json_round_trips(tmp_path: Path) -> None:
    from omegaconf import OmegaConf
    d = tmp_path / "models" / "ppo_hoop_blue_4_20260511_202612"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "ppo_hoop_blue_4",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
    }), d / ".hydra" / "config.yaml")

    runner = CliRunner()
    result = runner.invoke(app, ["inventory", "--json",
                                 "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 1
    assert rows[0]["short_name"] == "blue_4"
```

`tests/dsim/test_obs_preflight_cli.py`:

```python
"""Smoke + exit-code contract for dsim obs-preflight."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def _make_parent(tmp: Path, obs: str, n_stack: int) -> Path:
    d = tmp / "parent"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "p", "obs": {"name": obs, "n_stack": n_stack},
    }), d / ".hydra" / "config.yaml")
    return d


def test_compatible_exits_zero(tmp_path: Path) -> None:
    p = _make_parent(tmp_path, "DUEL_V2_WORLD", 3)
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(p),
        "--child-obs", "DUEL_V2_WORLD", "--child-n-stack", "3",
    ])
    assert result.exit_code == 0
    assert "compatible" in result.output


def test_surgery_required_exits_one(tmp_path: Path) -> None:
    p = _make_parent(tmp_path, "DUEL_V1_BODY", 1)
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(p),
        "--child-obs", "DUEL_V2_WORLD", "--child-n-stack", "3",
    ])
    assert result.exit_code == 1
    assert "surgery required" in result.output


def test_missing_parent_exits_two(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, [
        "obs-preflight", "--parent", str(tmp_path / "nope"),
        "--child-obs", "DUEL_V2_WORLD",
    ])
    assert result.exit_code == 2
```

`tests/dsim/test_obs_specs_cli.py`:

```python
from typer.testing import CliRunner
from dsim.cli import app


def test_obs_specs_lists_known_specs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["obs-specs"])
    assert result.exit_code == 0
    # At least three known specs should appear.
    assert "DUEL_V1_BODY" in result.output
    assert "DUEL_V2_WORLD" in result.output
    assert "DUEL_V3_BODY_EGO" in result.output
```

`tests/dsim/test_describe_run_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from dsim.cli import app


def test_describe_run_prints_existing_model_md(tmp_path: Path) -> None:
    md = tmp_path / "models" / "ppo_hoop_blue_4" / "MODEL.md"
    md.parent.mkdir(parents=True)
    md.write_text("# MODEL: ppo_hoop_blue_4\n\nstub")
    runner = CliRunner()
    result = runner.invoke(app, ["describe-run", "ppo_hoop_blue_4",
                                 "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0
    assert "ppo_hoop_blue_4" in result.output


def test_describe_run_errors_without_model_md(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["describe-run", "nope",
                                 "--models-dir", str(tmp_path / "models"),
                                 "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 2
    assert "render_model_doc" in result.output
```

- [ ] **Step 7: Run, fix until green, commit**

```bash
conda run -n uav python -m pytest tests/dsim/ -v
conda run -n uav dsim --help
conda run -n uav dsim inventory --help
conda run -n uav dsim obs-preflight --help
```

Expected: tests pass; help text renders.

```bash
git add dsim/commands/{inventory,obs_preflight,obs_specs,describe_run}.py dsim/cli.py tests/dsim/
git commit -m "feat(dsim): read-only subcommands (inventory, obs-preflight, obs-specs, describe-run)

Four Typer subcommands, all thin wrappers over core/ functions:
  - dsim inventory       → core.inventory
  - dsim obs-preflight   → core.obs_compat.preflight  (exit codes: 0/1/2)
  - dsim obs-specs       → pretty-prints SPEC_BY_NAME
  - dsim describe-run    → cats existing MODEL.md (no regen, per design 2026-05-18)

Rich tables for inventory + obs-preflight; --json on inventory for
TUI / script consumption.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: `dsim` subcommands — dispatch (lineage, list-runs, resume, promote, sweep)

**Files:**
- Create: `dsim/commands/lineage.py`
- Create: `dsim/commands/list_runs.py`
- Create: `dsim/commands/resume.py`
- Create: `dsim/commands/promote.py`
- Create: `dsim/commands/sweep.py`
- Modify: `dsim/cli.py` (register the new subcommands)
- Create: `tests/dsim/test_lineage_cli.py`
- Create: `tests/dsim/test_list_runs_cli.py`
- Create: `tests/dsim/test_resume_cli.py`
- Create: `tests/dsim/test_promote_cli.py`
- Create: `tests/dsim/test_sweep_cli.py`

- [ ] **Step 1: Implement each command** — each ≤50 lines.

`dsim/commands/lineage.py`:

```python
from __future__ import annotations

import typer

from core.lineage import walk_chain_local, walk_chain_wandb

app = typer.Typer(help="Walk a run's pretrain ancestry.")


@app.callback(invoke_without_command=True)
def main(
    target: str = typer.Option(..., "--target",
                               help="Filesystem path or wandb:// URI"),
    local: bool = typer.Option(False, "--local", help="Walker A only"),
    both: bool = typer.Option(False, "--both",
                              help="Both walkers side-by-side"),
) -> None:
    def render(chain):
        for i, n in enumerate(chain):
            ind = "  " * i
            sfx = "  (truncated)" if n.truncated else ""
            typer.echo(f"{ind}{n.name}  steps={n.final_steps}  chain={n.parent_chain_total}{sfx}")

    if local:
        render(walk_chain_local(target))
        return
    if both:
        typer.echo("--- local ---"); render(walk_chain_local(target))
        typer.echo("\n--- wandb ---")
        try:
            render(walk_chain_wandb(target))
        except Exception as e:
            typer.echo(f"(wandb failed: {e})")
        return
    if target.startswith(("wandb://", "wandb-artifact://")):
        try:
            render(walk_chain_wandb(target))
            return
        except Exception as e:
            typer.echo(f"(wandb failed: {e}; falling back to local)")
    render(walk_chain_local(target))
```

`dsim/commands/list_runs.py`:

```python
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.run_listing import list_runs

app = typer.Typer(help="Enumerate runs/ with latest trial + checkpoint.")


@app.callback(invoke_without_command=True)
def main(
    run_filter: str = typer.Option(None, "--run",
                                   help="Filter by substring of run name"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    rows = list_runs(runs_dir=runs_dir, run_filter=run_filter)
    table = Table(title=f"Runs ({len(rows)})")
    table.add_column("Run name")
    table.add_column("Latest trial")
    table.add_column("Latest checkpoint")
    for r in rows:
        ck = r.latest_checkpoint.name if r.latest_checkpoint else "—"
        table.add_row(r.run_name, r.latest_trial.name, ck)
    Console().print(table)
```

`dsim/commands/resume.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from core.run_listing import resolve_checkpoint, resolve_trial

app = typer.Typer(help="Resume a training run from its latest checkpoint.")


@app.callback(invoke_without_command=True)
def main(
    run_name: str = typer.Argument(..., help="Run name"),
    trial: str = typer.Option(None, "--trial",
                              help="Specific trial; default: latest"),
    ckpt: str = typer.Option(None, "--ckpt",
                             help="Specific checkpoint; default: highest-step"),
    exp: str = typer.Option(None, "--exp",
                            help="conf/experiment override; default: same as parent's"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    ckpt_path = resolve_checkpoint(trial_dir, ckpt=ckpt)

    # Read the parent trial's hydra-choices to determine which experiment
    # YAML to compose with on resume.
    from omegaconf import OmegaConf
    hydra_meta = OmegaConf.load(trial_dir / ".hydra" / "hydra.yaml")
    parent_exp = hydra_meta.hydra.runtime.choices.get("experiment") if exp is None else exp
    if parent_exp is None:
        typer.echo(f"error: cannot infer experiment for {run_name}; pass --exp",
                   err=True)
        raise typer.Exit(code=2)

    cmd = [
        sys.executable, "-m", "scripts.train",
        f"+experiment={parent_exp}",
        "init.mode=resume",
        f"init.parent={ckpt_path}",
    ]
    typer.echo(f"$ {' '.join(cmd)}")
    raise typer.Exit(code=subprocess.call(cmd))
```

`dsim/commands/promote.py`:

```python
from __future__ import annotations

from pathlib import Path

import typer

from core.promote import promote_run
from core.run_listing import resolve_trial

app = typer.Typer(help="Promote a run's best model to canonical/vendored.")


@app.callback(invoke_without_command=True)
def main(
    run_name: str = typer.Argument(..., help="Run name"),
    trial: str = typer.Option(None, "--trial"),
    alias: str = typer.Option("prod", "--alias"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    trial_dir = resolve_trial(run_name, trial=trial, runs_dir=runs_dir)
    result = promote_run(trial_dir, alias=alias)
    typer.echo(f"promoted {result.run_name} ({result.wandb_version}, alias={result.wandb_alias})")
    typer.echo(f"  copied {len(result.copied_files)} files into {result.target_dir}")
    typer.echo("git add models/ && git commit -m 'model: promote ...' to vendor.")
```

`dsim/commands/sweep.py`:

```python
"""dsim sweep: create / agent / agents — thin wrappers over wandb CLI."""
from __future__ import annotations

import subprocess

import typer

app = typer.Typer(help="W&B sweep controller subcommands.")


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Sweep YAML name (sweeps/<name>.yaml)"),
) -> None:
    cmd = ["wandb", "sweep", f"sweeps/{name}.yaml"]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("agent")
def agent(
    sweep_id: str = typer.Argument(...),
) -> None:
    cmd = ["wandb", "agent", sweep_id]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("agents")
def agents(
    sweep_id: str = typer.Argument(...),
    n: int = typer.Option(1, "--n", "-n", help="Number of parallel agents"),
) -> None:
    procs = [
        subprocess.Popen(["wandb", "agent", sweep_id])
        for _ in range(n)
    ]
    rcs = [p.wait() for p in procs]
    raise typer.Exit(code=max(rcs))
```

- [ ] **Step 2: Register the five subcommands in `dsim/cli.py`**

```python
from dsim.commands import (
    lineage as _lineage_cmd,
    list_runs as _list_runs_cmd,
    resume as _resume_cmd,
    promote as _promote_cmd,
    sweep as _sweep_cmd,
)

app.add_typer(_lineage_cmd.app, name="lineage")
app.add_typer(_list_runs_cmd.app, name="list-runs")
app.add_typer(_resume_cmd.app, name="resume")
app.add_typer(_promote_cmd.app, name="promote")
app.add_typer(_sweep_cmd.app, name="sweep")
```

- [ ] **Step 3: Write CLI smoke tests**

For each command, a small test that asserts `--help` works and the dispatch shape is right. Example — `tests/dsim/test_list_runs_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from dsim.cli import app


def test_list_runs_renders(tmp_path: Path) -> None:
    (tmp_path / "runs" / "blue_5" / "20260514_120000").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 0
    assert "blue_5" in result.output
```

Similar smoke tests for `lineage`, `resume` (with the subprocess monkeypatched), `promote` (monkeypatch `core.promote.promote_run`), `sweep` (monkeypatch `subprocess.call` to return 0).

- [ ] **Step 4: Run, commit**

```bash
conda run -n uav python -m pytest tests/dsim/ -v
conda run -n uav dsim --help        # all subcommands listed
```

```bash
git add dsim/commands/{lineage,list_runs,resume,promote,sweep}.py dsim/cli.py tests/dsim/test_*_cli.py
git commit -m "feat(dsim): dispatch subcommands (lineage, list-runs, resume, promote, sweep)

Five Typer subcommands, each ≤50 lines, all wrapping core/ functions
or shelling out:
  - dsim lineage      → core.lineage walkers
  - dsim list-runs    → core.run_listing.list_runs
  - dsim resume       → resolves trial+ckpt, shells out to scripts.train
  - dsim promote      → core.promote.promote_run
  - dsim sweep        → wandb sweep / agent / agents (parallel agents
                         via subprocess.Popen)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: `scripts/train.py` — obs-preflight warn-then-raise guard rail

When `cfg.init.mode == "pretrain"` and the parent's obs spec doesn't match the child's, currently `check_obs_compat` strict-raises mid-init (per `scripts/_train_common.py:check_obs_compat`'s `sys.exit(2)` path). Add an earlier preflight call so the user sees the friendly diff *first*, then the existing strict-raise still fires.

**Files:**
- Modify: `scripts/train.py`
- Create: `tests/scripts/test_train_preflight_guard.py`

- [ ] **Step 1: Locate the pretrain branch in `scripts/train.py`**

Read `scripts/train.py` and find where `cfg.init.mode == "pretrain"` is handled. It will be the branch that calls `PPO.load(...)` and (currently) the strict `check_obs_compat`.

- [ ] **Step 2: Insert the preflight call before the load**

```python
# scripts/train.py — inside the init-mode dispatch, in the pretrain branch:

if cfg.init.mode == "pretrain":
    from core.obs_compat import preflight

    parent_uri = str(cfg.init.parent)
    child_obs = str(cfg.obs.name)
    child_n_stack = int(cfg.obs.get("n_stack", 1))

    try:
        report = preflight(parent_uri, child_obs, child_n_stack)
    except Exception as e:
        log.warning("obs-preflight could not be run (%s); falling through to "
                    "load — the existing check_obs_compat will catch any "
                    "real mismatch.", e)
        report = None

    if report is not None and not report.compatible:
        # Surface the friendly diff before strict_compat raises.
        log.warning(
            "obs-spec preflight WARNING: parent %s × n_stack=%d  →  "
            "child %s × n_stack=%d  (surgery_required=%s)",
            report.parent_spec_name, report.parent_n_stack,
            report.child_spec_name, report.child_n_stack,
            report.surgery_required,
        )
        for d in report.diff:
            log.warning("  %-12s dim=%d  parent_frame=%s child_frame=%s  status=%s",
                        d.block, d.dim, d.parent_frame, d.child_frame, d.status)
        log.warning("init.mode=pretrain will strict-raise below.  "
                    "Use init.mode=warm_start to small-init the mismatched "
                    "columns instead.")
        # Do NOT auto-switch — that's "too magical" per design decision 3.

    # … then the existing PPO.load + check_obs_compat call continues unchanged.
```

The point is **warn first, then let the existing strict-raise fire normally**. The behavior on the failure path is the same; we just emit a clearer diff before it.

- [ ] **Step 3: Write a test**

`tests/scripts/test_train_preflight_guard.py`:

```python
"""When init.mode=pretrain + obs mismatch, the preflight warning fires
before the strict-raise."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_pretrain_with_obs_mismatch_logs_preflight_then_exits(tmp_path) -> None:
    # blue_4 is DUEL_V2_WORLD n_stack=3; tell the experiment to use
    # DUEL_V1_BODY n_stack=1 (which is a real spec from blue_v1/red_v1
    # but incompatible).  pretrain (not warm_start) ⇒ strict-raise expected.
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train",
         "+experiment=canary_team",
         "init.mode=pretrain",
         "init.parent=models/ppo_hoop_blue_4_20260511_202612/best_model",
         "obs=duel_v1_body",   # forces a mismatch vs the V2 parent
         "trainer.total_timesteps=10",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    # Should exit non-zero (the existing check_obs_compat strict-raises).
    assert result.returncode != 0, result.stdout
    # And the friendly preflight warning must appear before the strict-raise:
    out = result.stdout + result.stderr
    assert "preflight WARNING" in out
    assert "surgery_required" in out
```

- [ ] **Step 4: Run + canary regression**

```bash
conda run -n uav python -m pytest tests/scripts/test_train_preflight_guard.py -v -m slow
make test   # canary stays green
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/scripts/test_train_preflight_guard.py
git commit -m "feat(train): warn-then-raise obs-preflight in init.mode=pretrain path

Runs core.obs_compat.preflight() before PPO.load, so users see the
friendly diff (matched / frame_changed / removed / added rows) ahead
of the existing check_obs_compat strict-raise.  No auto-switch to
warm_start — design decision 2026-05-18 calls auto-switching too magical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Makefile reshape — chores only + `make train`

**Files:**
- Modify: `Makefile`
- Verify: `make help`, `make tui`, `make train EXP=canary_team OVERRIDES="trainer.total_timesteps=10"`

- [ ] **Step 1: Read current `Makefile` to identify what stays vs goes**

Currently ~20 targets. Per spec section "Final Makefile — 8 targets":
- KEEP: `help`, `install`, `clean`, `test`, `test-fast`, `test-warm`, `tui`, `train`
- REMOVE: `demo`, `camera-test`, `eval`, `eval-headless`, `eval-team`, `resume`, `lineage`, `promote`, `list-runs`, `obs-specs`, `describe-run`, `sweep`, `sweep-agent`, `sweep-agents`

`make tui` is new (`make train` keeps muscle memory; other migrations land in Task 18 docs).

- [ ] **Step 2: Replace `Makefile` body**

```makefile
# Quidditch-Sim Makefile — Slice 1 (Part 3 ML infra) reshape.
#
# Day-to-day commands moved to:
#   dsim --help                            inspection + dispatch (Typer CLI)
#   python -m scripts.train +experiment=X  Hydra entrypoint (composable)
#   python -m scripts.eval_team  …         Hydra entrypoint
#   python -m scripts.eval_battery …       Hydra entrypoint
#   python -m scripts.eval_ppo   …         Hydra entrypoint
#   make tui                               opens the controller TUI (Slice 2)

CONDA_ENV  ?= uav
EXP        ?=
OVERRIDES  ?=

CONDA := $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null))
ifeq ($(CONDA),)
$(error conda not found — activate a conda shell or set CONDA_EXE)
endif

CONDA_RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python
MJPYTHON  := $(CONDA_RUN) mjpython

.PHONY: help install clean test test-fast test-warm tui train

.DEFAULT_GOAL := help

help: ## 📋 Show targets + pointers to dsim and Hydra entrypoints
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Other day-to-day commands (no make wrapper):"
	@echo "  dsim --help                              List dsim subcommands"
	@echo "  python -m scripts.eval_team +eval_team=default learner=blue \\"
	@echo "      learner.uri=<…> opponent=beeline_red"
	@echo "  python -m scripts.eval_battery +eval_battery=default candidate=<uri>"
	@echo ""

install: ## ⚙️  Create conda env + install dsim editable
	conda env create -f environment.yml || conda env update -f environment.yml
	$(CONDA_RUN) pip install -e .

clean: ## 🧹 Remove build artifacts and __pycache__
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

test: ## ✅ Full test suite (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip @pytest.mark.slow)
	@$(PYTHON) -m pytest -m "not slow"

test-warm: ## ✅ Warm-start integration test  MODEL=<run-name>
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see 'dsim inventory')"; exit 1; }; \
	 MODEL="$(MODEL)" $(PYTHON) -m pytest tests/core/policies/test_warm_start.py

tui: ## 🖼  Open the controller TUI (Slice 2; subprocess slots use mjpython when needed)
	@$(PYTHON) -m dsim tui

train: ## 🚀 Launch a training run  EXP=<name> [OVERRIDES="key=val key=val"]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required (ls conf/experiment/)"; exit 1; }
	@$(PYTHON) -m scripts.train +experiment=$(EXP) $(OVERRIDES)
```

Note: `make tui` references `dsim tui` which isn't implemented until Slice 2. For Slice 1, leave the target in place; running it will fail with a clear "no such command" error until Slice 2 lands. (Alternative: add a stub `dsim tui` subcommand in Task 14 that prints "Slice 2 not yet landed; use the Hydra entrypoints directly." but that's optional polish.)

- [ ] **Step 3: Verify the Makefile**

```bash
make help                                                # lists all 8 targets cleanly
make test-fast                                           # canary regression gate
make train EXP=canary_team OVERRIDES="trainer.total_timesteps=10"  # smoke
```

Expected: help renders with the new pointers; `test-fast` PASSES; the smoke train runs to ~10 steps without error.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "refactor(make): chores-only Makefile (+ make train kept for muscle memory)

Per spec section 'Final Makefile — 8 targets'.  Removed 13 day-to-day
targets (demo, camera-test, eval, eval-headless, eval-team, resume,
lineage, promote, list-runs, obs-specs, describe-run, sweep,
sweep-agent, sweep-agents) — all replaced by dsim or direct Hydra
invocations per Task 18's docs.

make tui references Slice 2's dsim tui subcommand; works once that
ships.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Docs — README + CLAUDE.md + brain updates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify (brain, NOT in git, lives outside repo/): `../../brain/index.md`, `../../brain/changelog.md`, `../../brain/decisions.md`, `../../brain/tasks.md`

- [ ] **Step 1: Add a "CLI surface" section near the top of `README.md`**

Insert just below the existing intro/install section:

```markdown
## CLI surface

Three command surfaces, by purpose:

| Purpose | Surface | Examples |
|---|---|---|
| Inspection / read-only / one-shot dispatch | **`dsim`** (Typer) | `dsim inventory`, `dsim obs-preflight --parent X --child-obs Y`, `dsim lineage --target ...`, `dsim list-runs`, `dsim resume <run-name>`, `dsim promote <run-name>`, `dsim describe-run <run-name>`, `dsim obs-specs`, `dsim sweep create <name>` |
| Composable runs (training, eval, battery, sweep) | **Hydra apps** (`python -m scripts.X`) | `python -m scripts.train +experiment=blue_v5`, `python -m scripts.eval_team +eval_team=default learner=blue learner.uri=<uri> opponent=beeline_red`, `python -m scripts.eval_battery +eval_battery=default candidate=<uri>`, `python -m scripts.eval_ppo +eval_ppo=default model_uri=<uri>` |
| Chores (install, test, train one-off, TUI) | **`make`** | `make install`, `make test`, `make test-fast`, `make test-warm MODEL=<x>`, `make train EXP=blue_v5`, `make tui` |

Inspection and dispatch went to `dsim` for discoverability (`dsim --help` lists everything). Hydra apps cover anything with composable configs. Make is reserved for true chores plus the one-off `make train EXP=X` muscle-memory shortcut.

### Replacing removed `make` targets

| Old | New |
|---|---|
| `make demo` | TUI → Demo task; or `mjpython demo/menu.py <key>` directly |
| `make camera-test CAM=<x>` | TUI → Camera Test; or `python demo/camera_test.py --cam <x>` |
| `make eval` | `python -m scripts.eval_ppo +eval_ppo=default model_uri=<uri>` |
| `make eval-team LEARNER=… BLUE=… RED=… GUI=1` | `python -m scripts.eval_team +eval_team=default learner=<side> learner.uri=<uri> opponent=<choice> eval.gui=true` |
| `make resume RUN_NAME=…` | `dsim resume <run-name>` |
| `make lineage RUN_NAME=…` | `dsim lineage --target models/ppo_hoop_<name>_*/best_model` |
| `make promote RUN_NAME=…` | `dsim promote <run-name>` |
| `make list-runs` | `dsim list-runs` |
| `make obs-specs` | `dsim obs-specs` |
| `make describe-run RUN_NAME=…` | `dsim describe-run <run-name>` |
| `make sweep SWEEP=…` / `sweep-agents ID=… N=…` | `dsim sweep create <name>` / `dsim sweep agents <id> --n N` |
```

- [ ] **Step 2: Add a brief "CLI surface" pointer to `CLAUDE.md`**

```markdown
## CLI surface (Slice 1 of ML infra Part 3, landed 2026-05-XX)

Three surfaces by purpose:
- `dsim <cmd>` — Typer CLI for inspection (inventory, lineage, list-runs, obs-preflight, obs-specs, describe-run) and dispatch (resume, promote, sweep). Single binary, `dsim --help` lists all.
- `python -m scripts.<name>` — Hydra apps for composable runs: `train`, `eval_team`, `eval_ppo`, `eval_battery`.
- `make <target>` — chores only: `install`, `clean`, `test*`, `tui`, `train`.

See `README.md` "CLI surface" section for the migration map.
```

- [ ] **Step 3: Update brain files**

`brain/changelog.md` (top of file): add a 2026-05-XX entry summarizing Slice 1 — what landed, which old surfaces went away, which canaries stayed green.

`brain/index.md` "Recent Context": add a brief summary of the Slice 1 landing. `brain/index.md` "Active Priorities": remove the now-resolved obs-preflight / inventory / battery items.

`brain/decisions.md`: add ADR entries for the five resolved decisions from the spec's "Resolved decisions" section.

`brain/tasks.md`: mark the Slice 1 line item complete; add the Slice 2 TUI plan as the next focus.

These brain edits do NOT commit (brain/ lives outside the git repo).

- [ ] **Step 4: Commit the docs**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README + CLAUDE.md — Slice 1 CLI surface + migration map

Document the three CLI surfaces (dsim / Hydra apps / make) and the
migration map for the 13 removed make targets.  CLAUDE.md gets a
short pointer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Final regression gate**

```bash
make test
```

Expected: all tests pass (the canaries — single-agent step 434/reward 7.3837 + team canary — must remain byte-identical through Slice 1).

- [ ] **Step 6: Push the branch + open the PR**

```bash
git push -u origin feature/ml-infra-part-3
gh pr create --title "ML infra Part 3 — Slice 1: engine + dsim CLI" --body "$(cat <<'EOF'
## Summary
- `core/` services: `inventory`, `obs_compat`, `eval_core`, `eval_report`, `lineage`, `promote`, `run_listing`, `run_context`
- Hydra migration of `eval_team.py` + `eval_ppo.py`; new `eval_battery.py` with local Markdown/CSV/JSONL reports
- `dsim` Typer CLI: 9 subcommands (inventory, obs-preflight, obs-specs, describe-run, lineage, list-runs, resume, promote, sweep) + `dsim tui` placeholder for Slice 2
- `scripts/train.py` warn-then-raise obs-preflight guard rail (warns before the existing strict-raise; no auto-switch)
- Makefile chopped to 8 chore targets (+ `make train` for muscle memory)
- README + CLAUDE.md updated with CLI surface section + migration map

## Test plan
- [ ] `make test` — full suite passes; canaries byte-identical
- [ ] `dsim inventory` against real `models/` shows 7 promoted models
- [ ] `dsim obs-preflight --parent models/ppo_hoop_blue_4_*/best_model --child-obs DUEL_V2_WORLD --child-n-stack 3` exits 0
- [ ] `python -m scripts.eval_battery +eval_battery=quick candidate=models/ppo_hoop_blue_4_20260511_202612/best_model` writes a 3-scenario report
- [ ] `dsim describe-run ppo_hoop_blue_4_20260511_202612` cats the existing MODEL.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opens; CI passes; ready for review.

---

## Plan self-review checklist (run after writing — fix inline, then move on)

1. **Spec coverage** — every Slice 1 section from the spec is implemented:
   - core/inventory.py ✓ (Task 3)
   - core/obs_compat.py ✓ (Task 4)
   - core/eval_core.py ✓ (Task 8)
   - core/eval_report.py ✓ (Task 12)
   - scripts/eval_team.py Hydra migration ✓ (Task 10)
   - scripts/eval_ppo.py Hydra migration ✓ (Task 11)
   - scripts/eval_battery.py + conf/eval_battery/ ✓ (Task 13)
   - conf/eval/, conf/learner/, conf/eval_team/, conf/eval_ppo/ ✓ (Task 9)
   - dsim package + 9 subcommands ✓ (Tasks 1, 14, 15)
   - resolve_parent metadata_only ✓ (Task 4)
   - scripts/lineage.py refactor ✓ (Task 5)
   - scripts/promote.py refactor ✓ (Task 6)
   - core/run_listing.py ✓ (Task 7)
   - scripts/train.py guard rail ✓ (Task 16)
   - Makefile reshape ✓ (Task 17)
   - README + CLAUDE.md docs ✓ (Task 18)
   - All five resolved decisions implemented (preflight warn-then-raise, dsim describe-run displays existing MODEL.md, dsim sweep create/agent/agents, --include-cache off by default, [e] edit overrides deferred to Slice 2 since Slice 1 has no TUI).

2. **Type consistency**:
   - `ScenarioSpec.opponent_model_path` is `str | None` in `core.eval_core` and in `conf/eval_battery/*.yaml` (matches; YAMLs use `null` which OmegaConf maps to None).
   - `PreflightReport` field names (`compatible`, `surgery_required`, `diff`, etc.) match across `core/obs_compat.py`, `dsim/commands/obs_preflight.py`, and `scripts/train.py`'s guard rail.
   - `ModelInfo` field names match across `core/inventory.py`, `dsim/commands/inventory.py`, and `scripts/eval_battery.py:_candidate_summary_from_ctx` (which uses inventory field names verbatim).
   - `resolve_parent(metadata_only=True)` signature matches between `scripts/_artifact_io.py` and `core/obs_compat.py`'s call site.

3. **No placeholders**: spot-check each task for "TBD", "TODO", "implement later". The `???` in YAML cfg files is OmegaConf's required-field marker, intentional. The `raise NotImplementedError("port from …")` markers in Task 5 (walk_chain_wandb) and Task 6 (promote_run body) and Task 8 (run_scenario body) are signposts for the executor — each is paired with an explicit pointer to the source code to port from, with detail on what to preserve. Acceptable as long as the executor reads that source to implement the body.
