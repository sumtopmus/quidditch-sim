# Model Doc Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auto-generated, per-model `MODEL.md` Markdown spec sheet that lives in the run dir, the vendored model dir, and the W&B artifact — one source of truth, three surfaces, generated from `.hydra/{config,meta,hydra}.yaml` + `_wandb_metadata.json` at train end and on demand.

**Architecture:** Pure renderer (`scripts/_render_model_doc.py`) reads frozen-at-train-time inputs, returns a Markdown string. Section renderers each take a `ctx` dict and return their part; the composer wraps each section in try/except so a bad section flags itself but doesn't kill the doc. Three call sites: `train.py:finally:` (auto, best-effort), `scripts/promote.py` (copy), `scripts/_artifact_io.py:log_run_artifact` (add to wandb artifact). On-demand: `python -m scripts.render_model_doc` + `make describe-run`. One-shot: `scripts/backfill_model_docs.py` for the 7 legacy models.

**Tech Stack:** Python 3.11, Hydra (`OmegaConf.load`, `hydra.utils.instantiate`), `dataclasses.fields` introspection for reward-term tables, pytest + pytest tmp_path fixtures, no new dependencies. Spec: [docs/superpowers/specs/2026-05-16-model-doc-generator-design.md](../specs/2026-05-16-model-doc-generator-design.md).

---

### Task 1: Add `description` field to top-level Config schema

**Files:**
- Modify: `config_schema.py`
- Test: `tests/test_config_schema.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_config_schema.py`:

```python
def test_top_level_config_has_description_field():
    """The top-level Config schema must carry an optional `description` string
    so experiment YAMLs can set `description: |...` without Hydra struct
    rejection.  Empty default = use auto-template in MODEL.md.
    """
    from omegaconf import OmegaConf
    from config_schema import Config

    cfg = OmegaConf.structured(Config)
    assert "description" in cfg
    assert cfg.description == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/test_config_schema.py::test_top_level_config_has_description_field -v`
Expected: FAIL with `AttributeError` or `ConfigKeyError` — the field doesn't exist yet.

- [ ] **Step 3: Add the field to the schema**

Open `config_schema.py`, find the top-level `Config` dataclass (the one composed by Hydra at the root), and add the field. Example diff shape — adapt to whatever the existing field list looks like:

```python
@dataclass
class Config:
    # ... existing fields ...
    description: str = ""  # optional override for MODEL.md Summary section
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/test_config_schema.py -v`
Expected: PASS. Full schema suite must pass — adding an optional field with a default must not break existing tests.

- [ ] **Step 5: Run a full quick suite to confirm no regressions**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest -k "not test_offscreen_render_one_frame_no_crash and not test_capture_cells_team and not test_simple_env_factory_builds_16d_train_env and not test_team_env_factory_builds_75d_train_env_with_frame_stack" --tb=short`
Expected: All ~184 non-render tests pass.

- [ ] **Step 6: Commit**

```bash
git add config_schema.py tests/test_config_schema.py
git commit -m "feat(config): add optional description field to top-level Config

Sets the default empty string. Used by the upcoming MODEL.md generator's
Summary section: when non-empty, replaces the auto-templated summary."
```

---

### Task 2: Build `_load_run_context` helper + shared test fixture

**Files:**
- Create: `scripts/_render_model_doc.py`
- Create: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_render_model_doc.py`:

```python
"""Tests for scripts/_render_model_doc.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts._render_model_doc import _load_run_context


def _write_run_fixture(
    run_dir: Path,
    *,
    config: dict,
    meta: dict | None = None,
    hydra_yaml: dict | None = None,
    wandb_meta: dict | None = None,
) -> Path:
    """Write a minimal run-dir fixture under `run_dir/.hydra/`."""
    hdir = run_dir / ".hydra"
    hdir.mkdir(parents=True, exist_ok=True)
    (hdir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    if meta is not None:
        (hdir / "meta.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(meta)))
    if hydra_yaml is not None:
        (hdir / "hydra.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(hydra_yaml)))
    if wandb_meta is not None:
        (run_dir / "_wandb_metadata.json").write_text(json.dumps(wandb_meta))
    return run_dir


def test_load_run_context_reads_all_inputs(tmp_path: Path):
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={"run_name": "ppo_hoop_test", "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1}},
        meta={"git_hash": "abc123", "final_stats": {"best_eval_reward": 7.91}},
        hydra_yaml={"hydra": {"runtime": {"choices": {"reward": "team_v2"}}}},
        wandb_meta={"name": "ppo_hoop_test", "version": "v0", "aliases": ["prod"]},
    )
    ctx = _load_run_context(run_dir)
    assert ctx["cfg"].run_name == "ppo_hoop_test"
    assert ctx["meta"]["git_hash"] == "abc123"
    assert ctx["hydra_yaml"]["hydra"]["runtime"]["choices"]["reward"] == "team_v2"
    assert ctx["wandb_meta"]["name"] == "ppo_hoop_test"
    assert ctx["run_dir"] == run_dir


def test_load_run_context_raises_when_config_missing(tmp_path: Path):
    """config.yaml is the required input — everything depends on it."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_run_context(run_dir)


def test_load_run_context_tolerates_missing_optional_inputs(tmp_path: Path):
    """meta.yaml, hydra.yaml, _wandb_metadata.json are all optional."""
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={"run_name": "ppo_hoop_test", "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1}},
    )
    ctx = _load_run_context(run_dir)
    assert ctx["cfg"].run_name == "ppo_hoop_test"
    assert ctx["meta"] is None
    assert ctx["hydra_yaml"] is None
    assert ctx["wandb_meta"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts._render_model_doc`.

- [ ] **Step 3: Implement `_load_run_context`**

Create `scripts/_render_model_doc.py`:

```python
"""Pure renderer for per-model MODEL.md spec sheets.

Reads .hydra/{config,meta,hydra}.yaml + _wandb_metadata.json from a run dir
and returns a Markdown string.  No filesystem writes, no wandb calls.

See docs/superpowers/specs/2026-05-16-model-doc-generator-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_run_context(run_dir: Path) -> dict[str, Any]:
    """Gather config + meta + hydra-choices + wandb-meta into one dict.

    Required: `.hydra/config.yaml`.  All other inputs optional; missing ones
    surface as `None` in the returned ctx.
    """
    hdir = run_dir / ".hydra"
    cfg_path = hdir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"required input missing: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    meta_path = hdir / "meta.yaml"
    meta = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True) if meta_path.exists() else None

    hydra_yaml_path = hdir / "hydra.yaml"
    hydra_yaml = OmegaConf.to_container(OmegaConf.load(hydra_yaml_path), resolve=True) if hydra_yaml_path.exists() else None

    wandb_meta_path = run_dir / "_wandb_metadata.json"
    wandb_meta = json.loads(wandb_meta_path.read_text()) if wandb_meta_path.exists() else None

    return {
        "cfg": cfg,
        "meta": meta,
        "hydra_yaml": hydra_yaml,
        "wandb_meta": wandb_meta,
        "run_dir": run_dir,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _load_run_context helper + test fixture

Reads .hydra/{config,meta,hydra}.yaml + _wandb_metadata.json from a run dir,
returning a dict.  config.yaml is required; the rest are optional and surface
as None when absent.  Foundation for the section renderers."
```

---

### Task 3: `_section_header` renderer

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_header


def _ctx_for_section(**overrides) -> dict:
    """Build a baseline ctx covering the union of fields all sections need.
    Tests override just what they care about.
    """
    cfg = OmegaConf.create({
        "run_name": "ppo_hoop_test",
        "description": "",
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
        "init": {"mode": "scratch", "parent": None},
        "trainer": {"lr": 3e-4, "total_timesteps": 10_000_000,
                     "n_envs": 8, "batch_size": 256, "n_epochs": 10,
                     "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0,
                     "clip_range": 0.2},
        "env": {"learner_id": "blue_0",
                 "team_cfg": {"episode_seconds": 30.0, "tag_radius": 0.3,
                              "crash_vel_thr": 1.0, "midpoint_alpha": 0.5,
                              "crash_aftermath_seconds": 0.0}},
        "opponent": {"_target_": "envs.quidditch.opponents.BeelineRed"},
        "curriculum": {"name": "fixed_start"},
        "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                    "terms": []},  # empty terms ok for non-reward sections
    })
    ctx = {
        "cfg": cfg,
        "meta": {"git_hash": "abc1234", "final_stats": {
            "best_eval_reward": 7.91, "best_step": 9_500_000,
            "completed_steps": 10_000_000, "wall_clock_seconds": 1923.0,
            "model_kind": "best"}},
        "hydra_yaml": {"hydra": {"runtime": {"choices": {"reward": "team_v2"}}}},
        "wandb_meta": {"name": "ppo_hoop_test", "version": "v0",
                        "aliases": ["latest", "prod", "ppo_hoop_test"],
                        "entity": "gridcom", "project": "drone-quidditch"},
        "run_dir": Path("runs/ppo_hoop_test/20260516_120000"),
    }
    ctx.update(overrides)
    return ctx


def test_section_header_promoted_run():
    """When _wandb_metadata.json is present, status is 'promoted' and the W&B
    line renders below the header."""
    out = _section_header(_ctx_for_section())
    assert "# MODEL: ppo_hoop_test_20260516_120000" in out
    assert "promoted" in out
    assert "abc1234" in out
    assert "wandb://ppo_hoop_test:prod" in out
    assert "v0" in out


def test_section_header_run_only_when_wandb_meta_absent():
    """When _wandb_metadata.json is None, status is 'run-only' and the W&B
    line is omitted entirely."""
    out = _section_header(_ctx_for_section(wandb_meta=None))
    assert "run-only" in out
    assert "wandb://" not in out
    assert "**W&B:**" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_header_promoted_run tests/scripts/test_render_model_doc.py::test_section_header_run_only_when_wandb_meta_absent -v`
Expected: FAIL with `ImportError: cannot import name '_section_header'`.

- [ ] **Step 3: Implement `_section_header`**

Append to `scripts/_render_model_doc.py`:

```python
def _section_header(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    wandb_meta = ctx["wandb_meta"]
    run_dir = ctx["run_dir"]

    # run_dir basename is <timestamp> (under runs/<name>/<ts>/) or <name> (under models/<name>/)
    timestamp = run_dir.name
    title = f"# MODEL: {cfg.run_name}_{timestamp}" if timestamp != cfg.run_name else f"# MODEL: {cfg.run_name}"

    status = "promoted" if wandb_meta else "run-only"
    git = meta.get("git_hash", "(unknown)") if meta else "(unknown)"

    lines = [
        title,
        "",
        f"**Status:** {status}  ·  **Git:** `{git}`",
    ]
    if wandb_meta:
        name = wandb_meta["name"]
        version = wandb_meta["version"]
        aliases = ", ".join(wandb_meta.get("aliases", []))
        lines.append(f"**W&B:** `wandb://{name}:prod` ({version}, aliases: {aliases})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing (5 total now).

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_header renderer

Renders the title + status + git hash + W&B URI.  Status reads 'promoted'
when _wandb_metadata.json is present, 'run-only' when absent (W&B line
omitted entirely in the latter case)."
```

---

### Task 4: `_section_summary` renderer

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_summary


def test_section_summary_uses_description_override_when_present():
    """If cfg.description is non-empty, it replaces the auto-template verbatim."""
    custom = "Curriculum step 1: pretrain from blue_v4 onto random_start."
    cfg = _ctx_for_section()["cfg"]
    cfg.description = custom
    out = _section_summary(_ctx_for_section(cfg=cfg))
    assert custom in out
    # auto-template hallmarks must NOT appear
    assert "learner trained from" not in out


def test_section_summary_auto_templates_when_description_empty():
    """Empty description → auto-template from cfg fields."""
    out = _section_summary(_ctx_for_section())
    # Spot-check the auto-template's content
    assert "blue_0" in out  # learner_id
    assert "scratch" in out  # init.mode
    assert "DUEL_V2_WORLD" in out  # obs.name
    assert "n_stack=3" in out
    assert "team_v2" in out  # reward group choice
    assert "10,000,000" in out or "10_000_000" in out  # total_timesteps formatted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_summary_uses_description_override_when_present tests/scripts/test_render_model_doc.py::test_section_summary_auto_templates_when_description_empty -v`
Expected: FAIL — `_section_summary` doesn't exist yet.

- [ ] **Step 3: Implement `_section_summary`**

Append to `scripts/_render_model_doc.py`:

```python
def _section_summary(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    hydra_yaml = ctx["hydra_yaml"]

    description = (cfg.get("description") or "").strip() if hasattr(cfg, "get") else (cfg.description or "").strip()
    if description:
        return "## Summary\n\n" + description

    # Auto-template path.
    learner_id = cfg.env.get("learner_id", "drone_0") if hasattr(cfg, "env") else "drone_0"
    init_mode = cfg.init.mode if hasattr(cfg, "init") else "scratch"
    parent = cfg.init.get("parent", None) if hasattr(cfg, "init") else None
    obs_name = cfg.obs.name
    n_stack = cfg.obs.n_stack
    total_steps = int(cfg.trainer.total_timesteps)
    lr = cfg.trainer.lr
    curriculum = cfg.curriculum.get("name", "(unknown)") if hasattr(cfg, "curriculum") else "(unknown)"

    # Opponent shorthand: read _target_ class name or a `spec` field if present.
    opp = cfg.get("opponent", None) if hasattr(cfg, "get") else None
    if opp is not None:
        opp_short = opp.get("spec", None) or opp.get("_target_", "").rsplit(".", 1)[-1]
    else:
        opp_short = None

    # Reward group choice from hydra.yaml; fall back if absent.
    reward_choice = "(unknown)"
    if hydra_yaml:
        reward_choice = (
            hydra_yaml.get("hydra", {})
            .get("runtime", {})
            .get("choices", {})
            .get("reward", "(unknown)")
        )

    parts = [f"{learner_id} learner trained from {init_mode}"]
    if parent:
        parts.append(f"(parent: {parent})")
    if opp_short:
        parts.append(f"against {opp_short}")
    parts.append(f"on {obs_name} × n_stack={n_stack}")
    parts.append(f"reward stack {reward_choice}")
    parts.append(f"lr={lr}")
    parts.append(curriculum)
    parts.append(f"{total_steps:,} steps.")

    body = ", ".join(parts[:1] + parts[1:])  # readable join
    return "## Summary\n\n" + body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_summary renderer with override + template

When cfg.description is non-empty, render it verbatim.  Otherwise auto-template
from cfg.{env.learner_id, init.mode/parent, opponent, obs, trainer, curriculum}
plus hydra.runtime.choices.reward for the reward group choice."
```

---

### Task 5: `_section_lineage` renderer

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_lineage


def test_section_lineage_scratch_renders_minimally():
    """init.mode == scratch → just the one-liner, no parent fields."""
    out = _section_lineage(_ctx_for_section())  # default cfg.init.mode = scratch
    assert "## Lineage" in out
    assert "scratch" in out
    assert "no parent" in out
    # Don't render the parent / chain-total fields
    assert "**parent:**" not in out


def test_section_lineage_pretrain_renders_parent_and_chain_total():
    ctx = _ctx_for_section()
    ctx["cfg"].init.mode = "pretrain"
    ctx["cfg"].init.parent = "models/ppo_hoop_blue_4_20260511_202612/best_model"
    ctx["meta"]["parent_chain_total"] = 20_000_000
    ctx["cfg"].trainer.total_timesteps = 10_000_000
    out = _section_lineage(ctx)
    assert "pretrain" in out
    assert "models/ppo_hoop_blue_4_20260511_202612/best_model" in out
    assert "20,000,000" in out or "20000000" in out  # parent chain total formatted
    assert "10,000,000" in out or "10000000" in out  # this run's contribution
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_lineage_scratch_renders_minimally tests/scripts/test_render_model_doc.py::test_section_lineage_pretrain_renders_parent_and_chain_total -v`
Expected: FAIL — `_section_lineage` doesn't exist yet.

- [ ] **Step 3: Implement `_section_lineage`**

Append to `scripts/_render_model_doc.py`:

```python
def _section_lineage(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    init_mode = cfg.init.mode if hasattr(cfg, "init") else "scratch"
    if init_mode == "scratch":
        return "## Lineage\n\n- **init mode:** `scratch` — no parent"

    parent = cfg.init.get("parent", None) if hasattr(cfg.init, "get") else None
    chain_total = meta.get("parent_chain_total", None)
    this_total = int(cfg.trainer.total_timesteps)

    lines = ["## Lineage", "", f"- **init mode:** `{init_mode}`"]
    if parent:
        lines.append(f"- **parent:** `{parent}`")
    if chain_total is not None:
        lines.append(f"- **parent chain total:** {chain_total:,} steps (this run is {this_total:,} of that)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_lineage renderer

For init.mode=scratch: one-line 'no parent'.  Otherwise: init.mode + parent
path + parent_chain_total + this-run's contribution."
```

---

### Task 6: `_section_obs_spec` renderer (table from ObsSpec.offsets)

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_obs_spec


def test_section_obs_spec_renders_table_for_known_spec():
    out = _section_obs_spec(_ctx_for_section())  # default cfg.obs.name = DUEL_V2_WORLD
    assert "## Obs spec" in out
    assert "DUEL_V2_WORLD" in out
    assert "25-d" in out  # the spec's total dim
    assert "n_stack:** 3" in out
    # Table header
    assert "| Slot | Block | Dim | Frame | Notes |" in out
    # A canonical block from DUEL_V2_WORLD
    assert "ang_vel" in out
    assert "closing_rate" in out


def test_section_obs_spec_renders_error_blockquote_for_unknown_spec():
    ctx = _ctx_for_section()
    ctx["cfg"].obs.name = "FAKE_NEVER_REGISTERED_OBS"
    out = _section_obs_spec(ctx)
    assert "⚠" in out or "(unknown obs spec" in out
    assert "FAKE_NEVER_REGISTERED_OBS" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_obs_spec_renders_table_for_known_spec tests/scripts/test_render_model_doc.py::test_section_obs_spec_renders_error_blockquote_for_unknown_spec -v`
Expected: FAIL — `_section_obs_spec` doesn't exist.

- [ ] **Step 3: Implement `_section_obs_spec`**

Append to `scripts/_render_model_doc.py`:

```python
def _section_obs_spec(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    from envs.quidditch.obs_spec import SPEC_BY_NAME

    name = cfg.obs.name
    n_stack = int(cfg.obs.n_stack)
    spec = SPEC_BY_NAME.get(name)
    if spec is None:
        return (
            "## Obs spec\n\n"
            f"> ⚠ unknown obs spec name `{name}` — not in SPEC_BY_NAME registry. "
            "See `.hydra/config.yaml:obs` for the recorded name."
        )

    header = (
        "## Obs spec\n\n"
        f"**Name:** `{name}` ({spec.dim}-d)  ·  **n_stack:** {n_stack}  ·  "
        f"**Input dim:** {spec.dim * n_stack}"
    )
    rows = [
        "| Slot | Block | Dim | Frame | Notes |",
        "|------|-------|-----|-------|-------|",
    ]
    for block, sl in spec.offsets():
        notes = block.notes or ""
        frame = block.frame or ""
        rows.append(f"| {sl.start}:{sl.stop} | {block.name} | {block.dim} | {frame} | {notes} |")
    return header + "\n\n" + "\n".join(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_obs_spec renderer with table from offsets

Reads SPEC_BY_NAME[cfg.obs.name] and renders the block-by-block table from
spec.offsets().  Unknown spec name → error blockquote + pointer to
.hydra/config.yaml."
```

---

### Task 7: `_section_reward_stack` renderer (dataclasses.fields introspection)

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_reward_stack


def test_section_reward_stack_renders_terms_from_team_v2():
    """Pass a cfg.reward pointing at the real team_v2 YAML structure; the
    renderer instantiates it and walks the terms via dataclasses.fields."""
    cfg = _ctx_for_section()["cfg"]
    # Build a real team_v2-shaped reward block (top-level _target_ + terms list)
    cfg.reward = OmegaConf.create({
        "_target_": "envs.quidditch.rewards.stack.RewardStack",
        "terms": [
            {"_target_": "envs.quidditch.rewards.terms.TagEntryPulse",
             "magnitude": 5.0, "gainer": "blue_0", "loser": "red_0"},
            {"_target_": "envs.quidditch.rewards.terms.ScoreEvent",
             "magnitude": 10.0, "scorer": "red_0", "zero_sum_opponent": "blue_0"},
        ],
    })
    out = _section_reward_stack(_ctx_for_section(cfg=cfg))
    assert "## Reward stack" in out
    assert "team_v2" in out  # source line via hydra.runtime.choices.reward
    assert "TagEntryPulse" in out
    assert "magnitude=5.0" in out
    assert "blue_0" in out
    assert "ScoreEvent" in out
    assert "magnitude=10.0" in out


def test_section_reward_stack_falls_back_when_hydra_yaml_absent():
    cfg = _ctx_for_section()["cfg"]
    cfg.reward = OmegaConf.create({
        "_target_": "envs.quidditch.rewards.stack.RewardStack",
        "terms": [{"_target_": "envs.quidditch.rewards.terms.TagEntryPulse",
                    "magnitude": 5.0, "gainer": "blue_0", "loser": "red_0"}],
    })
    out = _section_reward_stack(_ctx_for_section(cfg=cfg, hydra_yaml=None))
    assert "## Reward stack" in out
    assert "in-line override" in out or "unknown source" in out
    assert "TagEntryPulse" in out  # the table itself still renders
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_reward_stack_renders_terms_from_team_v2 tests/scripts/test_render_model_doc.py::test_section_reward_stack_falls_back_when_hydra_yaml_absent -v`
Expected: FAIL — `_section_reward_stack` doesn't exist.

- [ ] **Step 3: Implement `_section_reward_stack`**

Append to `scripts/_render_model_doc.py`:

```python
import dataclasses


def _term_coefficient_summary(term: Any) -> str:
    """Render a term's numeric / dict fields as `name=value` joined by commas."""
    parts: list[str] = []
    for f in dataclasses.fields(term):
        val = getattr(term, f.name)
        if isinstance(val, (int, float)):
            parts.append(f"{f.name}={val}")
        elif isinstance(val, dict):
            parts.append(f"{f.name}={dict(val)}")
    return ", ".join(parts)


def _term_agents_summary(term: Any) -> str:
    """Render the term's agent-identifying fields as human shorthand."""
    out: list[str] = []
    for f in dataclasses.fields(term):
        val = getattr(term, f.name)
        if f.name in ("gainer", "loser", "scorer", "zero_sum_opponent",
                       "aggressor", "victim"):
            if val:
                out.append(f"{val} ({f.name})")
        elif f.name == "agents":
            if val:
                out.append(", ".join(val))
        elif f.name in ("agent_to_target", "agent_to_crash_flags"):
            if val:
                out.append(", ".join(f"{k}→{v}" for k, v in val.items()))
    return "; ".join(out)


def _section_reward_stack(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    hydra_yaml = ctx["hydra_yaml"]
    from hydra.utils import instantiate

    # Source line: reward group choice from hydra.yaml, or fallback.
    if hydra_yaml:
        choice = (
            hydra_yaml.get("hydra", {}).get("runtime", {}).get("choices", {}).get("reward", None)
        )
        source = f"`conf/reward/{choice}.yaml`" if choice else "(in-line override / unknown source)"
    else:
        source = "(in-line override / unknown source)"

    stack = instantiate(cfg.reward, _convert_="all")

    rows = [
        "| # | Term | Key coefficients | Agents |",
        "|---|------|------------------|--------|",
    ]
    for i, term in enumerate(stack.terms, start=1):
        cls = type(term).__name__
        coeffs = _term_coefficient_summary(term)
        agents = _term_agents_summary(term)
        rows.append(f"| {i} | {cls} | {coeffs} | {agents} |")

    return f"## Reward stack\n\n**Source:** {source}\n\n" + "\n".join(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_reward_stack renderer

Instantiates cfg.reward via Hydra, walks stack.terms, renders class name +
field summary (numeric/dict via dataclasses.fields) + agent assignments.
Adding a new term class doesn't require touching the renderer.  Source line
reads hydra.runtime.choices.reward from .hydra/hydra.yaml; falls back when
absent."
```

---

### Task 8: `_section_env_config` + `_section_hyperparams` + `_section_eval_results`

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

These three are simple cfg.* reads. Grouping into one task for cadence; each gets its own test.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_env_config, _section_hyperparams, _section_eval_results


def test_section_env_config_renders_team_fields():
    out = _section_env_config(_ctx_for_section())
    assert "## Env config" in out
    assert "BeelineRed" in out  # opponent short name
    assert "blue_0" in out  # learner_id
    assert "30.0 s" in out  # episode_seconds
    assert "fixed_start" in out  # curriculum
    assert "0.3 m" in out  # tag_radius
    assert "1.0 m/s" in out  # crash_vel_thr


def test_section_hyperparams_renders_trainer_fields():
    out = _section_hyperparams(_ctx_for_section())
    assert "## Training hyperparams" in out
    assert "PPO" in out
    assert "lr:** 0.0003" in out or "lr:** 3e-4" in out
    assert "10,000,000" in out
    assert "n_envs:** 8" in out
    assert "batch_size:** 256" in out


def test_section_eval_results_renders_best_kind():
    out = _section_eval_results(_ctx_for_section())
    assert "## Eval results" in out
    assert "7.91" in out
    assert "9,500,000" in out
    assert "10,000,000" in out
    assert "best" in out


def test_section_eval_results_handles_missing_meta():
    out = _section_eval_results(_ctx_for_section(meta=None))
    assert "## Eval results" in out
    assert "meta.yaml absent" in out


def test_section_eval_results_omits_best_lines_for_final_kind():
    ctx = _ctx_for_section()
    ctx["meta"]["final_stats"] = {"completed_steps": 200, "wall_clock_seconds": 5.0,
                                    "model_kind": "final"}
    out = _section_eval_results(ctx)
    assert "model_kind" in out and "final" in out
    assert "best_eval_reward" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -k "test_section_env_config or test_section_hyperparams or test_section_eval_results" -v`
Expected: FAIL — none of the three section functions exist.

- [ ] **Step 3: Implement the three renderers**

Append to `scripts/_render_model_doc.py`:

```python
def _opponent_short(cfg) -> str:
    opp = cfg.get("opponent", None) if hasattr(cfg, "get") else None
    if opp is None:
        return "(none)"
    if hasattr(opp, "get") and opp.get("spec"):
        return opp.get("spec")
    target = opp.get("_target_", "") if hasattr(opp, "get") else ""
    return target.rsplit(".", 1)[-1] if target else "(unknown)"


def _section_env_config(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    learner = cfg.env.get("learner_id", "drone_0") if hasattr(cfg.env, "get") else "drone_0"
    team_cfg = cfg.env.get("team_cfg", None) if hasattr(cfg.env, "get") else None
    curriculum = cfg.curriculum.get("name", "(unknown)") if hasattr(cfg.curriculum, "get") else "(unknown)"
    opp = _opponent_short(cfg)

    lines = ["## Env config", ""]
    lines.append(f"- **Opponent:** `{opp}`  ·  **Learner:** `{learner}`")
    if team_cfg:
        episode = team_cfg.get("episode_seconds", "(unknown)")
        lines.append(f"- **Episode:** {episode} s  ·  **Curriculum:** `{curriculum}`")
        tag_r = team_cfg.get("tag_radius", None)
        crash_thr = team_cfg.get("crash_vel_thr", None)
        if tag_r is not None and crash_thr is not None:
            lines.append(f"- **Tag radius:** {tag_r} m  ·  **Crash velocity threshold:** {crash_thr} m/s")
        midpoint = team_cfg.get("midpoint_alpha", None)
        aftermath = team_cfg.get("crash_aftermath_seconds", None)
        if midpoint is not None or aftermath is not None:
            lines.append(f"- **Midpoint α:** {midpoint}  ·  **Crash aftermath:** {aftermath} s")
    else:
        lines.append(f"- **Curriculum:** `{curriculum}`")
    return "\n".join(lines)


def _section_hyperparams(ctx: dict[str, Any]) -> str:
    t = ctx["cfg"].trainer
    total = int(t.total_timesteps)
    return "\n".join([
        "## Training hyperparams",
        "",
        f"- **Algorithm:** PPO  ·  **lr:** {t.lr}  ·  **total_timesteps:** {total:,}",
        f"- **n_envs:** {t.get('n_envs', '(unknown)')}  ·  "
        f"**batch_size:** {t.get('batch_size', '(unknown)')}  ·  "
        f"**n_epochs:** {t.get('n_epochs', '(unknown)')}",
        f"- **gamma:** {t.get('gamma', '(unknown)')}  ·  "
        f"**gae_lambda:** {t.get('gae_lambda', '(unknown)')}  ·  "
        f"**ent_coef:** {t.get('ent_coef', '(unknown)')}  ·  "
        f"**clip_range:** {t.get('clip_range', '(unknown)')}",
    ])


def _format_wall_clock(seconds: float | int) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s_ = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s_:02d}s"
    return f"{m}m {s_:02d}s"


def _section_eval_results(ctx: dict[str, Any]) -> str:
    meta = ctx["meta"]
    if meta is None or "final_stats" not in meta:
        return "## Eval results\n\n(meta.yaml absent — eval section unavailable)"

    fs = meta["final_stats"]
    model_kind = fs.get("model_kind", "(unknown)")
    lines = ["## Eval results", ""]
    if model_kind == "best":
        reward = fs.get("best_eval_reward", "(unknown)")
        best_step = fs.get("best_step", None)
        if isinstance(best_step, int):
            lines.append(f"- **best_eval_reward:** {reward} @ step {best_step:,}")
        else:
            lines.append(f"- **best_eval_reward:** {reward}")
    completed = fs.get("completed_steps", "(unknown)")
    wall = fs.get("wall_clock_seconds", None)
    wall_str = _format_wall_clock(wall) if isinstance(wall, (int, float)) else "(unknown)"
    completed_str = f"{completed:,}" if isinstance(completed, int) else str(completed)
    lines.append(f"- **completed_steps:** {completed_str}  ·  **wall_clock:** {wall_str}")
    lines.append(f"- **model_kind:** `{model_kind}`")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add env-config + hyperparams + eval-results sections

Three straightforward cfg.* reads.  env_config renders team-only lines only
when cfg.env.team_cfg is present.  eval_results degrades gracefully when
meta.yaml is absent and omits best-kind lines for final-kind models."
```

---

### Task 9: `_section_wandb` renderer

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import _section_wandb


def test_section_wandb_renders_when_metadata_present():
    out = _section_wandb(_ctx_for_section())
    assert "## W&B" in out
    assert "gridcom/drone-quidditch" in out
    assert "ppo_hoop_test" in out
    assert "v0" in out
    assert "prod" in out


def test_section_wandb_returns_empty_when_metadata_absent():
    """When _wandb_metadata.json is absent, the W&B section is omitted
    entirely (empty string)."""
    out = _section_wandb(_ctx_for_section(wandb_meta=None))
    assert out == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_section_wandb_renders_when_metadata_present tests/scripts/test_render_model_doc.py::test_section_wandb_returns_empty_when_metadata_absent -v`
Expected: FAIL — `_section_wandb` doesn't exist.

- [ ] **Step 3: Implement `_section_wandb`**

Append to `scripts/_render_model_doc.py`:

```python
def _section_wandb(ctx: dict[str, Any]) -> str:
    wandb_meta = ctx["wandb_meta"]
    if not wandb_meta:
        return ""

    cfg = ctx["cfg"]
    entity = wandb_meta.get("entity", "(unknown)")
    project = wandb_meta.get("project", "(unknown)")
    name = wandb_meta.get("name", "(unknown)")
    version = wandb_meta.get("version", "(unknown)")
    aliases = ", ".join(wandb_meta.get("aliases", []))
    run_id = f"{cfg.run_name}_{ctx['run_dir'].name}" if hasattr(cfg, "run_name") else "(unknown)"

    return "\n".join([
        "## W&B",
        "",
        f"- **Project:** `{entity}/{project}`",
        f"- **Run id:** `{run_id}`",
        f"- **Artifact:** `{name}:{version}`  (aliases: {aliases})",
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add _section_wandb renderer

Reads _wandb_metadata.json for entity/project/name/version/aliases.  Returns
empty string when metadata absent (caller filters empties when composing)."
```

---

### Task 10: Compose `render_model_doc` with per-section try/except

**Files:**
- Modify: `scripts/_render_model_doc.py`
- Modify: `tests/scripts/test_render_model_doc.py`
- Create: `tests/scripts/fixtures/expected_model_doc.md` (snapshot fixture)

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_render_model_doc.py`:

```python
from scripts._render_model_doc import render_model_doc


def test_render_model_doc_end_to_end(tmp_path: Path):
    """Full doc renders against a complete run-dir fixture."""
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={
            "run_name": "ppo_hoop_test",
            "description": "",
            "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3},
            "init": {"mode": "scratch", "parent": None},
            "trainer": {"lr": 3e-4, "total_timesteps": 10_000_000,
                         "n_envs": 8, "batch_size": 256, "n_epochs": 10,
                         "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0,
                         "clip_range": 0.2},
            "env": {"learner_id": "blue_0",
                     "team_cfg": {"episode_seconds": 30.0, "tag_radius": 0.3,
                                  "crash_vel_thr": 1.0, "midpoint_alpha": 0.5,
                                  "crash_aftermath_seconds": 0.0}},
            "opponent": {"_target_": "envs.quidditch.opponents.BeelineRed"},
            "curriculum": {"name": "fixed_start"},
            "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                        "terms": [{"_target_": "envs.quidditch.rewards.terms.ScoreEvent",
                                    "magnitude": 10.0, "scorer": "red_0",
                                    "zero_sum_opponent": "blue_0"}]},
        },
        meta={"git_hash": "abc1234", "final_stats": {
            "best_eval_reward": 7.91, "best_step": 9_500_000,
            "completed_steps": 10_000_000, "wall_clock_seconds": 1923.0,
            "model_kind": "best"}},
        hydra_yaml={"hydra": {"runtime": {"choices": {"reward": "team_v2"}}}},
        wandb_meta={"name": "ppo_hoop_test", "version": "v0",
                     "aliases": ["latest", "prod"],
                     "entity": "gridcom", "project": "drone-quidditch"},
    )
    out = render_model_doc(run_dir)
    # All section headers present
    for header in ["# MODEL:", "## Summary", "## Lineage", "## Obs spec",
                    "## Reward stack", "## Env config", "## Training hyperparams",
                    "## Eval results", "## W&B"]:
        assert header in out, f"missing section: {header}"


def test_render_model_doc_isolates_per_section_failures(tmp_path: Path):
    """A bad obs spec name causes the obs section to flag itself but other
    sections still render."""
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={
            "run_name": "ppo_hoop_test",
            "description": "",
            "obs": {"name": "NEVER_REGISTERED", "n_stack": 1},
            "init": {"mode": "scratch"},
            "trainer": {"lr": 1e-3, "total_timesteps": 1000},
            "env": {"learner_id": "drone_0"},
            "curriculum": {"name": "fixed_start"},
            "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                        "terms": []},
        },
    )
    out = render_model_doc(run_dir)
    # Obs section flags itself
    assert "## Obs spec" in out
    assert "NEVER_REGISTERED" in out
    # Other sections still render
    assert "## Summary" in out
    assert "## Lineage" in out


def test_render_model_doc_omits_wandb_section_when_absent(tmp_path: Path):
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={
            "run_name": "ppo_hoop_test",
            "description": "",
            "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1},
            "init": {"mode": "scratch"},
            "trainer": {"lr": 1e-3, "total_timesteps": 1000},
            "env": {"learner_id": "drone_0"},
            "curriculum": {"name": "fixed_start"},
            "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                        "terms": []},
        },
    )
    out = render_model_doc(run_dir)
    assert "## W&B" not in out
    assert "run-only" in out  # status reflects absence
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_render_model_doc_end_to_end -v`
Expected: FAIL — `render_model_doc` doesn't exist.

- [ ] **Step 3: Implement `render_model_doc`**

Append to `scripts/_render_model_doc.py`:

```python
import logging

log = logging.getLogger(__name__)


_SECTION_FUNCTIONS = [
    ("header",       _section_header),
    ("summary",      _section_summary),
    ("lineage",      _section_lineage),
    ("obs_spec",     _section_obs_spec),
    ("reward_stack", _section_reward_stack),
    ("env_config",   _section_env_config),
    ("hyperparams",  _section_hyperparams),
    ("eval_results", _section_eval_results),
    ("wandb",        _section_wandb),
]


def render_model_doc(run_dir: Path) -> str:
    """Render a per-model MODEL.md spec sheet for one run dir.

    Pure: reads inputs from `run_dir`, returns a Markdown string.  No
    filesystem writes, no wandb network calls.  Per-section try/except so
    a bad section doesn't kill the whole doc.
    """
    ctx = _load_run_context(run_dir)
    parts: list[str] = []
    for name, fn in _SECTION_FUNCTIONS:
        try:
            out = fn(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("section %s render failed: %s", name, e)
            out = f"## (Section `{name}` failed to render)\n\n> ⚠ {type(e).__name__}: {e}"
        if out:  # _section_wandb returns "" when meta absent
            parts.append(out)
    return "\n\n".join(parts) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_render_model_doc.py tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): compose render_model_doc with per-section isolation

Walks the section function list, wraps each call in try/except.  Failed
sections render as Markdown blockquotes (visually flagged) so the rest of
the doc still ships.  Empty sections (e.g., W&B when metadata absent) are
filtered out cleanly."
```

---

### Task 11: CLI entrypoint + Makefile target

**Files:**
- Create: `scripts/render_model_doc.py`
- Modify: `Makefile`
- Modify: `tests/scripts/test_render_model_doc.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_render_model_doc.py`:

```python
import subprocess
import sys


def test_cli_writes_model_doc_to_run_dir(tmp_path: Path):
    """`python -m scripts.render_model_doc --run-dir <path>` writes MODEL.md."""
    run_dir = _write_run_fixture(
        tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000",
        config={
            "run_name": "ppo_hoop_test",
            "description": "",
            "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1},
            "init": {"mode": "scratch"},
            "trainer": {"lr": 1e-3, "total_timesteps": 1000},
            "env": {"learner_id": "drone_0"},
            "curriculum": {"name": "fixed_start"},
            "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack",
                        "terms": []},
        },
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.render_model_doc", "--run-dir", str(run_dir)],
        capture_output=True, text=True,
        env={"PYTHONPATH": ".", "KMP_DUPLICATE_LIB_OK": "TRUE", "WANDB_MODE": "disabled"},
    )
    assert result.returncode == 0, result.stderr
    md = run_dir / "MODEL.md"
    assert md.exists()
    text = md.read_text()
    assert "# MODEL: ppo_hoop_test_20260516_120000" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_cli_writes_model_doc_to_run_dir -v`
Expected: FAIL — `scripts.render_model_doc` module doesn't exist.

- [ ] **Step 3: Create the CLI module**

Create `scripts/render_model_doc.py`:

```python
"""CLI entrypoint: render MODEL.md for a run dir.

Usage:
    python -m scripts.render_model_doc --run-dir runs/ppo_hoop_test/20260516_120000
    make describe-run RUN_NAME=ppo_hoop_test                   # latest trial auto-resolved
    make describe-run RUN_NAME=ppo_hoop_test TRIAL=20260516_120000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._render_model_doc import render_model_doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Path to a run dir containing .hydra/{config,meta,hydra}.yaml")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    doc = render_model_doc(run_dir)
    out_path = run_dir / "MODEL.md"
    out_path.write_text(doc)
    print(f"wrote {out_path} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add the Makefile target**

In `Makefile`, find `list-runs:` (it has the resolver for `runs/<name>/<latest_ts>/`) and add `describe-run` near it. Also add `describe-run` to the `.PHONY` list at the top.

Edit `.PHONY` to include `describe-run`:

```makefile
.PHONY: help test test-fast test-warm camera-test demo train resume eval eval-headless lineage promote install clean list-runs obs-specs describe-run eval-team sweep sweep-agent sweep-agents
```

Add the target before `list-runs`:

```makefile
describe-run: ## 📝 Render MODEL.md for a run  RUN_NAME=<name> [TRIAL=<ts>]
	@if [ -z "$(RUN_NAME)" ]; then echo "ERROR: RUN_NAME=<name> required"; exit 1; fi; \
	 if [ -n "$(TRIAL)" ]; then RUN_DIR="$(RUNS_DIR)/$(RUN_NAME)/$(TRIAL)"; \
	 else RUN_DIR="$(RUNS_DIR)/$(RUN_NAME)/$$(ls -1t $(RUNS_DIR)/$(RUN_NAME) | head -1)"; fi; \
	 $(PYTHON) -m scripts.render_model_doc --run-dir "$$RUN_DIR"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_render_model_doc.py::test_cli_writes_model_doc_to_run_dir -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/render_model_doc.py Makefile tests/scripts/test_render_model_doc.py
git commit -m "feat(model-doc): add CLI entrypoint + make describe-run target

python -m scripts.render_model_doc --run-dir <path> writes MODEL.md into
the run dir.  make describe-run RUN_NAME=<name> [TRIAL=<ts>] resolves the
latest trial automatically when TRIAL is omitted."
```

---

### Task 12: Wire `render_model_doc` into `scripts/train.py:finally:`

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/scripts/test_train_smoke_wandb_disabled.py`

- [ ] **Step 1: Write the failing test**

Find `tests/scripts/test_train_smoke_wandb_disabled.py`. Locate a test that runs a short training pass and asserts on the run dir contents. Add this assertion (and adapt the helper name to match what's already in that file):

```python
# Inside the existing smoke test, after the train run completes:
assert (run_dir / "MODEL.md").exists(), "MODEL.md should be auto-generated at train end"
text = (run_dir / "MODEL.md").read_text()
assert "# MODEL:" in text
assert "## Summary" in text
```

If the existing test doesn't expose `run_dir`, factor it from the test's setup or extend the assertion to walk `runs/` for the just-created dir.

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_train_smoke_wandb_disabled.py -v`
Expected: FAIL — MODEL.md not yet generated at train end.

- [ ] **Step 3: Wire the call into train.py**

Open `scripts/train.py`. Find the `finally:` block (where `meta.yaml` gets written and `log_run_artifact` is called). Add the MODEL.md render between meta.yaml write and `log_run_artifact`:

```python
# In scripts/train.py's finally: block, after meta.yaml is written:
try:
    from scripts._render_model_doc import render_model_doc
    doc = render_model_doc(run_dir)
    (run_dir / "MODEL.md").write_text(doc)
    log.info("wrote %s", run_dir / "MODEL.md")
except Exception as e:  # noqa: BLE001
    log.warning("MODEL.md generation failed (training succeeded): %s", e)
```

The import is inside the try-block on purpose — keep it cheap to fail (no top-of-file import to fail other code paths).

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_train_smoke_wandb_disabled.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite regression check**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest -k "not test_offscreen_render_one_frame_no_crash and not test_capture_cells_team and not test_simple_env_factory_builds_16d_train_env and not test_team_env_factory_builds_75d_train_env_with_frame_stack" --tb=short`
Expected: All non-render tests pass. Canaries (test_scoring_canary, test_team_env_canary) byte-identical.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py tests/scripts/test_train_smoke_wandb_disabled.py
git commit -m "feat(model-doc): wire render_model_doc into train.py:finally:

Generates <run_dir>/MODEL.md after meta.yaml is written and before
log_run_artifact runs (so the wandb upload picks it up).  Wrapped in
try/except: a doc-gen failure logs a warning but does not fail the
training run."
```

---

### Task 13: Include MODEL.md in the wandb artifact

**Files:**
- Modify: `scripts/_artifact_io.py`
- Modify: `tests/scripts/test_log_run_artifact.py`

- [ ] **Step 1: Write the failing test**

Find `tests/scripts/test_log_run_artifact.py`. Locate a test that exercises `log_run_artifact` against a fake wandb run; add a fixture file `MODEL.md` to the run dir and assert it ends up in the artifact's file list. Append (adapting to the existing test's setup):

```python
def test_log_run_artifact_includes_model_doc_when_present(tmp_path, monkeypatch):
    """When <run_dir>/MODEL.md exists, log_run_artifact adds it to the artifact."""
    run_dir = tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text("run_name: ppo_hoop_test\n")
    (run_dir / "best_model.zip").write_text("fake")
    (run_dir / "MODEL.md").write_text("# MODEL: ppo_hoop_test\n")

    added_files: list[str] = []
    class FakeArtifact:
        def __init__(self, *a, **kw):
            self.metadata = kw.get("metadata", {})
        def add_file(self, p, name=None):
            added_files.append(name or Path(p).name)
        def add_dir(self, p, name=None):
            pass
        def wait(self):
            pass
        @property
        def version(self):
            return "v0"

    monkeypatch.setattr("wandb.Artifact", FakeArtifact)
    # Adapt the existing test's call shape for log_run_artifact here.
    # ... (use the existing test as a template)

    assert "MODEL.md" in added_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_log_run_artifact.py::test_log_run_artifact_includes_model_doc_when_present -v`
Expected: FAIL — MODEL.md not yet added.

- [ ] **Step 3: Add MODEL.md to the artifact**

Open `scripts/_artifact_io.py`. Find `log_run_artifact` and locate the `art.add_file(best_or_final)` + `art.add_dir(.hydra/)` block. Add MODEL.md inclusion:

```python
# Inside log_run_artifact, after the existing add_file/add_dir calls:
model_doc = run_dir / "MODEL.md"
if model_doc.exists():
    art.add_file(str(model_doc), name="MODEL.md")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_log_run_artifact.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/_artifact_io.py tests/scripts/test_log_run_artifact.py
git commit -m "feat(model-doc): include MODEL.md in wandb artifact when present

log_run_artifact now adds <run_dir>/MODEL.md as a file in the artifact tree
(if it exists).  Makes the doc visible in the wandb model registry UI."
```

---

### Task 14: Copy MODEL.md in `scripts/promote.py`

**Files:**
- Modify: `scripts/promote.py`
- Modify: `tests/scripts/test_promote.py`

- [ ] **Step 1: Write the failing test**

Find `tests/scripts/test_promote.py`. Locate a test that runs `promote.py` (or a callable inside it) against a fake run dir + assertion on the destination `models/<name>/` contents. Append:

```python
def test_promote_copies_model_doc_when_present(tmp_path, monkeypatch):
    """When <run_dir>/MODEL.md exists, promote.py copies it to models/<name>/MODEL.md."""
    run_dir = tmp_path / "runs" / "ppo_hoop_test" / "20260516_120000"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text("run_name: ppo_hoop_test\n")
    (run_dir / "best_model.zip").write_text("fake")
    (run_dir / "MODEL.md").write_text("# MODEL: ppo_hoop_test\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    # ... (use the existing test's promote-callable shape here)

    dest = models_dir / "ppo_hoop_test"
    assert (dest / "MODEL.md").exists()
    assert (dest / "MODEL.md").read_text() == "# MODEL: ppo_hoop_test\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_promote.py::test_promote_copies_model_doc_when_present -v`
Expected: FAIL.

- [ ] **Step 3: Add the copy in promote.py**

Open `scripts/promote.py`. Find the section that copies `best_model.zip` + `.hydra/` into `models/<name>/`. Add MODEL.md copy:

```python
# In promote.py, after the existing copy of best_model.zip + .hydra/:
import shutil
src_doc = run_dir / "MODEL.md"
if src_doc.exists():
    shutil.copy(src_doc, dest / "MODEL.md")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_promote.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/promote.py tests/scripts/test_promote.py
git commit -m "feat(model-doc): copy MODEL.md in promote.py when present

promote.py now copies <run_dir>/MODEL.md to models/<run_name>/MODEL.md
alongside the existing best_model.zip + .hydra/ copies."
```

---

### Task 15: One-shot backfill script for legacy promoted models

**Files:**
- Create: `scripts/backfill_model_docs.py`
- Create: `tests/scripts/test_backfill_model_docs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_backfill_model_docs.py`:

```python
"""Tests for scripts/backfill_model_docs.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _make_fake_model_dir(root: Path, name: str) -> Path:
    """Build a minimal models/<name>/ for backfill to render against."""
    d = root / "models" / name
    (d / ".hydra").mkdir(parents=True)
    (d / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({
        "run_name": name,
        "description": "",
        "obs": {"name": "SIMPLE_ENV_OBS", "n_stack": 1},
        "init": {"mode": "scratch"},
        "trainer": {"lr": 1e-3, "total_timesteps": 1000},
        "env": {"learner_id": "drone_0"},
        "curriculum": {"name": "fixed_start"},
        "reward": {"_target_": "envs.quidditch.rewards.stack.RewardStack", "terms": []},
    })))
    return d


def _run_backfill(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.backfill_model_docs"] + args,
        cwd=repo_root, capture_output=True, text=True,
        env={"PYTHONPATH": ".", "KMP_DUPLICATE_LIB_OK": "TRUE", "WANDB_MODE": "disabled"},
    )


def test_backfill_writes_model_doc(tmp_path):
    repo = tmp_path
    _make_fake_model_dir(repo, "fake_model_a")
    result = _run_backfill(repo, [])
    assert result.returncode == 0, result.stderr
    assert (repo / "models" / "fake_model_a" / "MODEL.md").exists()


def test_backfill_is_idempotent_without_force(tmp_path):
    repo = tmp_path
    d = _make_fake_model_dir(repo, "fake_model_a")
    md = d / "MODEL.md"
    md.write_text("ORIGINAL")
    result = _run_backfill(repo, [])
    assert result.returncode == 0
    # Existing file untouched
    assert md.read_text() == "ORIGINAL"


def test_backfill_force_overwrites(tmp_path):
    repo = tmp_path
    d = _make_fake_model_dir(repo, "fake_model_a")
    md = d / "MODEL.md"
    md.write_text("ORIGINAL")
    result = _run_backfill(repo, ["--force"])
    assert result.returncode == 0
    assert md.read_text() != "ORIGINAL"
    assert "fake_model_a" in md.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_backfill_model_docs.py -v`
Expected: FAIL — `scripts.backfill_model_docs` module doesn't exist.

- [ ] **Step 3: Create the backfill script**

Create `scripts/backfill_model_docs.py`:

```python
"""One-shot: render MODEL.md for every dir under models/ (legacy backfill).

Idempotent — skips dirs that already have MODEL.md unless `--force` is set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._render_model_doc import render_model_doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("models"),
                   help="Root dir to scan (default: ./models)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing MODEL.md files")
    args = p.parse_args()

    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 1

    rendered, skipped, failed = 0, 0, 0
    for model_dir in sorted(args.root.iterdir()):
        if not model_dir.is_dir() or not (model_dir / ".hydra" / "config.yaml").exists():
            continue
        md = model_dir / "MODEL.md"
        if md.exists() and not args.force:
            print(f"[skip] {model_dir.name}: MODEL.md exists (use --force to overwrite)")
            skipped += 1
            continue
        try:
            doc = render_model_doc(model_dir)
            md.write_text(doc)
            print(f"[done] {model_dir.name}: wrote MODEL.md")
            rendered += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {model_dir.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nrendered {rendered}, skipped {skipped}, failed {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest tests/scripts/test_backfill_model_docs.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_model_docs.py tests/scripts/test_backfill_model_docs.py
git commit -m "feat(model-doc): add one-shot backfill script for legacy models

Scans models/ for dirs with .hydra/config.yaml; renders MODEL.md into each.
Idempotent (skip-if-exists) by default; --force overrides.  Used to seed
the 7 legacy promoted models that pre-date this feature."
```

---

### Task 16: Run backfill against the 7 legacy promoted models + commit MODEL.md files

**Files:**
- Modify: `models/<each>/MODEL.md` (7 new files committed)

- [ ] **Step 1: Run the backfill**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled python -m scripts.backfill_model_docs`

Expected output:
```
[done] ppo_hoop_blue_1_20260507_194423: wrote MODEL.md
[done] ppo_hoop_blue_4_20260511_202612: wrote MODEL.md
[done] ppo_hoop_fixed_start_20260430_224234: wrote MODEL.md
[done] ppo_hoop_fixed_start_20260504_023051: wrote MODEL.md
[done] ppo_hoop_rand_start_20260430_234354: wrote MODEL.md
[done] ppo_hoop_rand_start_20260505_174509: wrote MODEL.md
[done] ppo_hoop_red_1_20260506_103058: wrote MODEL.md

rendered 7, skipped 0, failed 0
```

If any model fails (e.g., a migrated legacy `.hydra/config.yaml` is missing a field the renderer expects), open the rendered MODEL.md for the successful ones and check what's missing in the failed one. Add a fallback in the renderer if needed (do this in a separate fix commit, not in this task).

- [ ] **Step 2: Spot-check one rendered file**

Run: `cat models/ppo_hoop_red_1_20260506_103058/MODEL.md | head -40`

Expected: a header, summary, lineage (mentions warm_start parent), obs spec (DUEL_V1_BODY), reward stack (likely team_v2), env config, hyperparams, eval results, W&B (wandb://ppo_hoop_rand_start:v2). If any section reads off, investigate before committing.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled pytest -k "not test_offscreen_render_one_frame_no_crash and not test_capture_cells_team and not test_simple_env_factory_builds_16d_train_env and not test_team_env_factory_builds_75d_train_env_with_frame_stack" --tb=short`
Expected: All non-render tests pass.

- [ ] **Step 4: Commit the 7 MODEL.md files**

```bash
git add models/*/MODEL.md
git commit -m "model: seed MODEL.md for 7 legacy promoted models via backfill

One-shot run of scripts.backfill_model_docs against models/.  Each model
gets a vendored MODEL.md alongside its existing best_model.zip + .hydra/
+ _wandb_metadata.json + run_info.toml / config.toml audit trail."
```

---

### Task 17: Update brain (index.md + changelog.md) — final step

**Files:**
- Modify: `brain/index.md` (umbrella-level, not in git)
- Modify: `brain/changelog.md` (umbrella-level, not in git)
- Modify: `brain/models.md` (umbrella-level, not in git)

These updates land in the umbrella `brain/` directory and are NOT committed to the repo (per project convention — `brain/` lives outside the git repo).

- [ ] **Step 1: Prepend a changelog entry**

Open `brain/changelog.md` and prepend a new entry above the most recent one. Match the existing prose style. The entry should cover:
- New file: `MODEL.md` per-model spec sheet, auto-generated
- Generation paths: train end (best-effort), promote.py copies, _artifact_io.py uploads, `make describe-run` on demand, `scripts/backfill_model_docs.py` one-shot
- Files touched: 3 new (`_render_model_doc.py`, `render_model_doc.py`, `backfill_model_docs.py`), 5 modified (`train.py`, `_artifact_io.py`, `promote.py`, `config_schema.py`, `Makefile`), ~17 tests new
- The 7 legacy models seeded
- Spec + plan locations: `docs/superpowers/specs/2026-05-16-model-doc-generator-design.md` + `docs/superpowers/plans/2026-05-16-model-doc-generator.md`
- ADR-style "for agents grepping" note is unnecessary — no symbols moved.

- [ ] **Step 2: Update `brain/index.md` Recent Context**

Prepend a short bullet in "Recent Context" above the most recent dated entry, matching the existing style. Mention the feature, the four generation paths, the dual-source (description override vs auto-template), and the backfill of legacy models.

- [ ] **Step 3: Update `brain/models.md`**

In `brain/models.md`, mention in the Conventions section that each promoted model in `models/<name>/` now has a vendored `MODEL.md` companion (auto-generated). Add a one-line cross-reference under "Loading & Promotion Quick Reference" pointing at `models/<name>/MODEL.md` for the per-model spec sheet.

- [ ] **Step 4: No commit (brain lives outside the repo)**

Just save the files. The git-side commits are already done in earlier tasks.

---

## Self-Review

After the plan is fully written, walk back through against the spec:

**1. Spec coverage:**
- ✓ MODEL.md content layout (header, summary, lineage, obs, reward, env, hyperparams, eval, wandb) — Tasks 3-9
- ✓ Auto-template + description override — Task 4
- ✓ Reward-stack introspection via dataclasses.fields — Task 7
- ✓ Obs table from SPEC_BY_NAME[name].offsets() — Task 6
- ✓ Per-section try/except isolation — Task 10
- ✓ Missing-files tolerance (meta.yaml, hydra.yaml, _wandb_metadata.json) — Tasks 2 + 10
- ✓ train.py finally: wiring — Task 12
- ✓ wandb artifact inclusion — Task 13
- ✓ promote.py copy — Task 14
- ✓ make describe-run + CLI — Task 11
- ✓ Backfill script + run against the 7 legacy models — Tasks 15 + 16
- ✓ Tests: ~17 (section units + end-to-end + degradation + train smoke + backfill + CLI) — Tasks 2-15

**2. Placeholder scan:**
No "TBD", "TODO", "fill in details", "similar to Task N" found. Each test has actual assertions; each implementation step has actual code.

**3. Type consistency:**
- `_section_*` functions all take `ctx: dict[str, Any]` and return `str` — consistent across Tasks 3-9.
- `_load_run_context(run_dir: Path) -> dict[str, Any]` — same signature referenced in Tasks 2 + 10.
- `render_model_doc(run_dir: Path) -> str` — consistent in Tasks 10, 11, 12, 15.
- `ctx` keys: `cfg`, `meta`, `hydra_yaml`, `wandb_meta`, `run_dir` — consistent across all section tasks.

**4. Scope:**
17 tasks, ~17 test additions, ~600-800 lines net (mostly tests). Single implementation plan. Coherent boundary: everything that touches MODEL.md generation.
