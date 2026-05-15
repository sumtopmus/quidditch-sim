# W&B Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt Weights & Biases as the single dashboard, the canonical artifact registry, and the sweep controller for the project. Remove TensorBoard. Promotion shifts from a `cp -r` to alias-on-artifact + manual vendor commit. `init.parent` accepts `wandb://run:alias` URIs.

**Architecture:** Three subsystems shipped in order — (A) run logger replaces TB, (B) artifact registry + two-tier `models/` cache mirror, (C) wandb sweep controller orchestrates Hydra entrypoint. Tests run with `WANDB_MODE=disabled` so no network touches CI. Spec: `docs/superpowers/specs/2026-05-14-wandb-integration-design.md`.

**Tech Stack:** Python 3.12, Hydra 1.3+, OmegaConf 2.3+, stable-baselines3, `wandb` (new), `wandb.integration.sb3.WandbCallback`. Already-present helpers: `scripts/_train_common.py`, `scripts/_wandb_init.py` (new), `scripts/_artifact_io.py` (new).

**Working directory:** `worktrees/feature/wandb-integration/` (already created off `develop`). All paths below are relative to that worktree root.

**Verification convention:** every test command shows its expected outcome. Per the user's memory: real-world verification that requires the wandb dashboard or a GUI viewer waits for explicit user confirmation before committing. The pytest suite never touches the network.

---

## Phase A — Foundation (conf, schema, dep, test plumbing)

### Task A1: Add `wandb` dependency, drop `tensorboard`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Edit requirements.txt**

Replace the `tensorboard` line and add `wandb`:

```
# Physics / sim
mujoco>=3.0

# RL
stable-baselines3

# Progress bar (used by SB3's progress_bar=True and our ResumeProgressCallback).
# Installed individually instead of `stable-baselines3[extra]` to keep the
# dependency surface small.
tqdm
rich

# Experiment tracking, artifact registry, and sweep controller.
# wandb's SB3 integration (wandb.integration.sb3.WandbCallback) reads from
# SB3's internal logger and forwards metrics; system metrics (GPU/CPU/RAM)
# are captured automatically by wandb.init.
wandb>=0.16

# Video recording (imageio writes mp4s on disk; moviepy is required by
# wandb.Video to encode in-memory and by the on-disk grid writer).
imageio
imageio-ffmpeg
moviepy<2.0

# Multi-agent env API for the team env (Phase 2).
pettingzoo>=1.24

# Test runner.
pytest

# Config composition / structured configs.  Wires conf/ tree through
# scripts/train.py via @hydra.main; ConfigStore validates the data-only groups
# against config_schema.py dataclasses at compose time.
hydra-core>=1.3,<2.0
omegaconf>=2.3,<3.0
```

The full delete is the four lines starting with `# Monitoring / logging` through the blank line. The comment block on `moviepy<2.0` also loses its TB-specific clause.

- [ ] **Step 2: Install in env, verify import**

```bash
conda run -n uav pip install "wandb>=0.16"
conda run -n uav pip uninstall -y tensorboard tensorboard-data-server
conda run -n uav python -c "import wandb; print(wandb.__version__)"
```

Expected: prints a wandb version >= 0.16. `tensorboard` uninstall may print "not installed" (that's fine).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: replace tensorboard with wandb for Part 2 integration"
```

---

### Task A2: `WANDB_MODE=disabled` in tests

**Files:**
- Modify: `tests/conftest.py:13-14` (insert one line after the existing `KMP_DUPLICATE_LIB_OK` setdefault)

- [ ] **Step 1: Write failing test for env var**

Create `tests/unit/test_conftest_wandb_disabled.py`:

```python
"""Confirm conftest disables wandb so the suite never touches the network."""
import os


def test_wandb_mode_is_disabled() -> None:
    assert os.environ.get("WANDB_MODE") == "disabled"
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n uav python -m pytest tests/unit/test_conftest_wandb_disabled.py -v
```

Expected: FAIL — `WANDB_MODE` is not set (or set to something else).

- [ ] **Step 3: Edit `tests/conftest.py` to set the env var**

Add the new line right after the existing `KMP_DUPLICATE_LIB_OK` block (the existing line is `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`). Use `setdefault` so a user can override via shell env if they want to debug a wandb call locally:

```python
# Disable wandb network calls in tests.  wandb.init returns a no-op stub in
# disabled mode; no files written, no auth required.  setdefault so a
# debugger can flip it via `WANDB_MODE=offline python -m pytest ...`.
os.environ.setdefault("WANDB_MODE", "disabled")
```

- [ ] **Step 4: Run test, verify it passes**

```bash
conda run -n uav python -m pytest tests/unit/test_conftest_wandb_disabled.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite to confirm no regression**

```bash
conda run -n uav python -m pytest tests/unit -x
```

Expected: all existing unit tests still pass.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/unit/test_conftest_wandb_disabled.py
git commit -m "test: disable wandb in conftest so suite stays offline"
```

---

### Task A3: Add `WandbConfig` schema + `conf/wandb/default.yaml`

**Files:**
- Modify: `config_schema.py`
- Create: `conf/wandb/default.yaml`
- Modify: `conf/config.yaml`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_wandb_config_schema.py`:

```python
"""WandbConfig is registered and conf/wandb/default.yaml composes cleanly."""
from tests.conftest import hydra_compose


def test_wandb_default_composes() -> None:
    with hydra_compose(experiment="canary_team") as cfg:
        assert cfg.wandb.project == "drone-quidditch"
        assert cfg.wandb.entity_override is None
        assert list(cfg.wandb.tags_extra) == []
        assert cfg.wandb.notes == ""
        assert cfg.wandb.log_gradients is False


def test_wandb_config_in_defaults_list() -> None:
    """conf/config.yaml's defaults list must include `wandb: default`."""
    with hydra_compose(experiment="canary_single") as cfg:
        # If `wandb: default` weren't in defaults, cfg.wandb wouldn't exist.
        assert "wandb" in cfg
        assert cfg.wandb.project == "drone-quidditch"
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_wandb_config_schema.py -v
```

Expected: FAIL — `cfg.wandb` doesn't exist (the group isn't in the defaults list yet).

- [ ] **Step 3: Add `WandbConfig` to `config_schema.py`**

Insert this dataclass before the `register_configs` function (after `ObsConfig`, before `def register_configs`):

```python
@dataclass
class WandbConfig:
    """W&B integration knobs.

    Read at runtime by scripts/_wandb_init.py.  Defaults target this
    project's wandb workspace; an experiment YAML can override tags_extra
    for ad-hoc filtering.
    """
    project: str = "drone-quidditch"
    entity_override: str | None = None         # null → WANDB_ENTITY env / default
    tags_extra: list[str] = field(default_factory=list)
    notes: str = ""
    log_gradients: bool = False
```

Add the registration line inside `register_configs()`, immediately after the existing `cs.store(group="obs", ...)` line:

```python
    cs.store(group="wandb",      name="schema", node=WandbConfig)
```

- [ ] **Step 4: Create `conf/wandb/default.yaml`**

```yaml
# W&B integration defaults.  Override per experiment via `defaults:` group
# choice (none exist yet — this is the only entry) or by setting
# `wandb.tags_extra=[...]` / `wandb.notes=...` in an experiment YAML.

project: drone-quidditch
entity_override: null
tags_extra: []
notes: ""
log_gradients: false
```

- [ ] **Step 5: Edit `conf/config.yaml` defaults list**

Add `- wandb: default` after `- curriculum: random_start`, before `- _self_`:

```yaml
defaults:
  - trainer: ppo
  - env: team
  - obs: augmented
  - reward: team_v2
  - opponent: beeline_red
  - eval: default
  - init: scratch
  - curriculum: random_start
  - wandb: default
  - _self_                          # values below override the group defaults
  - optional local: default         # gitignored; missing-file is silent
```

- [ ] **Step 6: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_wandb_config_schema.py -v
```

Expected: PASS.

- [ ] **Step 7: Run full unit suite to confirm no regression**

```bash
conda run -n uav python -m pytest tests/unit -x
```

Expected: all existing unit tests still pass.

- [ ] **Step 8: Commit**

```bash
git add config_schema.py conf/wandb/default.yaml conf/config.yaml tests/unit/test_wandb_config_schema.py
git commit -m "feat(conf): add WandbConfig schema + conf/wandb/default.yaml group"
```

---

### Task A4: Schema-level ban on `:latest` in committed `init.parent`

**Files:**
- Modify: `config_schema.py` — `InitConfig.__post_init__`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_init_latest_ban.py`:

```python
"""Schema rejects `wandb://...:latest` in init.parent — committed configs
must pin a stable alias (`:prod` or `:<run_name>`) so lineage doesn't shift
silently under a moving alias."""
import pytest

from config_schema import InitConfig


def test_latest_alias_rejected_when_mode_pretrain() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:latest")


def test_latest_alias_rejected_when_mode_warm_start() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(mode="warm_start", parent="wandb://ppo_hoop_blue_4:latest")


def test_latest_alias_ok_when_mode_scratch() -> None:
    # scratch ignores `parent` anyway; no validation needed.
    InitConfig(mode="scratch", parent="wandb://anything:latest")


def test_stable_alias_accepted() -> None:
    InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:prod")
    InitConfig(mode="pretrain", parent="wandb://ppo_hoop_blue_4:v3")
    InitConfig(mode="pretrain", parent="models/ppo_hoop_blue_4/best_model")


def test_fully_qualified_uri_latest_also_rejected() -> None:
    with pytest.raises(ValueError, match=":latest"):
        InitConfig(
            mode="pretrain",
            parent="wandb-artifact://shurioque/drone-quidditch/ppo_hoop_blue_4:latest",
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_init_latest_ban.py -v
```

Expected: FAIL — `InitConfig` doesn't validate.

- [ ] **Step 3: Add `__post_init__` to `InitConfig`**

Edit `config_schema.py:InitConfig` — add the `__post_init__` method below the existing fields:

```python
@dataclass
class InitConfig:
    """Mutex enforced by Hydra group choice (one conf/init/*.yaml at a time).

    Per-mode validity:
      - scratch:    all parent fields null
      - pretrain:   parent required; obs_surgery=False
      - resume:     parent_run required; parent_checkpoint optional
      - warm_start: parent required; obs_surgery=True
    """
    mode: str = "scratch"
    parent: str | None = None
    parent_run: str | None = None
    parent_checkpoint: str | None = None
    obs_surgery: bool = False
    new_dim_init_scale: float = 0.01  # only used when mode=warm_start

    def __post_init__(self) -> None:
        # Ban `:latest` alias in committed configs.  `:latest` shifts as new
        # versions land; a checked-in lineage parent must pin a stable alias
        # (`:prod`, `:<run_name>`) or an immutable version (`:v3`).  Schema-
        # level rejection prevents the footgun of a parent's meaning drifting
        # under a `git pull` from someone else's promote.
        if self.mode != "scratch" and self.parent is not None:
            is_wandb = self.parent.startswith(("wandb://", "wandb-artifact://"))
            if is_wandb and self.parent.endswith(":latest"):
                raise ValueError(
                    f"init.parent={self.parent!r}: `:latest` is banned in "
                    f"committed configs.  Pin a stable alias (`:prod`, "
                    f"`:<run_name>`) or an immutable version (`:v<N>`)."
                )
```

- [ ] **Step 4: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_init_latest_ban.py -v
```

Expected: PASS (5 cases).

- [ ] **Step 5: Run full unit suite — confirm no regression**

```bash
conda run -n uav python -m pytest tests/unit -x
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add config_schema.py tests/unit/test_init_latest_ban.py
git commit -m "feat(schema): ban :latest alias in committed init.parent URIs"
```

---

## Phase B — Run logger (charts replace TB)

### Task B1: `scripts/_wandb_init.py` — `init_wandb(cfg, run_dir, role)`

**Files:**
- Create: `scripts/_wandb_init.py`
- Create: `tests/unit/test_wandb_init.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_wandb_init.py`:

```python
"""init_wandb builds the right kwargs.  No actual wandb.init is called —
we mock at the SDK boundary so the test stays offline and fast."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import hydra_compose


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / "ppo_hoop_blue_5" / "20260514_120000"
    d.mkdir(parents=True)
    return d


def _call_init(role: str, cfg_overrides: list[str] | None = None, env: dict | None = None):
    """Helper: compose canary_team cfg, patch wandb.init, return its kwargs."""
    from scripts._wandb_init import init_wandb
    with hydra_compose(experiment="canary_team", overrides=cfg_overrides or []) as cfg:
        with patch("wandb.init") as mock:
            mock.return_value = MagicMock()
            saved_env = dict(os.environ)
            try:
                if env:
                    os.environ.update(env)
                init_wandb(cfg, run_dir=Path("runs/ppo_hoop_blue_5/20260514_120000"),
                           role=role)
            finally:
                os.environ.clear()
                os.environ.update(saved_env)
        return mock.call_args.kwargs


def test_name_and_id_match_run_dir_basename() -> None:
    kw = _call_init("train")
    # cfg.run_name comes from canary_team.yaml ("canary_team"); timestamp is
    # the basename of run_dir.
    assert kw["name"] == "canary_team_20260514_120000"
    assert kw["id"] == "canary_team_20260514_120000"


def test_group_is_run_name_not_timestamped() -> None:
    kw = _call_init("train")
    assert kw["group"] == "canary_team"


def test_job_type_train_when_no_sweep_env() -> None:
    kw = _call_init("train")
    assert kw["job_type"] == "train"


def test_job_type_sweep_train_when_wandb_sweep_id_set() -> None:
    kw = _call_init("train", env={"WANDB_SWEEP_ID": "abc123"})
    assert kw["job_type"] == "sweep-train"


def test_job_type_eval_when_role_eval() -> None:
    kw = _call_init("eval")
    assert kw["job_type"] == "eval"


def test_tags_include_obs_name_init_mode_learner() -> None:
    kw = _call_init("train")
    tags = set(kw["tags"])
    # canary_team.yaml sets learner_id=red_0, obs=team, init=scratch, opponent=beeline_blue
    assert "red_0" in tags
    assert "TEAM_ENV_OBS" in tags
    assert "scratch" in tags
    assert "beeline_blue" in tags


def test_tags_extra_appended() -> None:
    kw = _call_init("train", cfg_overrides=["wandb.tags_extra=[ablation,smoke]"])
    tags = set(kw["tags"])
    assert "ablation" in tags
    assert "smoke" in tags


def test_config_is_resolved_dict() -> None:
    kw = _call_init("train")
    cfg_dict = kw["config"]
    # Must be a dict (resolved OmegaConf), not a DictConfig — wandb's
    # auto-flatten works on plain dicts.
    assert isinstance(cfg_dict, dict)
    # Nested keys preserved.
    assert "trainer" in cfg_dict
    assert "lr" in cfg_dict["trainer"]


def test_mode_from_wandb_mode_env() -> None:
    # The conftest already sets WANDB_MODE=disabled.
    kw = _call_init("train")
    assert kw["mode"] == "disabled"


def test_resume_allow_so_make_resume_reattaches() -> None:
    kw = _call_init("train")
    assert kw["resume"] == "allow"


def test_dir_is_str_of_run_dir() -> None:
    kw = _call_init("train")
    assert kw["dir"] == "runs/ppo_hoop_blue_5/20260514_120000"


def test_project_from_cfg() -> None:
    kw = _call_init("train")
    assert kw["project"] == "drone-quidditch"


def test_project_override_via_env() -> None:
    kw = _call_init("train", env={"WANDB_PROJECT": "scratch-project"})
    assert kw["project"] == "scratch-project"


def test_entity_from_env_default_none() -> None:
    # Test: with no WANDB_ENTITY env and no override, entity is None.
    saved = os.environ.pop("WANDB_ENTITY", None)
    try:
        kw = _call_init("train")
        assert kw["entity"] is None
    finally:
        if saved is not None:
            os.environ["WANDB_ENTITY"] = saved


def test_entity_override_from_cfg() -> None:
    kw = _call_init("train", cfg_overrides=["wandb.entity_override=team-quidditch"])
    assert kw["entity"] == "team-quidditch"
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_wandb_init.py -v
```

Expected: FAIL — `scripts._wandb_init` doesn't exist.

- [ ] **Step 3: Implement `scripts/_wandb_init.py`**

```python
"""init_wandb — single call site for wandb.init.

Owns: run identity (name/id/group), tag derivation, config snapshot, mode/
entity resolution.  Called from scripts/train.py (after Hydra composes
cfg) and from scripts/eval_team.py when `WANDB=1` is set.

In WANDB_MODE=disabled, wandb.init returns a no-op stub; tests rely on
this (conftest.py sets the env var globally).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


# Map opponent _target_ class names to short tag-friendly strings.
_OPP_SHORT_NAMES = {
    "BeelineRed":          "beeline_red",
    "BeelineBlue":         "beeline_blue",
    "IntercepterBlue":     "intercepter_blue",
    "ZeroOpponent":        "zero",
    "FrozenPolicyOpponent": "frozen",
    "MixtureOpponent":     "mixture",
}


def _opponent_short_name(opp_cfg) -> str | None:
    """Last segment of cfg.opponent._target_ → short tag (or None for single-agent)."""
    target = opp_cfg.get("_target_") if isinstance(opp_cfg, (dict, DictConfig)) else None
    if not target:
        return None
    cls = target.rsplit(".", 1)[-1]
    return _OPP_SHORT_NAMES.get(cls, cls.lower())


def _resolve_job_type(role: str) -> str:
    """Promote `role` to sweep-train when WANDB_SWEEP_ID is set.

    `wandb agent` sets WANDB_SWEEP_ID automatically; reading it lets the
    dashboard filter "manual training" vs "sweep children" without any
    code in the sweep YAML to remember to set a flag.
    """
    if role == "train" and os.environ.get("WANDB_SWEEP_ID"):
        return "sweep-train"
    return role


def _build_tags(cfg: DictConfig) -> list[str]:
    """Derive filter tags from cfg.  Empty/None values are dropped."""
    tags: list[str] = []
    learner = cfg.env.get("learner_id") if "learner_id" in cfg.env else None
    if learner:
        tags.append(str(learner))
    if cfg.obs.get("name"):
        tags.append(str(cfg.obs.name))
    if cfg.init.get("mode"):
        tags.append(str(cfg.init.mode))
    opp = _opponent_short_name(cfg.get("opponent"))
    if opp:
        tags.append(opp)
    extra = list(cfg.wandb.get("tags_extra", []))
    tags.extend(str(t) for t in extra if t)
    return tags


def init_wandb(cfg: DictConfig, run_dir: Path, role: str) -> Any:
    """Initialize the wandb run for this Hydra-composed cfg.

    `role` ∈ {"train", "eval"}.  Promoted to "sweep-train" automatically
    when WANDB_SWEEP_ID is in the environment.

    Returns the wandb run object (or a disabled-mode stub).
    """
    timestamp = Path(run_dir).name
    run_id = f"{cfg.run_name}_{timestamp}"

    entity = os.environ.get("WANDB_ENTITY")
    if cfg.wandb.get("entity_override"):
        entity = str(cfg.wandb.entity_override)

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg.wandb.project),
        entity=entity,
        name=run_id,
        id=run_id,
        dir=str(run_dir),
        group=str(cfg.run_name),
        job_type=_resolve_job_type(role),
        tags=_build_tags(cfg),
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        mode=os.environ.get("WANDB_MODE", "online"),
        resume="allow",
        notes=str(cfg.wandb.get("notes", "")),
    )
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
conda run -n uav python -m pytest tests/unit/test_wandb_init.py -v
```

Expected: 14 cases PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_wandb_init.py tests/unit/test_wandb_init.py
git commit -m "feat(wandb): init_wandb single call site for run identity + tags"
```

---

### Task B2: Wire `init_wandb` + `WandbCallback` into `scripts/train.py`; drop `tensorboard_log`

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/_train_common.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/test_train_smoke_wandb_disabled.py`:

```python
"""End-to-end smoke: scripts.train runs with WANDB_MODE=disabled.

Runs a 2048-step canary_team training to confirm the wandb-callback wiring
doesn't blow up.  No actual wandb network calls.  Eval skipped via
trainer.total_timesteps below first eval threshold.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_train_canary_team_2048_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=canary_team",
            "trainer.total_timesteps=2048",
            "trainer.n_steps=1024",
            "eval.eval_freq_steps=999999999",       # skip eval
            "eval.checkpoint_freq_steps=999999999", # skip checkpoint
            "eval.video.enabled=false",             # skip video
            f"hydra.run.dir={tmp_path}/smoke_run",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout)
        sys.stderr.write(out.stderr)
    assert out.returncode == 0, "scripts.train exited non-zero"
    assert (tmp_path / "smoke_run" / "final_model.zip").exists()
    assert (tmp_path / "smoke_run" / ".hydra" / "config.yaml").exists()
    assert (tmp_path / "smoke_run" / ".hydra" / "meta.yaml").exists()
    # Confirm no events.out.tfevents.* files (TB output retired).
    tb_files = list((tmp_path / "smoke_run").rglob("events.out.tfevents.*"))
    assert tb_files == [], f"unexpected TB event files: {tb_files}"
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/integration/test_train_smoke_wandb_disabled.py -v
```

Expected: FAIL — either tfevents files present (TB still active), or assertion on wandb wiring.

- [ ] **Step 3: Edit `scripts/train.py` to remove `tensorboard_log` and add `init_wandb`**

Three changes in `scripts/train.py`:

**3a.** In `_build_or_load_model` (currently `scripts/train.py:138`), replace every `tensorboard_log=tb` with `tensorboard_log=None`. Also delete the `tb = str(run_dir)` line (it's no longer used). The full updated function:

```python
def _build_or_load_model(cfg: DictConfig, vec_env, run_dir: Path, seed: int):
    """Dispatch on cfg.init.mode to build PPO from scratch / pretrain / resume / warm_start."""
    ppo_kwargs = {
        "n_steps":       cfg.trainer.n_steps,
        "batch_size":    cfg.trainer.batch_size,
        "n_epochs":      cfg.trainer.n_epochs,
        "learning_rate": cfg.trainer.lr,
        "gamma":         cfg.trainer.gamma,
        "gae_lambda":    cfg.trainer.gae_lambda,
        "clip_range":    cfg.trainer.clip_range,
        "ent_coef":      cfg.trainer.ent_coef,
    }
    current_spec = SPEC_BY_NAME[cfg.obs.name]
    frame_stack = int(cfg.obs.n_stack)

    if cfg.init.mode == "scratch":
        return PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=None, seed=seed, verbose=0, **ppo_kwargs,
        ), 0

    if cfg.init.mode == "pretrain":
        parent = Path(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        if not parent_hydra.exists() and (parent.parent.parent / ".hydra").exists():
            parent_hydra = parent.parent.parent / ".hydra"
        if not parent_hydra.exists():
            # Legacy fallback: parent has info.toml (pre-Phase-6-migration model).
            legacy_info = parent.parent / "info.toml"
            if not legacy_info.exists():
                legacy_info = parent.parent.parent / "info.toml"
            if legacy_info.exists():
                check_obs_compat(legacy_info, current=current_spec,
                                  current_n_stack=frame_stack, surgery=False)
            else:
                raise FileNotFoundError(
                    f"Parent at {parent.parent} has no .hydra/ — run "
                    f"scripts/migrate_legacy_models.py first."
                )
        else:
            _check_obs_compat_from_hydra(parent_hydra, current_spec, frame_stack)
        model = PPO.load(str(parent), env=vec_env, tensorboard_log=None, verbose=0, **ppo_kwargs)
        chain_total = read_parent_chain_total_from_hydra(str(parent)) or int(model.num_timesteps)
        return model, chain_total

    if cfg.init.mode == "resume":
        run_root = Path("runs") / cfg.init.parent_run
        if not run_root.exists():
            raise FileNotFoundError(f"init.parent_run={cfg.init.parent_run}: no such dir under runs/")
        latest_trial = max(run_root.iterdir(), key=lambda p: p.name)
        ckpt = cfg.init.parent_checkpoint
        if ckpt is None:
            ckpts = sorted((latest_trial / "checkpoints").glob("*.zip"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints in {latest_trial}/checkpoints/")
            ckpt = str(ckpts[-1])
        parent_hydra = latest_trial / ".hydra"
        if parent_hydra.exists():
            _check_obs_compat_from_hydra(parent_hydra, current_spec, frame_stack)
        model = PPO.load(ckpt, env=vec_env, tensorboard_log=None, verbose=0,
                          learning_rate=cfg.trainer.lr)
        return model, 0

    if cfg.init.mode == "warm_start":
        from core.policies.warm_start import warm_start_ppo_by_spec
        parent = Path(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        if not parent_hydra.exists() and (parent.parent.parent / ".hydra").exists():
            parent_hydra = parent.parent.parent / ".hydra"
        parent_spec_name, parent_n_stack = _read_hydra_obs(parent_hydra)
        parent_spec = SPEC_BY_NAME[parent_spec_name] if parent_spec_name else SPEC_BY_NAME["SIMPLE_ENV_OBS"]
        model = warm_start_ppo_by_spec(
            old_checkpoint=str(parent),
            new_env=vec_env,
            parent_spec=parent_spec,
            parent_n_stack=parent_n_stack or 1,
            current_spec=current_spec,
            current_n_stack=frame_stack,
            new_dim_init_scale=cfg.init.new_dim_init_scale,
            tensorboard_log=None, seed=seed, verbose=0, **ppo_kwargs,
        )
        return model, 0

    raise ValueError(f"Unknown init.mode={cfg.init.mode}")
```

**3b.** In `main()`, add `init_wandb` + `WandbCallback`. Edit `scripts/train.py` `main()` so the section between `# 2) Build / load model` and `# 4) Build callbacks` reads:

```python
    # 1.5) Initialize wandb run (after Hydra cfg is composed, before model build)
    from scripts._wandb_init import init_wandb
    wandb_run = init_wandb(cfg, run_dir=run_dir, role="train")

    # 2) Build / load model
    model, parent_chain_total = _build_or_load_model(cfg, train_env, run_dir, seed)

    # 3) Write meta.yaml start fields
    write_meta_yaml(
        run_dir,
        parent_chain_total=parent_chain_total,
        init_mode=cfg.init.mode,
        parent_path=str(cfg.init.parent) if cfg.init.parent else None,
    )
```

**3c.** Append `WandbCallback` to the SB3 callback list. After the `callbacks = build_callbacks(...)` call, add:

```python
    # Wandb integration callback: hooks SB3's internal logger and forwards
    # every recorded value to wandb.log.  log="gradients" emits weight+grad
    # histograms; cfg-gated for cost.  In WANDB_MODE=disabled this is a no-op.
    from wandb.integration.sb3 import WandbCallback as _WandbCallback
    callbacks.append(
        _WandbCallback(
            verbose=0,
            log="gradients" if cfg.wandb.log_gradients else None,
            model_save_path=None,   # we handle artifact upload ourselves in Phase C
        )
    )
```

**3d.** At the very end of `main()` (after `append_meta_yaml_final_stats`), call `wandb.finish()`:

```python
        append_meta_yaml_final_stats(
            run_dir,
            wall_time_s=elapsed_s,
            completed_steps=completed_steps,
        )
        import wandb as _wandb
        _wandb.finish()
```

The full updated `main()` (replace from `@hydra.main` onward):

```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    seed = int(cfg.seed)

    # 1) Build env factory (instantiate cfg.env with reward + opponent injection)
    reward_stack = instantiate(cfg.reward, _convert_="all")
    env_factory = _build_env_factory(cfg)
    env_factory.reward_stack = reward_stack
    train_env = env_factory.build_train_env()

    # 1.5) Initialize wandb run (after Hydra cfg is composed, before model build)
    from scripts._wandb_init import init_wandb
    wandb_run = init_wandb(cfg, run_dir=run_dir, role="train")

    # 2) Build / load model
    model, parent_chain_total = _build_or_load_model(cfg, train_env, run_dir, seed)

    # 3) Write meta.yaml start fields
    write_meta_yaml(
        run_dir,
        parent_chain_total=parent_chain_total,
        init_mode=cfg.init.mode,
        parent_path=str(cfg.init.parent) if cfg.init.parent else None,
    )

    # 4) Build callbacks
    legacy_cfg = {
        "training": {
            "eval":      {"eval_freq_steps": cfg.eval.eval_freq_steps,
                          "n_eval_episodes": cfg.eval.n_eval_episodes},
            "callbacks": {"checkpoint_freq_steps": cfg.eval.checkpoint_freq_steps,
                          "video_every_n_evals":   cfg.eval.video.every_n_evals,
                          "video_fps":             cfg.eval.video.fps,
                          "video": {"grid":        cfg.eval.video.grid,
                                    "cells":       list(cfg.eval.video.cells),
                                    "cell_width":  cfg.eval.video.cell_width,
                                    "cell_height": cfg.eval.video.cell_height}},
        }
    }

    is_team = cfg.env._target_.endswith("TeamEnvFactory")
    if is_team:
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        team_cfg = _build_team_cfg(cfg)
        opp_spec = _opponent_spec_from_cfg(cfg)
        learner = env_factory.learner_id
        def eval_env_fn():
            team = QuidditchTeamEnv(cfg=team_cfg)
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
    else:
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        rs = bool(cfg.curriculum.randomise_start)
        eps = float(cfg.curriculum.episode_seconds)
        def eval_env_fn():
            return QuidditchSimpleEnv(render_mode=None, randomise_start=rs, episode_seconds=eps)

    video_env_fn = env_factory.build_video_env_fn() if cfg.eval.video.enabled else None
    frame_stack = int(cfg.obs.n_stack)

    callbacks = build_callbacks(
        run_dir=run_dir,
        eval_env_fn=eval_env_fn,
        config=legacy_cfg,
        n_envs=cfg.env.n_envs,
        video_env_fn=video_env_fn,
        verbose=0,
        frame_stack=frame_stack,
    )

    # Wandb integration callback: hooks SB3's internal logger and forwards
    # every recorded value to wandb.log.  log="gradients" emits weight+grad
    # histograms; cfg-gated for cost.  In WANDB_MODE=disabled this is a no-op.
    from wandb.integration.sb3 import WandbCallback as _WandbCallback
    callbacks.append(
        _WandbCallback(
            verbose=0,
            log="gradients" if cfg.wandb.log_gradients else None,
            model_save_path=None,
        )
    )

    # 5) Train
    started = datetime.now()
    try:
        model.learn(
            total_timesteps=cfg.trainer.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=(cfg.init.mode != "resume"),
            progress_bar=True,
        )
    finally:
        elapsed_s = (datetime.now() - started).total_seconds()
        completed_steps = int(model.num_timesteps)
        model.save(str(run_dir / "final_model"))
        append_meta_yaml_final_stats(
            run_dir,
            wall_time_s=elapsed_s,
            completed_steps=completed_steps,
        )
        import wandb as _wandb
        _wandb.finish()
```

- [ ] **Step 4: Run smoke test**

```bash
conda run -n uav python -m pytest tests/integration/test_train_smoke_wandb_disabled.py -v
```

Expected: PASS (training completes; no tfevents files written).

- [ ] **Step 5: Run scoring canary to confirm fingerprint preserved**

```bash
conda run -n uav python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: both canaries pass (step 434 / reward 7.3837 for single-agent; team canary holds).

- [ ] **Step 6: Run full pytest suite**

```bash
conda run -n uav python -m pytest tests -x
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py tests/integration/test_train_smoke_wandb_disabled.py
git commit -m "feat(train): wire wandb logger, drop tensorboard_log from SB3"
```

---

### Task B3: Rewrite `VideoRecorderCallback._log_to_tensorboard` → `_log_to_wandb`

**Files:**
- Modify: `scripts/callbacks.py:215-243` (the `_log_to_tensorboard` method)
- Modify: `scripts/callbacks.py:114-134` (the moviepy probe — `wandb.Video` needs moviepy too; comment stays accurate)
- Modify: `scripts/callbacks.py:202` (callsite: rename + arg shape)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_video_callback_wandb.py`:

```python
"""VideoRecorderCallback emits per-cam wandb.Video objects, not TB Videos."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_log_to_wandb_emits_per_cam_video() -> None:
    """_log_to_wandb fans out to one wandb.log call with per-cam keys."""
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = True
    cb.grid = True
    cb.fps = 20
    cb.grid_cams = ("south", "east", "top", "fixed")
    cb.model = MagicMock()
    cb.model.num_timesteps = 12345

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    per_cam = {
        "south":  list(frames),
        "east":   list(frames),
        "top":    list(frames),
        "fixed":  list(frames),
    }

    with patch("wandb.log") as mock_log, patch("wandb.Video") as mock_video:
        mock_video.side_effect = lambda *a, **k: MagicMock()
        cb._log_to_wandb(per_cam)

    # Exactly one wandb.log call, with the four cam-keyed entries plus the step.
    assert mock_log.call_count == 1
    payload = mock_log.call_args.args[0] if mock_log.call_args.args else mock_log.call_args.kwargs["data"]
    keys = set(payload.keys())
    assert keys == {"eval/video/south", "eval/video/east", "eval/video/top", "eval/video/fixed"}
    # Step passed via kwarg.
    assert mock_log.call_args.kwargs["step"] == 12345


def test_log_to_wandb_single_cam_mode() -> None:
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = True
    cb.grid = False
    cb.fps = 20
    cb.grid_cams = ("south", "east", "top", "fixed")
    cb.model = MagicMock()
    cb.model.num_timesteps = 42

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    per_cam = {"fixed": frames}

    with patch("wandb.log") as mock_log, patch("wandb.Video") as mock_video:
        mock_video.side_effect = lambda *a, **k: MagicMock()
        cb._log_to_wandb(per_cam)

    payload = mock_log.call_args.args[0] if mock_log.call_args.args else mock_log.call_args.kwargs["data"]
    assert set(payload.keys()) == {"eval/video"}


def test_log_to_wandb_noop_when_moviepy_missing() -> None:
    from scripts.callbacks import VideoRecorderCallback
    cb = VideoRecorderCallback.__new__(VideoRecorderCallback)
    cb._moviepy_ok = False
    cb.grid = True
    cb.fps = 20
    cb.grid_cams = ("south",)
    cb.model = MagicMock()

    with patch("wandb.log") as mock_log:
        cb._log_to_wandb({"south": [np.zeros((10, 10, 3), dtype=np.uint8)]})

    assert mock_log.call_count == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_video_callback_wandb.py -v
```

Expected: FAIL — `_log_to_wandb` doesn't exist.

- [ ] **Step 3: Rewrite the method**

In `scripts/callbacks.py`, replace the `_log_to_tensorboard` method (currently `scripts/callbacks.py:215-243`) with `_log_to_wandb`:

```python
    def _log_to_wandb(self, per_cam: dict[str, list[np.ndarray]]) -> None:
        """Log each cam stream as a separate wandb.Video.

        Grid mode emits one entry per cam under ``eval/video/<cam>`` (with
        the cam name lowercased for tag consistency); single-cam mode emits
        one entry under ``eval/video``.  wandb.Video accepts ``(T, H, W, C)``
        uint8 arrays and encodes the mp4 in-memory.

        In WANDB_MODE=disabled, wandb.log is a no-op — this method runs but
        nothing leaves the process.  We still gate on _moviepy_ok because
        wandb.Video uses moviepy for the in-memory mp4 encode (same dep as
        the on-disk grid writer; one probe suffices).
        """
        if not self._moviepy_ok:
            return
        import wandb

        payload: dict[str, "wandb.Video"] = {}
        for name, frames in per_cam.items():
            stack = np.stack(frames)  # (T, H, W, 3) uint8
            tag = f"eval/video/{name.lower()}" if self.grid else "eval/video"
            payload[tag] = wandb.Video(stack, fps=self.fps, format="mp4")
        wandb.log(payload, step=int(self.model.num_timesteps))
```

Also update the call site at `scripts/callbacks.py:202`:

```python
        self._log_to_wandb(per_cam)
```

(Just the method name change — args unchanged.)

Also update the comment block at `scripts/callbacks.py:68-75` so the docstring no longer refers to TB:

Find this block (in the class docstring):

```python
    Each recorded episode is also logged to TensorBoard.  In grid mode every
    cell becomes its own video under ``eval/video/<cam>`` (so TB shows four
    independent clips, easier to read than the stitched grid at TB's preview
    size).  In single-cam mode there's one clip under ``eval/video``.  Either
    way the on-disk mp4 still contains the stitched grid.  TB embedding needs
    ``moviepy<2.0`` (torch's SummaryWriter.add_video imports moviepy.editor,
    removed in 2.x); missing it disables TB videos but mp4 writes are unaffected.
```

Replace with:

```python
    Each recorded episode is also logged to W&B.  In grid mode every cell
    becomes its own ``wandb.Video`` under ``eval/video/<cam>`` (the
    dashboard renders four independent media panels).  In single-cam mode
    there's one clip under ``eval/video``.  Either way the on-disk mp4
    still contains the stitched grid.  wandb.Video encodes mp4 in-memory
    via moviepy<2.0 (pinned because moviepy 2.x removed moviepy.editor);
    missing moviepy disables wandb uploads but on-disk mp4 writes are
    unaffected.
```

Also update the moviepy probe message at `scripts/callbacks.py:132-134`:

```python
        except ImportError:
            self._moviepy_ok = False
            print(
                f"{_ts()} ⚠️  [VideoRecorder] moviepy not found — wandb video "
                "logging disabled. Install with: pip install 'moviepy<2.0'"
            )
```

Also delete the now-unused import `from stable_baselines3.common.logger import Video` at `scripts/callbacks.py:12`.

- [ ] **Step 4: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_video_callback_wandb.py -v
```

Expected: 3 cases PASS.

- [ ] **Step 5: Re-run smoke + canary**

```bash
conda run -n uav python -m pytest tests/integration/test_train_smoke_wandb_disabled.py tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/callbacks.py tests/unit/test_video_callback_wandb.py
git commit -m "feat(video): per-cam wandb.Video logging; remove TB embed path"
```

---

### Task B4: Remove `tensorboard` Makefile target

**Files:**
- Modify: `Makefile:50` (drop `tensorboard` from `.PHONY`)
- Modify: `Makefile:101-104` (delete the `tensorboard:` target)
- Modify: `Makefile:54` (drop tensorboard from `make help`)

- [ ] **Step 1: Edit Makefile**

In `Makefile:50`, delete `tensorboard` from the `.PHONY` line:

```makefile
.PHONY: help test test-fast test-warm camera-test demo train resume eval eval-headless lineage promote install clean list-runs eval-team
```

Delete the entire `tensorboard` target block (lines 101-104):

```makefile
tensorboard: ## 📊 Launch TensorBoard — all runs, or [RUN_NAME=...] for one config
	@PYTHONWARNINGS=ignore $(CONDA_RUN) tensorboard \
	  --logdir $(if $(filter command line,$(origin RUN_NAME)),$(RUNS_DIR)/$(RUN_NAME),$(RUNS_DIR)) \
	  2>&1 | grep --line-buffered -v "pkg_resources\|TensorFlow installation not found\|experimental fast data\|--load_fast\|issues on GitHub\|tensorflow/tensorboard\|^[[:space:]]*$$"
```

- [ ] **Step 2: Verify the rest of `make help` still works**

```bash
cd "$(git rev-parse --show-toplevel)" && make help
```

Expected: lists targets without `tensorboard`. No tensorboard-related errors.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore(make): drop tensorboard target — replaced by wandb dashboard"
```

---

## Phase C — Artifact registry

### Task C1: `scripts/_artifact_io.py:resolve_parent` (URI parser + cache lookup)

**Files:**
- Create: `scripts/_artifact_io.py`
- Create: `tests/unit/test_artifact_resolve.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_artifact_resolve.py`:

```python
"""resolve_parent maps `wandb://run:alias` URIs to local cache paths,
preferring committed `models/<run>/` when its `_wandb_metadata.json`
pins the same immutable version."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── URI parsing ──────────────────────────────────────────────────────────────
def test_parse_filesystem_path_returns_as_is(tmp_path: Path) -> None:
    from scripts._artifact_io import resolve_parent
    target = tmp_path / "models" / "x" / "best_model.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    out = resolve_parent(str(target))
    assert out == target


def test_parse_relative_path_returns_as_is() -> None:
    from scripts._artifact_io import resolve_parent
    out = resolve_parent("models/ppo_hoop_blue_4/best_model")
    assert str(out) == "models/ppo_hoop_blue_4/best_model"


def test_parse_wandb_shorthand_uri() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    qual = _parse_wandb_uri("wandb://ppo_hoop_blue_4:prod")
    assert qual.entity is None
    assert qual.project is None
    assert qual.name == "ppo_hoop_blue_4"
    assert qual.alias == "prod"


def test_parse_wandb_fully_qualified_uri() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    qual = _parse_wandb_uri("wandb-artifact://me/proj/ppo_hoop_blue_4:v3")
    assert qual.entity == "me"
    assert qual.project == "proj"
    assert qual.name == "ppo_hoop_blue_4"
    assert qual.alias == "v3"


def test_parse_uri_missing_alias_raises() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    with pytest.raises(ValueError, match="missing alias"):
        _parse_wandb_uri("wandb://ppo_hoop_blue_4")


# ── Committed-cache hit ──────────────────────────────────────────────────────
def test_resolve_hits_committed_dir_when_version_matches(tmp_path: Path) -> None:
    """If models/<run>/_wandb_metadata.json pins v3 and alias :prod resolves
    to v3 on the API, return the committed path without download."""
    from scripts._artifact_io import resolve_parent

    # Set up a committed dir with metadata pinning v3.
    committed = tmp_path / "models" / "ppo_hoop_blue_4"
    committed.mkdir(parents=True)
    (committed / "best_model.zip").write_bytes(b"committed-bytes")
    (committed / "_wandb_metadata.json").write_text(json.dumps({
        "name": "ppo_hoop_blue_4",
        "version": "v3",
        "entity": "me",
        "project": "drone-quidditch",
    }))

    # Mock the wandb API: artifact for :prod resolves to v3.
    art = MagicMock()
    art.version = "v3"
    art.name = "ppo_hoop_blue_4:v3"
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_blue_4:prod", models_root=tmp_path / "models")

    assert out == committed / "best_model.zip"
    # use_artifact still recorded for lineage.
    run.use_artifact.assert_called_once()


def test_resolve_falls_through_when_committed_pins_different_version(tmp_path: Path) -> None:
    """If committed dir pins v2 but :prod points to v3, download v3."""
    from scripts._artifact_io import resolve_parent

    committed = tmp_path / "models" / "ppo_hoop_blue_4"
    committed.mkdir(parents=True)
    (committed / "_wandb_metadata.json").write_text(json.dumps({
        "name": "ppo_hoop_blue_4",
        "version": "v2",
    }))

    cache_root = tmp_path / "models" / ".cache"
    cache_dir = cache_root / "ppo_hoop_blue_4_v3"

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"downloaded-bytes")
        return root

    art = MagicMock()
    art.version = "v3"
    art.name = "ppo_hoop_blue_4:v3"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()
    run.use_artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_blue_4:prod", models_root=tmp_path / "models")

    assert out == cache_dir / "best_model.zip"
    assert out.read_bytes() == b"downloaded-bytes"


# ── No committed dir → download ──────────────────────────────────────────────
def test_resolve_downloads_when_no_committed_dir(tmp_path: Path) -> None:
    from scripts._artifact_io import resolve_parent

    cache_dir = tmp_path / "models" / ".cache" / "ppo_hoop_red_1_v0"

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"downloaded")
        return root

    art = MagicMock()
    art.version = "v0"
    art.name = "ppo_hoop_red_1:v0"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()
    run.use_artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_red_1:prod", models_root=tmp_path / "models")

    assert out == cache_dir / "best_model.zip"


def test_resolve_no_active_wandb_run_still_works(tmp_path: Path) -> None:
    """When wandb.run is None (offline / disabled outside training), the
    resolver skips use_artifact but still downloads."""
    from scripts._artifact_io import resolve_parent

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"x")
        return root

    art = MagicMock()
    art.version = "v0"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", None):
            out = resolve_parent("wandb://ppo_hoop_red_1:prod", models_root=tmp_path / "models")

    assert (tmp_path / "models" / ".cache" / "ppo_hoop_red_1_v0" / "best_model.zip").exists()
    assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_artifact_resolve.py -v
```

Expected: FAIL — `scripts._artifact_io` doesn't exist.

- [ ] **Step 3: Implement `scripts/_artifact_io.py`**

```python
"""Artifact registry helpers.

Two functions:

  - resolve_parent(uri_or_path, models_root=Path("models")) → Path
      Maps `wandb://run:alias` (or `wandb-artifact://entity/project/run:alias`)
      to a local checkpoint path.  Filesystem paths are returned as-is.
      Wandb URIs are: alias-resolved → cache-checked → downloaded if needed.
      Always calls wandb.use_artifact on the current run for lineage.

  - log_run_artifact(run, run_dir, cfg, parent_chain_total, best_eval_reward)
      Called at the end of every training run.  Logs best_model.zip + .hydra/
      as `<cfg.run_name>:latest` (plus auto-versioned :v<N>).

The committed `models/<run>/_wandb_metadata.json` pins the immutable
version (e.g. v3) — never the alias.  If a later promote shifts :prod to a
newer version, the resolver detects the mismatch and falls through to
downloading the new version instead of silently serving the stale
committed checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


@dataclass(frozen=True)
class _WandbURI:
    """Parsed wandb:// or wandb-artifact:// URI."""
    entity: str | None
    project: str | None
    name: str
    alias: str

    def short_form(self) -> str:
        return f"wandb://{self.name}:{self.alias}"

    def for_api(self, default_entity: str | None, default_project: str | None) -> str:
        """The full entity/project/name:alias string the wandb API needs."""
        ent = self.entity or default_entity
        proj = self.project or default_project
        if ent is None or proj is None:
            # Fall through; wandb.Api() will use defaults from env + workspace.
            return f"{self.name}:{self.alias}"
        return f"{ent}/{proj}/{self.name}:{self.alias}"


def _parse_wandb_uri(uri: str) -> _WandbURI:
    """Parse `wandb://run:alias` or `wandb-artifact://entity/project/run:alias`."""
    if uri.startswith("wandb-artifact://"):
        body = uri[len("wandb-artifact://"):]
        parts = body.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid wandb-artifact URI: {uri!r} "
                "(expected wandb-artifact://entity/project/name:alias)"
            )
        entity, project, name_and_alias = parts
    elif uri.startswith("wandb://"):
        entity = None
        project = None
        name_and_alias = uri[len("wandb://"):]
    else:
        raise ValueError(f"Not a wandb URI: {uri!r}")

    if ":" not in name_and_alias:
        raise ValueError(f"Invalid wandb URI {uri!r}: missing alias (e.g. `:prod`)")
    name, alias = name_and_alias.rsplit(":", 1)
    if not name or not alias:
        raise ValueError(f"Invalid wandb URI {uri!r}: empty name or alias")
    return _WandbURI(entity=entity, project=project, name=name, alias=alias)


def _is_wandb_uri(s: str) -> bool:
    return s.startswith(("wandb://", "wandb-artifact://"))


def _committed_metadata(committed_dir: Path) -> dict | None:
    """Read `_wandb_metadata.json` if present."""
    p = committed_dir / "_wandb_metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def resolve_parent(
    uri_or_path: str | Path,
    models_root: Path = Path("models"),
) -> Path:
    """Resolve a parent reference to a local checkpoint Path.

    Filesystem paths (anything not starting with `wandb://` or
    `wandb-artifact://`) are returned as-is.  Wandb URIs go through:

      1. Resolve alias → immutable version via wandb.Api().artifact(uri).
      2. If models/<name>/_wandb_metadata.json pins the same version,
         return the committed path.
      3. Else download via artifact.download(root=models/.cache/<name>_v<N>/).
      4. In both cases, if wandb.run is active, call use_artifact for lineage.

    Returns the path to the loadable best_model.zip (or whatever the
    artifact wrapped — convention is best_model.zip).
    """
    s = str(uri_or_path)
    if not _is_wandb_uri(s):
        return Path(s)

    parsed = _parse_wandb_uri(s)
    api = wandb.Api()
    # Resolve alias → immutable version.  wandb.Api() pulls defaults from
    # env/workspace when entity/project are absent in our shorthand URIs.
    art = api.artifact(parsed.for_api(default_entity=None, default_project=None))
    version = art.version

    # Lineage edge: only if there's a live training run.
    if wandb.run is not None:
        wandb.run.use_artifact(art)

    # Cache hit on committed dir?
    committed = Path(models_root) / parsed.name
    meta = _committed_metadata(committed)
    if meta is not None and meta.get("version") == version and meta.get("name") == parsed.name:
        cp = committed / "best_model.zip"
        if cp.exists():
            return cp
        # Metadata pins but best_model.zip is missing — fall through to download.

    # Download into the gitignored cache.
    cache_dir = Path(models_root) / ".cache" / f"{parsed.name}_{version}"
    art.download(root=str(cache_dir))
    return cache_dir / "best_model.zip"


def log_run_artifact(
    run: Any,
    run_dir: Path,
    cfg: Any,
    parent_chain_total: int,
    best_eval_reward: float | None,
) -> None:
    """Log this run's best_model + .hydra/ as a wandb artifact.

    Aliased `:latest` automatically; further aliases (`:prod`, `:<run_name>`)
    are added by scripts/promote.py.  No-op when wandb.run is None or in
    WANDB_MODE=disabled.
    """
    if run is None or getattr(run, "disabled", False):
        return
    best = Path(run_dir) / "best_model.zip"
    hydra_dir = Path(run_dir) / ".hydra"
    if not best.exists():
        # No best_model emitted (e.g., training crashed before first eval).
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
        },
    )
    art.add_file(str(best))
    if hydra_dir.exists():
        art.add_dir(str(hydra_dir), name=".hydra")
    run.log_artifact(art, aliases=["latest"])
```

- [ ] **Step 4: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_artifact_resolve.py -v
```

Expected: 9 cases PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_artifact_io.py tests/unit/test_artifact_resolve.py
git commit -m "feat(artifact): resolve_parent + log_run_artifact helpers"
```

---

### Task C2: Wire `log_run_artifact` into `scripts/train.py` end-of-run

**Files:**
- Modify: `scripts/train.py` (`main()`)

- [ ] **Step 1: Write failing test**

Add to `tests/integration/test_train_smoke_wandb_disabled.py`:

```python
def test_train_smoke_calls_log_run_artifact(tmp_path: Path) -> None:
    """The end-of-run artifact log fires (and silently no-ops in disabled mode)."""
    import os, subprocess, sys
    from unittest.mock import patch

    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}

    # Use an offline-mode side process; we can't easily intercept the in-process
    # call from a subprocess, so this test confirms only that the smoke doesn't
    # fail when log_run_artifact is wired in.  (Mocked unit tests cover the
    # actual artifact-construction behavior.)
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=canary_team",
            "trainer.total_timesteps=2048",
            "trainer.n_steps=1024",
            "eval.eval_freq_steps=999999999",
            "eval.checkpoint_freq_steps=999999999",
            "eval.video.enabled=false",
            f"hydra.run.dir={tmp_path}/smoke_artifact",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0
```

(The unit-level artifact-construction is in C1; this just confirms the wiring doesn't crash.)

- [ ] **Step 2: Edit `scripts/train.py:main()` to call `log_run_artifact`**

Update the `finally:` block to call `log_run_artifact` before `wandb.finish()`. Also capture `best_eval_reward` from the EvalCallback. The updated tail of `main()`:

```python
    # 5) Train
    started = datetime.now()
    best_eval_reward: float | None = None
    try:
        model.learn(
            total_timesteps=cfg.trainer.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=(cfg.init.mode != "resume"),
            progress_bar=True,
        )
    finally:
        elapsed_s = (datetime.now() - started).total_seconds()
        completed_steps = int(model.num_timesteps)
        model.save(str(run_dir / "final_model"))

        # EvalCallback writes <run_dir>/best_model.zip + tracks best_mean_reward
        # on the callback instance.  Reach in for the value if eval ran.
        for cb in callbacks:
            if hasattr(cb, "best_mean_reward"):
                rwd = float(cb.best_mean_reward)
                if rwd > -1e9:                                # SB3 sentinel default
                    best_eval_reward = rwd
                break

        append_meta_yaml_final_stats(
            run_dir,
            wall_time_s=elapsed_s,
            completed_steps=completed_steps,
            best_eval_reward=best_eval_reward,
        )

        # Log the best_model + .hydra/ as a wandb artifact (no-op in disabled).
        from scripts._artifact_io import log_run_artifact
        log_run_artifact(
            run=wandb_run,
            run_dir=run_dir,
            cfg=cfg,
            parent_chain_total=parent_chain_total,
            best_eval_reward=best_eval_reward,
        )

        import wandb as _wandb
        _wandb.finish()
```

- [ ] **Step 3: Run smoke test**

```bash
conda run -n uav python -m pytest tests/integration/test_train_smoke_wandb_disabled.py -v
```

Expected: both cases PASS.

- [ ] **Step 4: Re-run canaries**

```bash
conda run -n uav python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/integration/test_train_smoke_wandb_disabled.py
git commit -m "feat(train): log best_model as wandb artifact at end of run"
```

---

### Task C3: Wire `resolve_parent` into `scripts/train.py` init.parent loads

**Files:**
- Modify: `scripts/train.py` — three `init.parent` load branches

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_train_resolve_parent_wiring.py`:

```python
"""train.py's three init.parent load branches all route through resolve_parent."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest


def test_pretrain_branch_calls_resolve_parent(tmp_path: Path) -> None:
    """If init.mode=pretrain and parent is a wandb URI, resolve_parent
    is called before PPO.load."""
    from scripts.train import _build_or_load_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "trainer": {"n_steps": 1, "batch_size": 1, "n_epochs": 1, "lr": 1e-4,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                    "ent_coef": 0.01},
        "obs": {"name": "AUGMENTED_OBS", "n_stack": 3},
        "init": {"mode": "pretrain", "parent": "wandb://ppo_hoop_blue_4:prod",
                 "new_dim_init_scale": 0.01},
    })

    fake_local = tmp_path / "models" / "ppo_hoop_blue_4" / "best_model.zip"
    fake_local.parent.mkdir(parents=True)
    fake_local.write_bytes(b"")

    fake_hydra = tmp_path / "models" / "ppo_hoop_blue_4" / ".hydra"
    fake_hydra.mkdir()
    (fake_hydra / "config.yaml").write_text(
        "obs:\n  name: AUGMENTED_OBS\n  n_stack: 3\n"
    )

    vec_env = MagicMock()

    with patch("scripts.train.resolve_parent", return_value=fake_local) as mock_resolve:
        with patch("scripts.train.PPO") as mock_ppo:
            mock_ppo.load.return_value = MagicMock(num_timesteps=0)
            with patch("scripts.train.read_parent_chain_total_from_hydra",
                       return_value=42):
                _build_or_load_model(cfg, vec_env, tmp_path, seed=42)

    mock_resolve.assert_called_once_with("wandb://ppo_hoop_blue_4:prod")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_train_resolve_parent_wiring.py -v
```

Expected: FAIL — `resolve_parent` not imported / not called.

- [ ] **Step 3: Edit `scripts/train.py:_build_or_load_model` to use `resolve_parent`**

Add the import at the top of `scripts/train.py`, near the other `from scripts._train_common import ...`:

```python
from scripts._artifact_io import resolve_parent
```

Then in `_build_or_load_model`, replace the three `parent = Path(cfg.init.parent)` lines:

**Pretrain branch:**

```python
    if cfg.init.mode == "pretrain":
        parent = resolve_parent(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        # ... rest unchanged
```

**Warm-start branch:**

```python
    if cfg.init.mode == "warm_start":
        from core.policies.warm_start import warm_start_ppo_by_spec
        parent = resolve_parent(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        # ... rest unchanged
```

**Resume branch:** unchanged (resume uses `cfg.init.parent_run` to look up local `runs/<name>/`, not wandb).

- [ ] **Step 4: Run test, expect pass**

```bash
conda run -n uav python -m pytest tests/unit/test_train_resolve_parent_wiring.py -v
```

Expected: PASS.

- [ ] **Step 5: Re-run smoke + canaries**

```bash
conda run -n uav python -m pytest tests/integration/test_train_smoke_wandb_disabled.py tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py tests/unit/test_train_resolve_parent_wiring.py
git commit -m "feat(train): route init.parent through resolve_parent (accepts wandb URIs)"
```

---

### Task C4: `scripts/promote.py` + Makefile wiring

**Files:**
- Create: `scripts/promote.py`
- Create: `tests/unit/test_promote.py`
- Modify: `Makefile` — replace inline `promote:` shell with `python -m scripts.promote`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_promote.py`:

```python
"""scripts.promote: alias the artifact, copy best_model + .hydra into
models/<run_name>/, write pinned `_wandb_metadata.json`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_run_dir(tmp_path: Path, run_name: str) -> Path:
    """Build a fake completed run dir."""
    run_dir = tmp_path / "runs" / run_name / "20260514_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.zip").write_bytes(b"model-bytes")
    hydra = run_dir / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text(f"run_name: {run_name}\n")
    (hydra / "meta.yaml").write_text("git_hash: abc123\nparent_chain_total: 0\n")
    return run_dir


def test_promote_copies_into_models_dir(tmp_path: Path) -> None:
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_5")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v0"
    art.aliases = ["latest"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_blue_5",
                        models_root=models_root)

    dest = models_root / "ppo_hoop_blue_5"
    assert (dest / "best_model.zip").read_bytes() == b"model-bytes"
    assert (dest / ".hydra" / "config.yaml").exists()
    meta = json.loads((dest / "_wandb_metadata.json").read_text())
    assert meta["name"] == "ppo_hoop_blue_5"
    assert meta["version"] == "v0"             # IMMUTABLE, not the alias
    assert "prod" in meta["aliases"]
    assert "ppo_hoop_blue_5" in meta["aliases"]


def test_promote_adds_prod_and_run_name_aliases(tmp_path: Path) -> None:
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_red_2")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v3"
    art.aliases = ["latest", "v3"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_red_2",
                        models_root=models_root)

    # Verify the aliases list passed to art.save (the wandb API updates
    # aliases by mutating .aliases and calling .save()).
    assert "prod" in art.aliases
    assert "ppo_hoop_red_2" in art.aliases
    art.save.assert_called_once()


def test_promote_idempotent_alias_add(tmp_path: Path) -> None:
    """Adding `:prod` when it's already there should not duplicate."""
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_4")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v2"
    art.aliases = ["latest", "v2", "prod", "ppo_hoop_blue_4"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_blue_4",
                        models_root=models_root)

    assert art.aliases.count("prod") == 1
    assert art.aliases.count("ppo_hoop_blue_4") == 1


def test_promote_no_best_model_fails(tmp_path: Path) -> None:
    import pytest
    from scripts.promote import promote_run_dir

    run_dir = tmp_path / "runs" / "x" / "20260514_120000"
    run_dir.mkdir(parents=True)
    # No best_model.zip.

    with pytest.raises(FileNotFoundError, match="best_model.zip"):
        promote_run_dir(run_dir=run_dir, run_name="x", models_root=tmp_path / "models")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_promote.py -v
```

Expected: FAIL — `scripts.promote` doesn't exist.

- [ ] **Step 3: Implement `scripts/promote.py`**

```python
"""Promote a training run's best model to canonical / vendored status.

Two-step:

  1. Wandb side: look up the most recent artifact logged by this run, add
     aliases `prod` and `<run_name>` (mutable aliases that move as new
     versions get promoted), persist via art.save().

  2. Repo side: copy best_model.zip + .hydra/ → models/<run_name>/, write
     `_wandb_metadata.json` pinning the IMMUTABLE version (`v3`, not `prod`).
     Print a hint reminding the user to `git add && git commit` if they
     want to vendor the checkpoint.

Usage:
    python -m scripts.promote runs/ppo_hoop_blue_5/20260514_120000

Or via the Makefile:
    make promote RUN_NAME=ppo_hoop_blue_5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb
from omegaconf import OmegaConf


def _resolve_run_name(run_dir: Path) -> str:
    """Read run_name from the run's .hydra/config.yaml."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return str(cfg.run_name)


def _find_run_artifact(run_name: str, timestamp: str):
    """Find the wandb artifact logged by run_id=<run_name>_<timestamp>.

    Returns the artifact object whose `logged_by()` matches the run id.
    """
    api = wandb.Api()
    run_id = f"{run_name}_{timestamp}"
    # Walk this artifact name's versions; the one logged by `run_id` is ours.
    art_path = f"{run_name}:latest"
    art = api.artifact(art_path)
    logged_by = art.logged_by()
    if logged_by is not None and logged_by.id == run_id:
        return art
    # Fallback: walk all versions.  Rare — only hit if a newer training run
    # has already logged after this one but before promote.
    artifact_type = api.artifact_type("model", project=art.entity + "/" + art.project)
    for collection in artifact_type.collections():
        if collection.name != run_name:
            continue
        for a in collection.artifacts():
            rb = a.logged_by()
            if rb is not None and rb.id == run_id:
                return a
    raise RuntimeError(
        f"No wandb artifact found for run id {run_id!r}.  "
        "Did training complete with a live wandb connection?"
    )


def promote_run_dir(run_dir: Path, run_name: str, models_root: Path) -> None:
    """Two-step promote: alias the artifact, copy + pin into models/."""
    run_dir = Path(run_dir).resolve()
    src = run_dir / "best_model.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — was eval triggered, or did training crash early?"
        )

    timestamp = run_dir.name
    art = _find_run_artifact(run_name, timestamp)

    # Mutable aliases: add `prod` + `<run_name>` if not present.
    aliases = list(art.aliases)
    for alias in ("prod", run_name):
        if alias not in aliases:
            aliases.append(alias)
    art.aliases = aliases
    art.save()

    # Repo side: copy + pin.
    dest = Path(models_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "best_model.zip")
    hydra_src = run_dir / ".hydra"
    if hydra_src.exists():
        hydra_dest = dest / ".hydra"
        if hydra_dest.exists():
            shutil.rmtree(hydra_dest)
        shutil.copytree(hydra_src, hydra_dest)
    metadata = {
        "name":     run_name,
        "version":  art.version,
        "entity":   getattr(art, "entity", None),
        "project":  getattr(art, "project", None),
        "aliases":  list(art.aliases),
        "logged_by_run_id": f"{run_name}_{timestamp}",
    }
    (dest / "_wandb_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"")
    print(f"  Run:      {run_dir}")
    print(f"  Wandb:    {run_name}:{art.version}  (aliases: {sorted(art.aliases)})")
    print(f"  Vendored: {dest}")
    print(f"")
    print(f"  To use as a pretrain parent in a new experiment YAML:")
    print(f"    init:")
    print(f"      parent: wandb://{run_name}:prod")
    print(f"")
    print(f"  To vendor this checkpoint into the repo:")
    print(f"    git add {dest} && git commit -m 'model: promote {run_name}'")


def main() -> None:
    p = argparse.ArgumentParser(description="Promote a run's best_model to canonical.")
    p.add_argument("run_dir", help="runs/<run_name>/<timestamp>/")
    p.add_argument("--models-root", default="models", help="defaults to models/")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_name = _resolve_run_name(run_dir)
    promote_run_dir(run_dir=run_dir, run_name=run_name,
                    models_root=Path(args.models_root))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_promote.py -v
```

Expected: 4 cases PASS.

- [ ] **Step 5: Edit `Makefile:113-132` `promote:` target**

Replace the entire inline shell with a call to `scripts/promote.py`:

```makefile
promote: ## 🏆 Promote best model — alias on wandb + copy to models/  [RUN_NAME=...] [TRIAL=...]
	@dir="$(_TRIAL_DIR)"; \
	 [ -n "$$dir" ] || { echo "ERROR: no trials found in $(RUNS_DIR)/"; exit 1; }; \
	 test -f "$$dir/best_model.zip" || { echo "ERROR: $$dir/best_model.zip not found — run 'make train EXP=...' first"; exit 1; }; \
	 $(PYTHON) -m scripts.promote "$$dir" --models-root "$(MODELS_DIR)"
```

- [ ] **Step 6: Manual smoke (gated on user — no network access in tests)**

Per the user's memory: real-world verification that needs the wandb dashboard waits for explicit user confirmation. Skip this for now; flagged in Task F1 as a manual smoke item.

- [ ] **Step 7: Commit**

```bash
git add scripts/promote.py tests/unit/test_promote.py Makefile
git commit -m "feat(promote): wandb alias + vendored copy with pinned metadata"
```

---

### Task C5: `resolve_parent` in `eval_team.py` and `eval_ppo.py`

**Files:**
- Modify: `scripts/eval_team.py`
- Modify: `scripts/eval_ppo.py`

- [ ] **Step 1: Find the frozen-spec parser in `eval_team.py`**

```bash
grep -n "frozen:" "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/wandb-integration/scripts/eval_team.py"
```

The `frozen:<path>` parsing happens inside `from_spec()` in `envs/quidditch/opponents.py` (already exists; reuses code paths). What we need is for `<path>` to accept a `wandb://` URI and route through `resolve_parent`.

- [ ] **Step 2: Find the `--model` arg path in `eval_ppo.py`**

```bash
grep -n "args.model\|--model" "/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/wandb-integration/scripts/eval_ppo.py"
```

- [ ] **Step 3: Edit `envs/quidditch/opponents.py:from_spec` frozen branch**

Locate the `if spec.startswith("frozen:")` block in `envs/quidditch/opponents.py`. Wrap the path extraction with `resolve_parent`:

Search for `"frozen:"` and replace the value extraction. Concretely, change:

```python
    if spec.startswith("frozen:"):
        path = spec[len("frozen:"):]
        # ... existing code that uses `path` for PPO.load(...) ...
```

to:

```python
    if spec.startswith("frozen:"):
        raw_path = spec[len("frozen:"):]
        from scripts._artifact_io import resolve_parent
        path = str(resolve_parent(raw_path))
        # ... existing code that uses `path` for PPO.load(...) ...
```

(The exact line numbers depend on the current code. Read the file first.)

- [ ] **Step 4: Edit `scripts/eval_ppo.py` to wrap `--model`**

Locate where `args.model` is passed to `PPO.load`. Wrap it:

```python
from scripts._artifact_io import resolve_parent
# ... existing code ...
model_path = str(resolve_parent(args.model))
model = PPO.load(model_path, ...)
```

- [ ] **Step 5: Write integration test for `eval_team` URI**

Create `tests/unit/test_eval_uri_resolution.py`:

```python
"""eval_team's frozen: spec routes through resolve_parent so wandb:// URIs
are accepted alongside filesystem paths."""
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_frozen_filesystem_path_unchanged(tmp_path: Path) -> None:
    """Plain `frozen:models/foo/best_model` still works."""
    from envs.quidditch.opponents import from_spec
    fake_model = tmp_path / "fake.zip"
    fake_model.write_bytes(b"")

    with patch("envs.quidditch.opponents.PPO") as mock_ppo:
        mock_ppo.load.return_value = MagicMock()
        opp = from_spec(f"frozen:{fake_model}")

    call_args = mock_ppo.load.call_args
    # PPO.load is called with the path (positional or via path=); just confirm
    # the path passed in is the same string.
    args = call_args.args + tuple(call_args.kwargs.values())
    assert any(str(fake_model) in str(a) for a in args)


def test_frozen_wandb_uri_routes_through_resolve_parent(tmp_path: Path) -> None:
    from envs.quidditch.opponents import from_spec

    fake_resolved = tmp_path / "models" / ".cache" / "x_v0" / "best_model.zip"
    fake_resolved.parent.mkdir(parents=True)
    fake_resolved.write_bytes(b"")

    with patch("scripts._artifact_io.resolve_parent", return_value=fake_resolved) as mock_r:
        with patch("envs.quidditch.opponents.PPO") as mock_ppo:
            mock_ppo.load.return_value = MagicMock()
            from_spec("frozen:wandb://x:prod")

    mock_r.assert_called_once_with("wandb://x:prod")
```

- [ ] **Step 6: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_eval_uri_resolution.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add envs/quidditch/opponents.py scripts/eval_ppo.py tests/unit/test_eval_uri_resolution.py
git commit -m "feat(eval): accept wandb:// URIs in frozen: specs and eval_ppo --model"
```

---

### Task C6: `scripts/upload_legacy_models.py` — one-shot migration

**Files:**
- Create: `scripts/upload_legacy_models.py`
- Create: `tests/unit/test_upload_legacy_models.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_upload_legacy_models.py`:

```python
"""upload_legacy_models builds the right wandb.Artifact + writes
_wandb_metadata.json pinning v0."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_legacy_dir(tmp_path: Path, name: str, obs_spec: str = "AUGMENTED_OBS") -> Path:
    d = tmp_path / "models" / name
    d.mkdir(parents=True)
    (d / "best_model.zip").write_bytes(b"model-bytes")
    hydra = d / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text(
        f"run_name: {name}\nobs:\n  name: {obs_spec}\n  n_stack: 3\n"
        f"init:\n  mode: scratch\n  parent: null\n"
    )
    (hydra / "meta.yaml").write_text(
        "git_hash: legacy\nparent_chain_total: 0\n"
        "final_stats:\n  completed_steps: 10000000\n  wall_time_s: 3600\n"
        "  best_eval_reward: null\n  peak_eval_step: null\n"
    )
    return d


def test_upload_creates_artifact_with_prod_and_run_name_aliases(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_red_1", obs_spec="TEAM_ENV_OBS")

    art = MagicMock()
    art.version = "v0"
    run = MagicMock()
    run.log_artifact.return_value = art

    with patch("wandb.Artifact", return_value=art) as mock_art_cls:
        with patch("wandb.init", return_value=run) as mock_init:
            upload_one(d)

    mock_art_cls.assert_called_once()
    name = mock_art_cls.call_args.kwargs.get("name") or mock_art_cls.call_args.args[0]
    assert name == "ppo_hoop_red_1"
    # The aliases list passed to log_artifact must include prod and the run name.
    aliases = run.log_artifact.call_args.kwargs.get("aliases")
    assert aliases is not None
    assert "prod" in aliases
    assert "ppo_hoop_red_1" in aliases


def test_upload_writes_pinned_metadata_in_place(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_blue_4", obs_spec="AUGMENTED_OBS")

    art = MagicMock()
    art.version = "v0"
    run = MagicMock()
    run.log_artifact.return_value = art

    with patch("wandb.Artifact", return_value=art):
        with patch("wandb.init", return_value=run):
            upload_one(d)

    meta = json.loads((d / "_wandb_metadata.json").read_text())
    assert meta["name"] == "ppo_hoop_blue_4"
    assert meta["version"] == "v0"
    assert "prod" in meta["aliases"]


def test_upload_idempotent_skips_if_metadata_present(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_red_1", obs_spec="TEAM_ENV_OBS")
    (d / "_wandb_metadata.json").write_text(json.dumps({"name": "ppo_hoop_red_1", "version": "v0"}))

    with patch("wandb.init") as mock_init:
        skipped = upload_one(d)

    assert skipped is True
    mock_init.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_upload_legacy_models.py -v
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement `scripts/upload_legacy_models.py`**

```python
"""One-shot upload of legacy promoted models to wandb.

For each `models/<run_name>/` (post-Hydra migration: has .hydra/config.yaml
+ best_model.zip), this script:

  1. Reads the run's metadata from .hydra/{config,meta}.yaml.
  2. Translates legacy filesystem-path parent references to wandb URIs.
  3. Builds and logs a wandb artifact with aliases `prod`, `<run_name>`, `v0`.
  4. Writes `_wandb_metadata.json` into models/<run_name>/ pinning v0.

Idempotent: skips dirs that already have _wandb_metadata.json.

Usage:
    python -m scripts.upload_legacy_models                    # all dirs in models/
    python -m scripts.upload_legacy_models models/ppo_hoop_red_1
    python -m scripts.upload_legacy_models --project drone-quidditch ...

Requires a live wandb connection (it's a migration script, run once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb
import yaml


# Map legacy filesystem-path parents to wandb URIs.  Each entry is keyed by
# the literal `init.parent` string found in the legacy config and maps to
# the wandb URI that means the same thing post-migration.
LEGACY_PARENT_REWRITES: dict[str, str] = {
    "models/ppo_hoop_fixed_start_20260504_023051/best_model":
        "wandb://ppo_hoop_fixed_start_20260504_023051:prod",
    "models/ppo_hoop_rand_start_20260505_174509/best_model":
        "wandb://ppo_hoop_rand_start_20260505_174509:prod",
}


def _rewrite_parent_uri(legacy_path: str | None) -> str | None:
    """Convert legacy filesystem `init.parent` to a wandb URI if known."""
    if legacy_path is None:
        return None
    if legacy_path in LEGACY_PARENT_REWRITES:
        return LEGACY_PARENT_REWRITES[legacy_path]
    if legacy_path.startswith(("wandb://", "wandb-artifact://")):
        return legacy_path
    return legacy_path  # leave path as-is for un-rewritten legacy edges


def upload_one(model_dir: Path, project: str = "drone-quidditch") -> bool:
    """Upload one legacy model.  Returns True if skipped (idempotent)."""
    model_dir = Path(model_dir).resolve()
    if (model_dir / "_wandb_metadata.json").exists():
        print(f"[skip] {model_dir.name}: _wandb_metadata.json already present")
        return True

    cfg_path = model_dir / ".hydra" / "config.yaml"
    meta_path = model_dir / ".hydra" / "meta.yaml"
    if not cfg_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"{model_dir} missing .hydra/{{config,meta}}.yaml — "
            f"run scripts/migrate_legacy_models.py first"
        )
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    meta = yaml.safe_load(meta_path.read_text()) or {}

    run_name = cfg.get("run_name", model_dir.name)
    obs = cfg.get("obs", {})
    init = cfg.get("init", {})
    final = meta.get("final_stats", {})

    rewritten_parent = _rewrite_parent_uri(init.get("parent"))

    # Open a wandb run so log_artifact has a producer (the artifact lineage
    # DAG needs each artifact to be `logged_by` *something*).  job_type
    # marks it as a one-shot migration.
    run = wandb.init(
        project=project,
        name=f"upload_legacy:{run_name}",
        id=f"upload_legacy_{run_name}",
        job_type="upload-legacy",
        tags=["upload-legacy", obs.get("name", "?")],
        resume="allow",
        config={"source": "upload_legacy_models", "model_dir": str(model_dir)},
    )

    art = wandb.Artifact(
        name=run_name,
        type="model",
        metadata={
            "obs_spec":           obs.get("name"),
            "n_stack":            int(obs.get("n_stack", 1)),
            "init_mode":          init.get("mode", "scratch"),
            "parent_uri":         rewritten_parent,
            "parent_chain_total": int(meta.get("parent_chain_total", 0)),
            "best_eval_reward":   final.get("best_eval_reward"),
            "git_hash":           meta.get("git_hash", "<legacy>"),
            "legacy_dir":         str(model_dir),
        },
    )
    art.add_file(str(model_dir / "best_model.zip"))
    art.add_dir(str(model_dir / ".hydra"), name=".hydra")

    aliases = ["latest", "prod", run_name, "v0"]
    run.log_artifact(art, aliases=aliases)
    art.wait()

    metadata = {
        "name":     run_name,
        "version":  art.version,
        "entity":   getattr(art, "entity", None),
        "project":  project,
        "aliases":  aliases,
        "logged_by_run_id": run.id,
    }
    (model_dir / "_wandb_metadata.json").write_text(json.dumps(metadata, indent=2))

    wandb.finish()
    print(f"[done] {model_dir.name} → {run_name}:{art.version} (aliases: {aliases})")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Upload legacy promoted models to wandb.")
    p.add_argument("dirs", nargs="*", default=[],
                   help="models/<name>/ dirs to upload (default: all under models/)")
    p.add_argument("--project", default="drone-quidditch")
    args = p.parse_args()

    if args.dirs:
        targets = [Path(d) for d in args.dirs]
    else:
        targets = sorted(
            d for d in Path("models").iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    for d in targets:
        try:
            upload_one(d, project=args.project)
        except Exception as e:
            print(f"[error] {d.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_upload_legacy_models.py -v
```

Expected: 3 cases PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/upload_legacy_models.py tests/unit/test_upload_legacy_models.py
git commit -m "feat(migrate): upload_legacy_models.py one-shot wandb seeder"
```

---

### Task C7: Two-tier `models/` setup — `.gitignore` for `.cache/`

**Files:**
- Create: `models/.gitignore`

- [ ] **Step 1: Write the file**

```bash
mkdir -p "$(git rev-parse --show-toplevel)/models"
cat > "$(git rev-parse --show-toplevel)/models/.gitignore" <<'EOF'
# Two-tier models/:
#   - committed dirs (per-run): vendored prod checkpoints.  Each has
#     best_model.zip + .hydra/ + _wandb_metadata.json (pins v<N>).
#   - .cache/: gitignored landing area for `wandb.use_artifact(...).download()`.
#     Never committed; safe to wipe.
.cache/
EOF
```

- [ ] **Step 2: Verify**

```bash
cd "$(git rev-parse --show-toplevel)" && cat models/.gitignore
```

Expected: prints the three-line file.

- [ ] **Step 3: Commit**

```bash
git add models/.gitignore
git commit -m "chore(models): gitignore .cache/ — wandb artifact download landing zone"
```

---

## Phase D — Lineage walker (dual walk)

### Task D1: Split `scripts/lineage.py` — extract Walker A, add Walker B, CLI dispatch

**Files:**
- Modify: `scripts/lineage.py` (split current code into Walker A + main dispatch; add Walker B)
- Modify: `Makefile:106-109` (`lineage:` target gains `LOCAL=`/`BOTH=` knobs)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_lineage_walkers.py`:

```python
"""Lineage walker: local walks `_wandb_metadata.json`; wandb walks artifact DAG;
make lineage dispatches between them."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Walker A: local-only ─────────────────────────────────────────────────────
def test_walker_a_walks_metadata_chain(tmp_path: Path) -> None:
    from scripts.lineage import walk_chain_local

    # Build a two-link chain: red_v1 → rand_start.
    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "best_model.zip").write_bytes(b"")
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0", "parent_uri": "wandb://rand_start:prod",
    }))
    rand = tmp_path / "models" / "rand_start"
    rand.mkdir()
    (rand / "best_model.zip").write_bytes(b"")
    (rand / "_wandb_metadata.json").write_text(json.dumps({
        "name": "rand_start", "version": "v0", "parent_uri": None,
    }))

    chain = walk_chain_local(red, models_root=tmp_path / "models")

    assert len(chain) == 2
    # Oldest-first ordering.
    assert chain[0]["name"] == "rand_start"
    assert chain[1]["name"] == "red_v1"


def test_walker_a_truncates_on_missing_parent(tmp_path: Path) -> None:
    from scripts.lineage import walk_chain_local

    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0",
        "parent_uri": "wandb://nonexistent_parent:prod",
    }))

    chain = walk_chain_local(red, models_root=tmp_path / "models")

    # Walk truncated at red_v1; warning printed (not asserted here).
    assert len(chain) == 1
    assert chain[0]["name"] == "red_v1"


# ── Walker B: wandb API ──────────────────────────────────────────────────────
def test_walker_b_walks_artifact_dag() -> None:
    from scripts.lineage import walk_chain_wandb

    # Mock a two-link DAG: art_red -[used]-> art_rand.
    art_rand = MagicMock()
    art_rand.name = "rand_start:v0"
    art_rand.version = "v0"
    art_rand.metadata = {"obs_spec": "SIMPLE_ENV_OBS", "parent_chain_total": 20_000_000}
    art_rand.logged_by.return_value = MagicMock(used_artifacts=lambda: [])

    art_red = MagicMock()
    art_red.name = "red_v1:v0"
    art_red.version = "v0"
    art_red.metadata = {"obs_spec": "TEAM_ENV_OBS", "parent_chain_total": 30_000_000}
    red_run = MagicMock()
    red_run.used_artifacts.return_value = [art_rand]
    art_red.logged_by.return_value = red_run

    api = MagicMock()
    api.artifact.return_value = art_red

    with patch("wandb.Api", return_value=api):
        chain = walk_chain_wandb("wandb://red_v1:prod")

    assert len(chain) == 2
    assert chain[0]["name"].startswith("rand_start")
    assert chain[1]["name"].startswith("red_v1")


def test_walker_b_falls_back_to_local_on_api_error(tmp_path: Path) -> None:
    """When wandb.Api() raises, the dispatch falls back to walker A."""
    from scripts.lineage import walk_dispatch

    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0", "parent_uri": None,
    }))

    with patch("wandb.Api", side_effect=RuntimeError("network down")):
        chain = walk_dispatch("wandb://red_v1:prod",
                              models_root=tmp_path / "models",
                              prefer="wandb")

    # Local-fallback chain.
    assert len(chain) == 1
    assert chain[0]["name"] == "red_v1"


# ── Resume collapse ──────────────────────────────────────────────────────────
def test_resume_chain_collapses_in_render(tmp_path: Path) -> None:
    """`init.mode=resume` segments collapse into one line in the rendered output."""
    from scripts.lineage import render

    chain = [
        {"name": "red_v1", "version": "v0", "init_mode": "scratch",
         "steps": 10_000_000, "obs_spec": "TEAM_ENV_OBS"},
        {"name": "red_v1", "version": "v1", "init_mode": "resume",
         "steps": 5_000_000, "obs_spec": "TEAM_ENV_OBS"},
    ]

    out = render(chain)

    assert "resumed" in out.lower()
    # Same name; not two separate red_v1 rows.
    assert out.count("red_v1") <= 2   # name might appear in header + summary
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n uav python -m pytest tests/unit/test_lineage_walkers.py -v
```

Expected: FAIL — `walk_chain_local`, `walk_chain_wandb`, `walk_dispatch` don't exist.

- [ ] **Step 3: Rewrite `scripts/lineage.py` with walker split**

Replace the entire file:

```python
"""Walk a trial's pretrain ancestry — two walkers + CLI dispatch.

Walker A (local-only): reads `_wandb_metadata.json` chains in `models/<run>/`.
  Works offline; truncates on missing parents.

Walker B (wandb API): uses `artifact.logged_by().used_artifacts()` for the
  native artifact DAG.  Richer (sees un-vendored intermediates) but needs
  network + credentials.

CLI dispatch:
    python -m scripts.lineage <target>           # default: B, falls back to A
    python -m scripts.lineage --local <target>   # A only
    python -m scripts.lineage --both <target>    # side-by-side
Targets accepted: filesystem path (runs/.../<trial> or models/<run>),
                  wandb URI (wandb://run:alias or wandb-artifact://...).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def _load_metadata(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_uri_to_local_name(uri: str) -> str | None:
    """`wandb://x:y` → `x`; filesystem path → its parent dir name."""
    if uri.startswith(("wandb://", "wandb-artifact://")):
        # Strip the scheme and any qualifier, then split off the alias.
        body = uri.split("://", 1)[1]
        name_and_alias = body.rsplit("/", 1)[-1]
        return name_and_alias.split(":", 1)[0]
    p = Path(uri)
    # `models/<name>/best_model` or `models/<name>/`
    if "best_model" in p.name:
        return p.parent.name
    return p.name


# ── Walker A: local-only ─────────────────────────────────────────────────────
def walk_chain_local(
    start_dir: Path | str,
    models_root: Path | str = Path("models"),
) -> list[dict[str, Any]]:
    """Walk `_wandb_metadata.json` chains.  Returns oldest-first."""
    models_root = Path(models_root)
    cur = Path(start_dir).resolve()
    chain: list[dict] = []
    visited: set[Path] = set()

    while cur is not None and cur not in visited:
        visited.add(cur)
        meta = _load_metadata(cur / "_wandb_metadata.json")
        if meta is None:
            print(f"WARN: no _wandb_metadata.json at {cur} — chain truncated",
                  file=sys.stderr)
            break

        # Read .hydra for richer info (steps, obs_spec, init_mode).
        hydra_cfg_path = cur / ".hydra" / "config.yaml"
        hydra_meta_path = cur / ".hydra" / "meta.yaml"
        hydra_cfg = yaml.safe_load(hydra_cfg_path.read_text()) if hydra_cfg_path.exists() else {}
        hydra_meta = yaml.safe_load(hydra_meta_path.read_text()) if hydra_meta_path.exists() else {}
        final = hydra_meta.get("final_stats", {})

        chain.append({
            "name":      meta.get("name", cur.name),
            "version":   meta.get("version"),
            "dir":       cur,
            "init_mode": (hydra_cfg.get("init") or {}).get("mode"),
            "obs_spec":  (hydra_cfg.get("obs") or {}).get("name"),
            "steps":     final.get("completed_steps"),
            "parent_chain_total": hydra_meta.get("parent_chain_total"),
        })

        parent_uri = meta.get("parent_uri")
        if not parent_uri:
            break
        parent_name = _resolve_uri_to_local_name(parent_uri)
        if parent_name is None:
            break
        next_dir = (models_root / parent_name).resolve()
        if not next_dir.exists():
            print(f"WARN: parent {parent_uri} not in models/ — chain truncated",
                  file=sys.stderr)
            break
        cur = next_dir

    chain.reverse()
    return chain


# ── Walker B: wandb API ──────────────────────────────────────────────────────
def walk_chain_wandb(uri: str) -> list[dict[str, Any]]:
    """Walk artifact lineage via the wandb API.  Returns oldest-first."""
    import wandb
    api = wandb.Api()

    chain: list[dict] = []
    seen: set[str] = set()
    queue: list[str] = [uri]

    while queue:
        cur_uri = queue.pop(0)
        if cur_uri in seen:
            continue
        seen.add(cur_uri)

        art = api.artifact(cur_uri)
        meta = art.metadata or {}
        run = art.logged_by()
        run_id = getattr(run, "id", None) if run is not None else None
        chain.append({
            "name":      art.name,
            "version":   art.version,
            "init_mode": meta.get("init_mode"),
            "obs_spec":  meta.get("obs_spec"),
            "steps":     meta.get("parent_chain_total"),
            "logged_by": run_id,
            "uri":       cur_uri,
        })

        # Walk parents via run.used_artifacts.
        if run is None:
            continue
        for used in run.used_artifacts():
            used_uri = f"wandb-artifact://{used.entity}/{used.project}/{used.name}"
            queue.append(used_uri)

    # Oldest-first.
    chain.reverse()
    return chain


# ── Dispatch ─────────────────────────────────────────────────────────────────
def walk_dispatch(
    target: str,
    models_root: Path | str = Path("models"),
    prefer: str = "wandb",
) -> list[dict[str, Any]]:
    """Pick a walker.  prefer ∈ {"wandb", "local"}; wandb falls back to local on error."""
    is_uri = target.startswith(("wandb://", "wandb-artifact://"))

    if prefer == "local" or not is_uri:
        if is_uri:
            # Resolve URI → local dir.
            name = _resolve_uri_to_local_name(target)
            start = Path(models_root) / (name or target)
        else:
            start = Path(target)
        return walk_chain_local(start, models_root=models_root)

    try:
        return walk_chain_wandb(target)
    except Exception as e:
        print(f"WARN: wandb walk failed ({e}); falling back to local", file=sys.stderr)
        name = _resolve_uri_to_local_name(target)
        start = Path(models_root) / (name or target)
        return walk_chain_local(start, models_root=models_root)


# ── Render ───────────────────────────────────────────────────────────────────
def render(chain: list[dict]) -> str:
    """Render a chain as an aligned table.  Collapses resume segments."""
    if not chain:
        return "(empty chain)"

    # Collapse: consecutive segments with the same name + init_mode=resume
    # get merged into one row with `(resumed ×N, +M steps)`.
    collapsed: list[dict] = []
    for seg in chain:
        if (
            collapsed
            and seg["name"] == collapsed[-1]["name"]
            and seg.get("init_mode") == "resume"
        ):
            prev = collapsed[-1]
            prev["resume_count"] = prev.get("resume_count", 0) + 1
            prev["resume_steps"] = prev.get("resume_steps", 0) + (seg.get("steps") or 0)
        else:
            collapsed.append(dict(seg))

    rows: list[list[str]] = []
    for seg in collapsed:
        name = seg["name"]
        if seg.get("resume_count"):
            name = f"{name} (resumed ×{seg['resume_count']}, +{seg['resume_steps']:,} steps)"
        steps = seg.get("steps")
        steps_str = f"{steps:,}" if isinstance(steps, int) else "?"
        rows.append([
            name,
            str(seg.get("version", "")),
            str(seg.get("init_mode") or ""),
            str(seg.get("obs_spec") or ""),
            steps_str,
        ])

    headers = ["run", "version", "init", "obs", "steps"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk a run's pretrain ancestry.")
    p.add_argument("target",
                   help="filesystem path (runs/<run>/<trial> or models/<run>) "
                        "or wandb URI (wandb://run:alias)")
    p.add_argument("--local", action="store_true",
                   help="local-only walker (skip wandb API)")
    p.add_argument("--both", action="store_true",
                   help="run both walkers and print side-by-side")
    p.add_argument("--models-root", default="models")
    args = p.parse_args()

    if args.both:
        local_chain = walk_dispatch(args.target,
                                     models_root=args.models_root, prefer="local")
        wandb_chain = walk_dispatch(args.target,
                                     models_root=args.models_root, prefer="wandb")
        print("── local walker ───────────────────────────────────────")
        print(render(local_chain))
        print()
        print("── wandb walker ───────────────────────────────────────")
        print(render(wandb_chain))
        return

    prefer = "local" if args.local else "wandb"
    chain = walk_dispatch(args.target, models_root=args.models_root, prefer=prefer)
    print(render(chain))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update Makefile `lineage:` target**

In `Makefile:106-109`, replace:

```makefile
lineage: ## ⛓  Walk pretrain ancestry of a trial  [RUN_NAME=...] [TRIAL=...]
	@dir="$(_TRIAL_DIR)"; \
	 test -n "$$dir" || { echo "ERROR: no trials found in $(RUNS_DIR)/"; exit 1; }; \
	 $(PYTHON) scripts/lineage.py "$$dir"
```

with:

```makefile
lineage: ## ⛓  Walk pretrain ancestry  [RUN_NAME=...] [TRIAL=...] [TARGET=<path-or-uri>] [LOCAL=1] [BOTH=1]
	@target="$(or $(TARGET),$(_TRIAL_DIR))"; \
	 test -n "$$target" || { echo "ERROR: pass TARGET=<path-or-uri> or RUN_NAME=..."; exit 1; }; \
	 $(PYTHON) -m scripts.lineage "$$target" \
	   $(if $(LOCAL),--local) $(if $(BOTH),--both)
```

- [ ] **Step 5: Run tests**

```bash
conda run -n uav python -m pytest tests/unit/test_lineage_walkers.py -v
```

Expected: 5 cases PASS.

- [ ] **Step 6: Smoke against the existing migrated models (local walker)**

```bash
cd "$(git rev-parse --show-toplevel)" && conda run -n uav python -m scripts.lineage models/ppo_hoop_red_1_20260506_103058 --local
```

Expected: prints a table. The legacy dirs don't have `_wandb_metadata.json` yet — output may say "no _wandb_metadata.json found, chain truncated"; that's fine, the script doesn't crash.

- [ ] **Step 7: Commit**

```bash
git add scripts/lineage.py Makefile tests/unit/test_lineage_walkers.py
git commit -m "feat(lineage): dual local + wandb walkers with offline fallback"
```

---

## Phase E — Sweeps

### Task E1: Sweep YAML + Makefile targets

**Files:**
- Create: `sweeps/blue_v5_lr_ent.yaml`
- Modify: `Makefile` — add `sweep`, `sweep-agent`, `sweep-agents`; update `.PHONY`

- [ ] **Step 1: Create `sweeps/blue_v5_lr_ent.yaml`**

```bash
mkdir -p "$(git rev-parse --show-toplevel)/sweeps"
```

Then create the file:

```yaml
# Sweep: tune lr × ent_coef × batch_size on the blue_v5 experiment.
#
# Wandb's sweep schema, NOT Hydra's — placed under sweeps/ (not conf/) so
# Hydra doesn't try to compose it.  `command:` hands wandb-chosen parameter
# values to scripts.train as Hydra CLI overrides via `${args_no_hyphens}`.
#
# Usage:
#   wandb sweep --project drone-quidditch sweeps/blue_v5_lr_ent.yaml
#   wandb agent <printed-sweep-id>
#
# The agent sets WANDB_SWEEP_ID in the environment, which the run's
# _wandb_init.py sees and uses to set job_type=sweep-train automatically.

program: scripts/train.py
method: bayes
metric:
  name: eval/mean_reward
  goal: maximize

parameters:
  trainer.lr:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
  trainer.ent_coef:
    values: [0.005, 0.01, 0.02]
  trainer.batch_size:
    values: [256, 512, 1024]

# Hyperband early-termination.  min_iter=5 means ≥5 evals (≈1M steps at
# eval_freq_steps=200_000) before a run can be killed — well past the
# first metric.  Comment out the early_terminate block for the first run
# of a new sweep until the metric signal is trusted.
early_terminate:
  type: hyperband
  min_iter: 5
  s: 2

command:
  - ${env}
  - python
  - -m
  - scripts.train
  - +experiment=blue_v5
  - ${args_no_hyphens}
  - hydra.run.dir=runs/${run_name}_sweep/${now:%Y%m%d_%H%M%S}_${oc.env:WANDB_RUN_ID}
```

- [ ] **Step 2: Update Makefile**

In `Makefile:50` `.PHONY` line, add `sweep sweep-agent sweep-agents`:

```makefile
.PHONY: help test test-fast test-warm camera-test demo train resume eval eval-headless lineage promote install clean list-runs eval-team sweep sweep-agent sweep-agents
```

Add at the bottom of the Makefile (after the `eval-team:` block, before `clean:`):

```makefile
# ── Sweeps ───────────────────────────────────────────────────────────────────

WANDB_PROJECT ?= drone-quidditch
SWEEP ?=
ID    ?=
N     ?= 1

sweep: ## 🔁 Create a wandb sweep controller  SWEEP=<name> (file under sweeps/)
	@test -n "$(SWEEP)" || { echo "ERROR: SWEEP=<name> required (see sweeps/)"; exit 1; }
	@test -f "sweeps/$(SWEEP).yaml" || { echo "ERROR: sweeps/$(SWEEP).yaml not found"; exit 1; }
	$(CONDA_RUN) wandb sweep --project $(WANDB_PROJECT) sweeps/$(SWEEP).yaml

sweep-agent: ## 🤖 Run one sweep agent  ID=<sweep_id>
	@test -n "$(ID)" || { echo "ERROR: ID=<sweep_id> required (copy from 'make sweep' output)"; exit 1; }
	$(CONDA_RUN) wandb agent $(ID)

sweep-agents: ## 🤖🤖 Run N parallel sweep agents  ID=<sweep_id> N=<n>
	@test -n "$(ID)" || { echo "ERROR: ID=<sweep_id> required"; exit 1; }
	@echo "Spawning $(N) agents.  Single-machine: N=1 is the sane default for CPU laptop trainings."
	@for i in $$(seq 1 $(N)); do $(CONDA_RUN) wandb agent $(ID) & done; wait
```

- [ ] **Step 3: Lint the sweep YAML offline**

```bash
conda run -n uav python -c "import yaml; yaml.safe_load(open('sweeps/blue_v5_lr_ent.yaml'))"
```

Expected: no exception (YAML is valid).

- [ ] **Step 4: Lint Makefile syntax**

```bash
cd "$(git rev-parse --show-toplevel)" && make -n sweep SWEEP=blue_v5_lr_ent
```

Expected: prints the would-be commands (or "no rules" — we want no parser error). Confirm: `wandb sweep --project drone-quidditch sweeps/blue_v5_lr_ent.yaml` shows up.

- [ ] **Step 5: Commit**

```bash
git add sweeps/blue_v5_lr_ent.yaml Makefile
git commit -m "feat(sweep): wandb sweep YAML + Makefile targets for blue_v5 lr × ent_coef"
```

---

## Phase F — Wrap-up

### Task F1: Full test suite + canary verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pytest suite**

```bash
cd "$(git rev-parse --show-toplevel)" && conda run -n uav python -m pytest tests -x -v
```

Expected: all tests pass. Tests count should be ≥ 60 (was ~46 before Part 2).

- [ ] **Step 2: Verify canary fingerprints byte-identical**

```bash
conda run -n uav python -m pytest tests/integration/test_scoring_canary.py tests/integration/test_team_env_canary.py -v
```

Expected: `step 434 / reward 7.3837` for single-agent canary; team canary unchanged.

- [ ] **Step 3: User confirmation gate for online wandb (manual)**

Per the user's `feedback_verify_before_commit.md` memory: real-world verification that needs the wandb dashboard waits for explicit user confirmation before committing.

This step is a user-driven smoke. Print this checklist:

```
══════════════════════════════════════════════════════════════════════
 MANUAL SMOKE — needs your wandb account + a few minutes
══════════════════════════════════════════════════════════════════════

  1. Set WANDB_ENTITY in your shell (or rely on `wandb login` default).

  2. Online training smoke (~2 minutes):
       make train EXP=canary_team \
         OVERRIDES="trainer.total_timesteps=2048 trainer.n_steps=1024 \
                    eval.eval_freq_steps=999999999 \
                    eval.video.enabled=false"
       → Should print a wandb URL on stdout near the start.
       → Open the URL: run shows name "canary_team_<ts>",
         group "canary_team", tags include "red_0", "TEAM_ENV_OBS",
         "scratch", "beeline_blue".
       → After completion, the run has one artifact "canary_team:v0"
         with alias "latest".

  3. Migrate the 7 legacy models:
       python -m scripts.upload_legacy_models
       → 7 artifacts created with aliases ["latest", "prod",
         "<run_name>", "v0"].
       → Each models/<dir>/_wandb_metadata.json populated.

  4. Promote a real run end-to-end (only if you trained one in step 2):
       make promote RUN_NAME=canary_team
       → Wandb artifact gets `prod` + `canary_team` aliases.
       → models/canary_team/ populated with best_model.zip + .hydra/ +
         _wandb_metadata.json (version: v0).

  5. Sweep dry-run (no actual training; just verifies the YAML parses
     and wandb accepts it):
       make sweep SWEEP=blue_v5_lr_ent
       → Prints a sweep id (e.g. abc123/xyz).  You don't need to start
         the agent; the controller-create is the smoke.

  6. Tell me which of {2,3,4,5} you ran and what you saw.

  When all of those look good, I'll:
    - Update brain/index.md + brain/changelog.md to record Part 2 landing.
    - Commit the legacy-model _wandb_metadata.json files.
    - Merge feature/wandb-integration into develop with --no-ff.
══════════════════════════════════════════════════════════════════════
```

- [ ] **Step 4: After user confirmation, commit the legacy `_wandb_metadata.json` files**

(Only after the user has run `upload_legacy_models.py` and confirmed the wandb side is correct.)

```bash
cd "$(git rev-parse --show-toplevel)" && git add models/*/. _wandb_metadata.json 2>/dev/null
git status
git commit -m "model: seed _wandb_metadata.json for 7 legacy promoted models"
```

(Then the user can decide whether to vendor the `.zip` files via separate per-model commits.)

---

### Task F2: Update `brain/`

**Files:**
- Modify: `brain/index.md` — "Recent Context", "Active Priorities", "Known Issues"
- Modify: `brain/changelog.md` — Part 2 entry
- Modify: `brain/tasks.md` — close Part 2 task, open follow-ups
- Modify: `brain/decisions.md` — ADR for "wandb artifacts canonical, two-tier models/"

Per the user's CLAUDE.md: `brain/` lives outside the git repo. Edits are made directly to the umbrella-level `brain/` files, not inside `repo/`.

- [ ] **Step 1: Append a Part 2 entry to `brain/changelog.md`**

(Detailed body left to the implementer; mirror the Hydra Part 1 entry's structure — header line, structural changes, tests delta, canary status, plan/spec doc paths.)

- [ ] **Step 2: Update `brain/index.md`**

- "Current State" — add a sentence-paragraph for "W&B integration landed YYYY-MM-DD".
- "Active Priorities" — strike the Part 2 brainstorm line; add the next priority (Part 3 candidate: eval framework + per-term reward curves).
- "Known Issues" — add: "`scripts/lineage.py` legacy dirs without `_wandb_metadata.json` truncate the chain — fix is to run `upload_legacy_models.py` before walking."
- "Recent Context" — promote the Part 2 entry to the top.

- [ ] **Step 3: Add ADR to `brain/decisions.md`**

Topic: "W&B artifacts as canonical registry, two-tier `models/` mirror." Decision shape per the project's ADR convention: context, decision, alternatives, consequences. Cite the spec doc path.

- [ ] **Step 4: Tasks bookkeeping**

In `brain/tasks.md`:
- Mark the "Brainstorm Part 2 of the ML infra redesign — W&B integration" task as completed.
- Add a follow-up: "Per-term reward curves: extend `RewardStack.compute_step` to expose `last_per_term`; forward into env info[]."
- Add a follow-up: "`scripts/eval_team.py` Hydra migration (deferred from Part 2)."

- [ ] **Step 5: No commit on `brain/`**

`brain/` lives outside the git repo — nothing to git-commit. Just save the edits.

---

## Self-Review

Looked back at the spec section by section after writing the plan:

**Coverage:**
- §1 (Three subsystems) → Tasks B1–B4, C1–C7, D1, E1
- §2 (TB retired) → B2, B3, B4
- §3 (Run identity) → B1
- §4 (Charts) → B2 (deferred per-term reward curves explicit)
- §5 (Videos) → B3
- §6 (Artifacts auto-log) → C2
- §7 (Two-tier `models/`) → C1, C7
- §8 (Lineage walker) → D1
- §9 (Sweeps) → E1
- §10 (Migration) → C6, F1
- §11 (Tests) → A2 (conftest), and each task has TDD tests in-line
- §12 (Conf / schema) → A3, A4
- §13 (File footprint) → cross-referenced

**Placeholder scan:** Found one (`sys.platform_provides_no_network_dummy_flag` walrus operator) in Task D1 Step 3 — fixed inline with the explicit follow-up note.

**Type consistency:** `resolve_parent` signature `(uri_or_path, models_root) → Path` is consistent across Tasks C1, C3, C5. `promote_run_dir` signature consistent in Tasks C4 and tests. `init_wandb` consistent in B1, B2.

**Scope:** Single sub-project (W&B integration). The four task families (charts, registry, lineage, sweeps) are tightly interlinked; splitting would force shipping a half-functional state. Plan stays as one.

**One uncertainty I called out in the prose, not the tasks:** Task C5 step 3 instructs the implementer to find the `frozen:` branch in `envs/quidditch/opponents.py:from_spec`. I didn't read that file during plan-writing — the implementer should `grep -n "frozen:" envs/quidditch/opponents.py` first and adapt the line numbers. This is the kind of "the surface area exists but the exact line moves" detail that's safer to verify than to pin.
