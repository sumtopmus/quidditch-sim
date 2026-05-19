# ML Infra Redesign — Part 3: Eval Framework + Checkpoint UX + TUI Controller

**Date:** 2026-05-18
**Branch:** `feature/ml-infra-part-3` (worktree off develop)
**Closes:** the open Part 3 thread on the ML infra redesign (Part 1 = Hydra, Part 2 = W&B).

## Context

Parts 1 and 2 of the ML infra redesign landed:

- **Part 1 (Hydra, 2026-05-13 → 14):** retired the TOML + argparse setup; `conf/` tree with 9 config groups, structured dataclass schemas, single `scripts/train.py` entrypoint, `make train EXP=<name>`.
- **Part 2 (W&B, 2026-05-14 → 15):** retired TB; `scripts/_artifact_io.py` + `scripts/_wandb_init.py`; vendored + cached two-tier `models/` with `_wandb_metadata.json` pinning immutable versions; wandb-native sweeps bridged to Hydra; dual local/wandb lineage walkers.
- **Subsequent follow-ups (2026-05-15 → 18):** obs-spec renames (`DUEL_V1_BODY`, `DUEL_V2_WORLD`), reward magnitudes unified onto YAML, `feature/model-doc-generator` (per-run `MODEL.md` with header/summary/lineage/obs/reward/env/hyperparams/eval/wandb sections, auto-rendered at end of training, copied by `promote.py`, backfilled for 7 legacy models), `feature/blue-v7-body-ego` (`DUEL_V3_BODY_EGO` obs spec; `team_v3_intercept` reward stack with `InterceptShaping`; env_factory threads `learner_id`+`learner_spec` into team_env; OCE becomes a pure pass-through; per-agent obs builder dispatches on the agent's persisted spec).

Three open threads remain:

1. **Eval framework.** [scripts/eval_team.py](scripts/eval_team.py) is still argparse-driven (carried as out-of-Part-1 scope). There's no automated way to run a candidate through a fixed battery of scenarios and produce a single-glance report. Diagnosing model quality requires composing `make eval-team LEARNER=... LEARNER_FRAME_STACK=... BLUE=... RED=... GUI=... CRASH_AFTERMATH_SECONDS=...` from memory and reading scrollback. The 2026-05-13 "silent broken eval_team" episode (75-d frozen checkpoints fed 22-d obs for days without anyone noticing) is the cautionary tale.
2. **Checkpoint discovery + obs-spec preflight.** Inventory of promoted models, their obs specs, parents, and chain totals is buried in [brain/models.md](../../brain/models.md) (not in the repo) or requires manual `make lineage` walks. Obs-spec compatibility surfaces only mid-training via `check_obs_compat`'s strict raise — there's no way to ask "will this parent + this child experiment YAML load cleanly?" before `make train`.
3. **TUI controller.** `feature/tui-launcher` (~30 commits, idle since 2026-05-12) was a full Textual launcher with two-slot subprocess manager, action tree, dynamic form, training widget with sparkline, log overlay. With W&B owning live metrics and run comparisons, most of its original value prop is now covered elsewhere — but the *experiment picker* + *eval composer* + *inventory-driven dropdowns* pattern remains unsolved.

## Goals

- Single, automated eval battery: one command runs a candidate through a fixed scenario set and emits a Markdown + CSV + JSONL report locally. No more ad-hoc `make eval-team` invocations to assess a new ladder rung.
- Read-only inspection commands: `dsim inventory`, `dsim obs-preflight`, `dsim lineage`, `dsim list-runs` make state visible without grepping the brain or walking checkpoint dirs.
- Interactive eval composition: dispatching a one-off `eval_team` matchup with GUI from menus (using inventory data to populate learner / opponent dropdowns) instead of typing make-target arg lists from memory.
- TUI as a thin column-browser front-end (Task → Items → Details), one binary the TUI dispatches via subprocess, no SB3 callbacks owning the UX.
- Migrate `scripts/eval_team.py` and `scripts/eval_ppo.py` to Hydra so interactive eval, the battery, and Match-from-TUI all share one config surface.
- Reshape the Makefile: chores only (install, test, tui, **train**, demo-like things move into the TUI).

## Non-goals

- Head-to-head pairwise tournaments / Elo ranking (deferred — battery handles 1-vs-N for a candidate; N-vs-N is a Part-3-follow-up if ever).
- W&B Reports / Tables for battery results (local-only output keeps it offline-survivable; users can paste numbers into a W&B Report manually).
- An in-TUI experiment-YAML editor that writes new YAMLs (the `[+ new from <X>]` rung-from-parent generator is explicitly deferred).
- A live training metrics widget (W&B owns this; no `TUIProgressCallback` revival).
- Live in-TUI log scrollback (focus a subprocess slot and the bottom strip shows the last 6 lines; full scrollback is `tail -f` in a separate terminal).
- Mass-rebase of `feature/tui-launcher` onto current develop. Fresh `feature/ml-infra-part-3` worktree, cherry-pick the still-relevant pieces only.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Slice 2: TUI controller  (Textual, three-pane column browser)         │
│  ┌────────────┐ ┌─────────────────────┐ ┌──────────────────────────┐   │
│  │  Task      │ │  Items              │ │  Details                 │   │
│  │  (10)      │ │  (task-driven)      │ │  (task-driven; Match     │   │
│  │            │ │                     │ │   uses 3 sub-tabs)       │   │
│  └────────────┘ └─────────────────────┘ └──────────────────────────┘   │
│  ┌──────────────────────────┐ ┌──────────────────────────────────────┐ │
│  │  Slot 1 (train-ish)      │ │  Slot 2 (eval-ish)                   │ │
│  └──────────────────────────┘ └──────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                              ↓ subprocess
┌────────────────────────────────────────────────────────────────────────┐
│  Slice 1: commands (Typer for inspection/dispatch, Hydra for runs)     │
│                                                                        │
│  Hydra entrypoints:                                                    │
│    python -m scripts.train         +experiment=<name>                  │
│    python -m scripts.eval_team     +eval_team=default learner=… opponent=…│
│    python -m scripts.eval_ppo      +eval_ppo=default                   │
│    python -m scripts.eval_battery  +eval_battery=<preset> candidate=<uri>│
│                                                                        │
│  Typer CLI (single binary `dsim`):                                     │
│    dsim inventory            [--json] [--include-cache]                │
│    dsim obs-preflight        --parent X --child-obs Y                  │
│    dsim lineage              --target <uri-or-path> [--local|--both]   │
│    dsim list-runs            [--run NAME]                              │
│    dsim resume               <run-name> [--trial T] [--ckpt C]         │
│    dsim promote              <run-name> [--alias prod]                 │
│    dsim describe-run         <run-name> [--trial T]   # wraps render_model_doc│
│    dsim sweep create|agent|agents <args…>                              │
│    dsim tui                  # launches the Textual app                │
│                                                                        │
│  Shared core (importable, no CLI):                                     │
│    core/inventory.py     → list[ModelInfo]                             │
│    core/obs_compat.py    → preflight(parent, child) → PreflightReport  │
│    core/eval_core.py     → run_scenario(learner, scenario) → ScenarioResult│
│    core/eval_report.py   → write(scenario_results, output_dir)         │
└────────────────────────────────────────────────────────────────────────┘
                              ↓ reads
┌────────────────────────────────────────────────────────────────────────┐
│  Data sources (existing, unchanged)                                    │
│    models/<name>/.hydra/{config,meta}.yaml                             │
│    models/<name>/_wandb_metadata.json   models/<name>/MODEL.md         │
│    runs/<name>/<ts>/.hydra/             conf/experiment/*.yaml         │
│    W&B API (when online — offline-survivable fallback exists)          │
└────────────────────────────────────────────────────────────────────────┘
```

**Core design principles:**

1. **`core/` holds importable Python functions; `scripts/` and `dsim/` are thin CLI wrappers.** No business logic in CLI files. The TUI imports `core/` for read-only data and shells out for actions; it never imports `scripts/`.
2. **Two CLI patterns, by fit.** Hydra apps for composable configs (training, eval, sweeps); Typer for inspection / dispatch / one-shot ops (inventory, lineage, promote, describe-run, resume).
3. **Inventory + obs-preflight are pure read-only services.** Used by both the CLI and the TUI. No side effects.
4. **Battery dispatch is in-process.** `eval_battery.py` imports `eval_core` and runs scenarios sequentially in one Python process; one `wandb.init` per battery (tagged `mode=eval_battery`, no model artifact logged).
5. **TUI dispatches training and eval as subprocesses, not in-process.** Two slots in a bottom strip; Ctrl-C on the TUI doesn't kill subprocesses (they're set as group leaders).

## Slice 1: commands + engine

### 1.1 `core/inventory.py` — read-only model catalog

Lifts the `_load_run_context` helper at scripts/_render_model_doc.py:21 into a public API:

```python
@dataclass(frozen=True)
class ModelInfo:
    name: str                  # e.g. "ppo_hoop_blue_4_20260511_202612"
    short_name: str            # e.g. "blue_4"
    obs_spec: str              # e.g. "DUEL_V2_WORLD"
    n_stack: int
    parent: str | None         # init.parent URI or path; None for scratch
    parent_chain_total: int
    final_steps: int | None
    source: Literal["vendored", "cache"]
    path: Path                 # absolute, points at the model dir
    wandb_alias: str | None    # "prod" / "<run_name>" / None
    wandb_version: str | None  # "v<N>" pinned in _wandb_metadata.json
    has_model_doc: bool        # True if MODEL.md exists on disk

def inventory(
    models_dir: Path = MODELS_DIR,
    include_cache: bool = False,
) -> list[ModelInfo]: ...

def load_model_doc(info: ModelInfo) -> str | None: ...   # read MODEL.md
def load_run_context(info: ModelInfo) -> dict[str, Any]: ...  # full context dict
```

- Walks `models/*/` (committed-vendored) and, if `include_cache=True`, `models/.cache/*/`.
- For each entry, calls `scripts._render_model_doc._load_run_context(model_dir)` (which already handles `.hydra/config.yaml`, `meta.yaml`, `_wandb_metadata.json`, and the legacy `run_info.toml`+`config.toml` fallback per the 2026-05-13 migration).
- No W&B network calls; offline-survivable (matches Walker A semantics from lineage).
- Sorted by `short_name`, then timestamp suffix descending.

To keep the import boundary clean, `_load_run_context` moves from `scripts/_render_model_doc.py` into `core/run_context.py` and `scripts/_render_model_doc.py` re-imports it. That's a refactor inside Slice 1 — the public function signature and behavior do not change. (Already listed in 1.9 touchpoints.)

### 1.2 `core/obs_compat.py` — preflight without loading the model

```python
@dataclass(frozen=True)
class ObsBlockDiff:
    block: str
    dim: int
    parent_frame: str | None
    child_frame: str | None
    status: Literal["matched", "frame_changed", "removed", "added"]

@dataclass(frozen=True)
class PreflightReport:
    compatible: bool             # False ⇒ check_obs_compat would strict-raise
    surgery_required: bool       # True ⇒ needs init.mode=warm_start
    diff: list[ObsBlockDiff]     # column-by-column
    parent_spec_name: str
    child_spec_name: str
    parent_n_stack: int
    child_n_stack: int

def preflight(parent_uri: str, child_obs_name: str, child_n_stack: int = 1) -> PreflightReport: ...
```

- Resolves `parent_uri` to its `obs.name` + `n_stack` **without loading the model**:
  - For a local path: reads `<parent_dir>/.hydra/config.yaml` directly.
  - For a `wandb://` URI: downloads only `.hydra/` from the artifact (use `wandb.Api().artifact(...).file(".hydra/config.yaml").download()`), not the weights — fast enough to run interactively.
  - Falls back to reading `MODEL.md`'s obs-spec section if `.hydra/config.yaml` is unavailable (legacy migrated models).
- Looks up both `parent_spec_name` and `child_obs_name` in `SPEC_BY_NAME` (which now includes `DUEL_V3_BODY_EGO`), runs the existing `check_obs_compat` logic in dry-run mode, returns the diff.
- Used by `dsim obs-preflight`, the TUI's experiment-picker badge, and as a pre-load guard rail in `scripts/train.py`'s `init.mode == pretrain` path so users see the diff *before* the strict raise.

### 1.3 `core/eval_core.py` — extracted run loop

Lifts the per-episode loop, terminal-cause bucket counters, take-down counter, and deterministic-flag plumbing from [scripts/eval_team.py](scripts/eval_team.py) into pure functions.

```python
@dataclass(frozen=True)
class ScenarioSpec:
    opponent: str                  # e.g. "beeline_red", "intercepter_red:lookahead=0.5", "frozen"
    opponent_model_path: str | None  # required when opponent=="frozen"
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
    terminal_cause: str            # drone_drone_crash, red_floor, blue_floor, red_wall, blue_wall, red_oob, blue_oob, timeout, score
    take_down_fired: bool
    score_at_episode_end: int | None

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
    render: bool = False,            # True for GUI (launch_passive)
    progress_cb: Callable[[int, int], None] | None = None,
) -> ScenarioResult: ...
```

The `learner_id` + `learner_spec` env_factory plumbing landed in `feature/blue-v7-body-ego` (de34230 + 797ed7c) makes this clean: `run_scenario` resolves the learner's obs spec from its `.hydra/config.yaml`, sets `cfg.env.learner_id` and `cfg.env.learner_spec` on the team env, and the team env's per-agent obs builder handles the rest. OCE is a pure pass-through (per aa34387), so eval_core doesn't need to know about it.

### 1.4 `scripts/eval_team.py` — Hydra migration

Adopts the `scripts/train.py` pattern. New config groups:

- **`conf/eval/default.yaml`** — shared eval params: `n_episodes`, `deterministic`, `crash_aftermath_seconds`, `gui`, `randomise_start`, `seed`.
- **`conf/learner/{blue,red}.yaml`** — which side is the learner; pulls `learner_id`, `learner_spec` from the learner's `.hydra/config.yaml` (or override via CLI).
- **`conf/eval_team/default.yaml`** — top-level config composing the above + `opponent`.

Entrypoint:

```
python -m scripts.eval_team +eval_team=default \
    learner=blue learner.uri=wandb://ppo_hoop_blue_4:prod \
    opponent=beeline_red \
    eval.gui=true eval.crash_aftermath_seconds=3.0
```

Argparse surface is removed; the `--learner`, `--learner-frame-stack`, `--blue`, `--red`, `--gui`, `--crash-aftermath-seconds`, `--episodes`, `--deterministic`, `--randomise-start` flags all become Hydra config / overrides.

### 1.5 `scripts/eval_ppo.py` — Hydra migration

Same treatment for the single-agent eval entrypoint. `conf/eval_ppo/default.yaml` mirrors the single-agent training surface. Becomes part of Slice 1 because the Match TUI task dispatches against both kinds of learners (team and single-agent).

### 1.6 `scripts/eval_battery.py` + `conf/eval_battery/` — new Hydra entrypoint

```yaml
# conf/eval_battery/default.yaml
defaults:
  - _self_
candidate: ???              # required: wandb://… or models/… or runs/…
output_dir: null            # default = ${hydra.run.dir}/eval_report
seed: 0
scenarios:
  - opponent: beeline_red
    randomise_start: false
    n_episodes: 10
  - opponent: beeline_red
    randomise_start: true
    n_episodes: 10
  - opponent: intercepter_red:lookahead=0.5
    randomise_start: true
    n_episodes: 10
  - opponent: zero_red
    randomise_start: false
    n_episodes: 5
  - opponent: frozen
    opponent_model_path: models/ppo_hoop_red_1_20260506_103058/best_model
    randomise_start: true
    n_episodes: 10
```

Additional presets:

- **`conf/eval_battery/quick.yaml`** — 3 scenarios × 3 episodes (smoke check).
- **`conf/eval_battery/ladder.yaml`** — vs each frozen ladder rung × random start.
- **`conf/eval_battery/scripted_only.yaml`** — vs each scripted opponent, both start modes.

Each scenario gets a `ScenarioResult` from `run_scenario`. All scenarios run sequentially in one process under one `wandb.init` (tag `mode=eval_battery`); no model artifact is logged from a battery — it only records scenario metrics for searchability.

The candidate's reward stack and obs spec are read from the candidate's own `.hydra/config.yaml` (resolved via `core/inventory.load_run_context`); the battery YAML does not specify them.

### 1.7 `core/eval_report.py` — report writer

Given a `list[ScenarioResult]`, writes into `<output_dir>/`:

- **summary.md** — Markdown report:
  - Header: candidate metadata (run name, obs_spec × n_stack, parent_chain_total, source vendored/cache, wandb alias).
  - One table row per scenario: `opponent | start | n_eps | win% | mean_R_learner | mean_R_opp | take_down% | OOB% | mean_ep_len`.
  - Per-scenario detail subsections with terminal-cause stacked counts.
  - Optional reuse of `_render_model_doc._section_header` to keep formatting consistent with `MODEL.md`.
- **results.csv** — flat one-row-per-scenario table for grepping / diffing across batteries.
- **per_episode.jsonl** — one JSON object per episode (`scenario_id`, `ep_idx`, `length`, `reward_learner`, `reward_opponent`, `terminal_cause`, `take_down_fired`, `score_at_episode_end`) for ad-hoc re-aggregation.

Default `<output_dir>` = the Hydra `hydra.run.dir` of the battery (`runs/eval_battery_<candidate_short>/<ts>/eval_report/`).

### 1.8 `dsim` Typer CLI (new top-level package)

```
dsim/
├── __init__.py
├── __main__.py            # entrypoint
├── cli.py                 # Typer app
└── commands/
    ├── inventory.py       # dsim inventory [--json] [--include-cache]
    ├── obs_preflight.py   # dsim obs-preflight --parent X --child-obs Y [--child-n-stack N]
    ├── lineage.py         # dsim lineage --target X [--local|--both]
    ├── list_runs.py       # dsim list-runs [--run NAME]
    ├── resume.py          # dsim resume <run-name> [--trial T] [--ckpt C]
    ├── promote.py         # dsim promote <run-name> [--alias prod]
    ├── describe_run.py    # dsim describe-run <run-name> [--trial T]
    ├── obs_specs.py       # dsim obs-specs   (pretty-prints SPEC_BY_NAME)
    ├── sweep.py           # dsim sweep create / agent / agents
    └── tui.py             # dsim tui  → launches the Textual app
```

- Each command ≤50 lines: parse args → call `core/...` function or shell out to existing Hydra entrypoint → render result (rich.Table for inventory/lineage, plain text for the rest).
- `pyproject.toml` gains `[project.scripts] dsim = "dsim.cli:app"` so `pip install -e .` exposes `dsim` directly. `python -m dsim` works as a fallback.
- New requirements: `typer` (transitively pulls `rich`, which we import directly for table rendering).

### 1.9 Existing-code touchpoints

| Path | Change |
|---|---|
| `scripts/_render_model_doc.py` | `_load_run_context` moves to `core/run_context.py`; script re-imports it. `render_model_doc(run_dir)` unchanged. |
| `scripts/lineage.py` | Refactored to export `walk_chain_local(path)` and `walk_chain_wandb(uri)` as importable functions; CLI dispatch moves to `dsim/commands/lineage.py`. |
| `scripts/promote.py` | Splits into `core/promote.py:promote_run(run_name, alias)` (the logic) + `dsim/commands/promote.py` (the CLI). |
| `scripts/list_runs.py` | Re-implemented on develop (the version on `feature/tui-launcher` is stranded) as `core/run_listing.py:list_runs(filter)` + `dsim/commands/list_runs.py`. |
| `scripts/_train_common.py` | `resolve_parent()` grows a `metadata_only: bool = False` kwarg so `obs_preflight` can fetch a parent's `.hydra/` without the weights. |
| `scripts/eval_team.py` | Hydra migration (see 1.4). |
| `scripts/eval_ppo.py` | Hydra migration (see 1.5). |
| `scripts/callbacks.py` | No change; noted as a touchpoint because it's where 04f5842 + e43eed7 + cbb8cc0 + f629c3c landed recent fixes. Tests in `tests/scripts/test_video_callback_wandb.py` continue to apply. |
| `Makefile` | See Section "CLI / Makefile reshape" below. |

## Slice 2: TUI controller (three-pane column browser)

### 2.1 Branch strategy

Slice 2 lands on the same `feature/ml-infra-part-3` worktree, after Slice 1 (so the TUI is built against the engine it dispatches). Cherry-pick from `feature/tui-launcher` rather than rebase — the old branch is 30 commits across an obsolete `_train_common`, an obsolete Makefile rewrite, three scripts (`repro`, `list_runs`, `promote`) that now overlap with `dsim/commands/`, plus a `TUIProgressCallback` whose value prop W&B has eaten.

| File on `feature/tui-launcher` | Action | Why |
|---|---|---|
| `tui/glyphs.py` + CSS theme | Cherry-pick as-is | Violet accent, rounded panels — pure visual, no coupling |
| `tui/subprocess.py` (two-slot manager) | Cherry-pick, retarget | Now spawns `python -m scripts.X` / `dsim X` / `mjpython demo/menu.py` instead of `make` |
| `FormInput` + `FormSelect` (from `ab579a5`) | Cherry-pick | The two-mode vim-style focus + Select ↑/↓ override are load-bearing for nav (see 2.2) |
| `ActionForm._rebuild` async pattern (from `2685750`) | Cherry-pick the pattern | `await body.remove_children()` + bundled `await body.mount(*children)` |
| `**kwargs`-forwarding `__init__` (from `8c933a7`) | Cherry-pick the pattern | Applied to every new Pane widget |
| `tui/action_tree.py` | Re-implement | Action set is 10 tasks, not the old 16-action registry |
| `tui/state.py` (disk scanner) | Drop / replace | Replaced by `core.inventory.inventory()` |
| `tui/training_widget.py` (sparkline) | Drop | W&B owns live metrics |
| `tui/log_overlay.py` (log modal) | Drop | Focus a subprocess slot + `[Enter]` for fullscreen tail (~50 lines, future enhancement) |
| `tui/actions.py` (16-action registry) | Drop | Subsumed by the per-task argv builders in 2.4 |
| `TUIProgressCallback` + status-file plumbing | Drop | No new SB3 callbacks; subprocess pane tails stdout directly |

### 2.2 Nine-rule navigation discipline

Derived from the three nav-related commits on `feature/tui-launcher` (`ab579a5`, `2685750`, `8c933a7`) — each rule is a bug class that bit the old branch.

1. **`can_focus = True` on the focusable container; focus it on mount.** Each of the three panes is its own focusable container. Pane 1 (Task) gets focused in `on_mount`.
2. **Forward `**kwargs` to `super().__init__()` in every widget subclass.** Textual's `Widget.__init__` accepts `id`/`classes`/`name`/`disabled`. Locked in by a unit test that instantiates every widget with `id=...`, `classes=...` at import time.
3. **Never override `_render`.** Textual's internal method must return a Visual. Use `_rebuild`/`_refresh_pane` for "tear down + re-mount children" semantics. Locked in by a headless mount-and-render smoke test.
4. **`remove_children()` is async; `await` it before mounting new children.** Two Pane-2 items that share a sub-widget id (e.g. both Train and Resume items have a `field-RUN_NAME`) would raise `DuplicateIds` otherwise. Every rebuild path is `async`; chain Pane1.on_select → Pane2.set_task → Pane2._rebuild end-to-end. Locked in by a walk test with overlapping ids.
5. **Two-mode focus on every text Input (vim-style).** Land on field → **nav mode** (cursor hidden via `.nav-mode .input--cursor { background: transparent; ... }`, characters swallowed, arrows bubble). Enter → edit mode. Esc / blur → back to nav. Implemented as `FormInput(Input)` with `edit_mode: reactive[bool]`.
6. **Subclass `Select` to free up ↑/↓.** Textual's `Select` BINDINGS are `"enter,down,space,up"` → `show_overlay`. Subclass BINDINGS *accumulate* through MRO. Fix: explicit `Binding("down", "app.focus_next")` + `Binding("up", "app.focus_previous")` in `FormSelect(Select)` — per-key match wins over comma-string match.
7. **Esc as the global "back" key.** App-level `Binding("escape", "focus_root_pane")` returns focus to Pane 1. Forward bindings (Enter / → in each pane) are on the individual pane widgets so they only fire when that pane has focus. ← from the leftmost focusable widget in a row pops back one pane.
8. **`.panel:focus-within` for active-pane border highlight.** Same hue as the rest of the palette at full saturation. Plus a `.tree-cursor` row class (subtle background tint + bold) for unambiguous selection visibility even when focus leaves the pane.
9. **Smoke tests at the boundaries that bit us.** Four headless tests pin the rules: `test_app_smoke.py` (mount + one frame), `test_widget_kwargs.py` (every widget accepts id/classes), `test_pane_navigation.py` (a pilot.press walk across all three panes + Esc + back), `test_overlapping_ids.py` (walk between two Pane-2 items whose Pane-3 forms share field ids).

### 2.3 Three-pane layout

```
┌─ Task ──────────┐ ┌─ Items ──────────────────────┐ ┌─ Details ──────────────────────────┐
│   Demo          │ │ <items for selected task>    │ │ <details for selected item>        │
│   Camera Test   │ │                              │ │                                    │
│ ▶ Train         │ │                              │ │ (Match-only): three sub-tabs       │
│   Match  (1v1)  │ │                              │ │   [Learner] [Opponent] [Options]   │
│   Battery       │ │                              │ │   ─── resolved command ───         │
│   Resume        │ │                              │ │   python -m scripts.eval_team …    │
│   Sweep         │ │                              │ │                                    │
│   Inventory     │ │                              │ │                                    │
│   Lineage       │ │                              │ │ [Enter] launch  [e] overrides      │
│   Promote       │ │                              │ │                                    │
└─────────────────┘ └──────────────────────────────┘ └────────────────────────────────────┘
┌─ Slot 1 (train-ish) ────────────────────────┐ ┌─ Slot 2 (eval-ish) ────────────────────────┐
│ idle                                        │ │ idle                                       │
└─────────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

### 2.4 Per-task dispatch table

| # | Task | Pane 2 items | Pane 3 details | Subprocess dispatch | Python binary | Slot |
|---|---|---|---|---|---|---|
| 1 | **Demo** | `hover`, `waypoint`, `scenarios` from `demo/menu.py` registry | Description + viewer keymap reminder | `mjpython demo/menu.py <key>` | mjpython | 1 |
| 2 | **Camera Test** | `grid`, `fixed`, `north`, `east`, `south`, `west`, `top`, `fpv`, `tpv`, `port`, `starboard` | Cam excerpt from `conf/camera/default.yaml` + output mp4 path | `python demo/camera_test.py --cam <name>` | python | 1 |
| 3 | **Train** | `conf/experiment/*.yaml` + obs-preflight badge (`✓`/`⚠`/`❌`/`–`) | Composed Hydra config + lineage strip + preflight diff | `python -m scripts.train +experiment=<name>` | python | 1 |
| 4 | **Match** | Inventory (learner candidates) | Three sub-tabs (see 2.5) + resolved command | `mjpython -m scripts.eval_team …` if `eval.gui=true`, else python | mjpython if GUI else python | 2 |
| 5 | **Battery** | Inventory (candidates) | Battery preset picker + scenario count + expected runtime + `[o]` open last report | `python -m scripts.eval_battery +eval_battery=<preset> candidate=<uri>` | python | 2 |
| 6 | **Resume** | Recent runs (latest trial per run name, ts desc) | Latest checkpoint, parent_chain_total at checkpoint | `dsim resume <run-name>` | python | 1 |
| 7 | **Sweep** | `sweeps/*.yaml` | Sweep controller config preview + agent count slider | `dsim sweep create <name>` / `dsim sweep agents <id> --n N` | python | 1 |
| 8 | **Inventory** | Inventory rows | Renders `MODEL.md` (or `core.inventory.load_model_doc(info)` fallback for cache entries without docs) | none (read-only) | — | — |
| 9 | **Lineage** | Inventory rows (target) | ASCII tree walking back via parent chain (calls `walk_chain_local` or `walk_chain_wandb` based on URI form) | none (read-only) | — | — |
| 10 | **Promote** | Recent unpromoted runs (filter: has `best_model.zip` or `final_model.zip`) | Promote dry-run preview: target dir name, alias, files to copy | `dsim promote <run-name>` | python | 1 |

Inventory and Lineage are read-only — the user picks an item in Pane 2 and the details render in Pane 3 with no Enter required. The other tasks have an `[Enter] launch` footer that dispatches the subprocess.

### 2.5 Match Pane 3 sub-tabs

Three sub-tabs in Pane 3, navigable via `[` / `]`:

- **Learner**: radio list of inventory entries filtered to those with a loadable `best_model.zip`/`final_model.zip`. Row shows `obs_spec`, `n_stack`, source.
- **Opponent**: radio list with two sections — scripted (`zero_red/blue`, `beeline_red/blue`, `intercepter_red:lookahead=0.5`, `intercepter_blue:lookahead=0.5`) and frozen (inventory entries; future-proofs against new opp specs).
- **Options**: toggles for `eval.gui`, `eval.randomise_start`, `eval.deterministic`, `eval.crash_aftermath_seconds` (slider 0.0 → 5.0), `eval.n_episodes` (int).

Below the tabs, a live-updating resolved command line:

```
python -m scripts.eval_team +eval_team=default \
    learner=blue learner.uri=wandb://ppo_hoop_blue_4:prod \
    opponent=beeline_red \
    eval.gui=true eval.crash_aftermath_seconds=3.0 eval.n_episodes=5
```

`[Enter]` dispatches into Slot 2.

### 2.6 Entrypoint

- **Make**: `make tui` → `python -m dsim tui` (the TUI itself doesn't need mjpython — only its subprocess slots do).
- **Direct**: `dsim tui` (if `pip install -e .` was run) or `python -m dsim tui`.

## CLI / Makefile reshape

### Final Makefile — 8 targets

```makefile
.PHONY: help install clean test test-fast test-warm tui train

.DEFAULT_GOAL := help

help:        ## Show targets + pointers to dsim and Hydra entrypoints
install:     ## Create conda env + install dsim editable
clean:       ## Remove build artifacts and __pycache__
test:        ## Full test suite
test-fast:   ## Unit tests only (skip slow integration canaries)
test-warm:   ## Warm-start integration test  MODEL=<run-name>
tui:         ## Open the controller TUI
	@$(PYTHON) -m dsim tui
train:       ## Launch a single training run  EXP=<name> [OVERRIDES="key=val …"]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required"; exit 1; }
	@$(PYTHON) -m scripts.train +experiment=$(EXP) $(OVERRIDES)
```

`make train` is kept for muscle-memory one-off training without engaging the TUI. Everything else moves to `dsim` or direct Hydra invocations.

### Removed Makefile targets and their replacements

| Removed | Replaced by |
|---|---|
| `make demo` | TUI → Demo task; or `mjpython demo/menu.py <key>` directly |
| `make camera-test CAM=<name>` | TUI → Camera Test task; or `python demo/camera_test.py --cam <name>` directly |
| `make eval` | `python -m scripts.eval_ppo +eval_ppo=default …` |
| `make eval-headless` | `python -m scripts.eval_ppo +eval_ppo=default eval.gui=false` |
| `make eval-team` | `python -m scripts.eval_team +eval_team=default learner=… opponent=…` or TUI → Match |
| `make resume` | `dsim resume <run-name>` |
| `make lineage` | `dsim lineage --target <uri-or-path>` |
| `make promote` | `dsim promote <run-name>` |
| `make list-runs` | `dsim list-runs [--run <name>]` |
| `make obs-specs` | `dsim obs-specs` (kept as a `dsim` subcommand — pretty-prints SPEC_BY_NAME) |
| `make describe-run` | `dsim describe-run <run-name> [--trial T]` (wraps the existing `scripts.render_model_doc`) |
| `make sweep` / `sweep-agent` / `sweep-agents` | `dsim sweep create` / `dsim sweep agent` / `dsim sweep agents <args…>` |

[repo/README.md](README.md) and [repo/CLAUDE.md](CLAUDE.md) get a "CLI surface" section explaining the three modes: `dsim` (inspection/dispatch), Hydra apps (composable runs), Make (chores).

## Testing strategy

### Slice 1 tests

| Path | Coverage |
|---|---|
| `tests/core/test_inventory.py` | Reads a synthetic `models/` fixture (vendored + cache); asserts ModelInfo fields; covers the legacy `run_info.toml` fallback path. |
| `tests/core/test_obs_compat.py` | Preflight for matched / surgery-required / incompatible parent×child combinations; covers all three obs specs (`DUEL_V1_BODY`, `DUEL_V2_WORLD`, `DUEL_V3_BODY_EGO`). |
| `tests/core/test_eval_core.py` | `run_scenario` with a stub learner against `zero_red` (single-step termination); asserts terminal-cause bucketing for each cause + take-down counter. |
| `tests/core/test_eval_report.py` | Writes a 3-scenario battery into a tmp dir; asserts summary.md, results.csv, per_episode.jsonl shape; CSV round-trips through pandas. |
| `tests/scripts/test_eval_battery.py` | End-to-end: runs `conf/eval_battery/quick.yaml` against the canary_team setup, 1 episode per scenario; asserts MD/CSV/JSONL exist. |
| `tests/scripts/test_eval_team_hydra.py` | Asserts the Hydra-migrated `eval_team` produces the same per-episode result as the old argparse path for a fixed seed + scenario. |
| `tests/scripts/test_eval_ppo_hydra.py` | Same for single-agent. |
| `tests/dsim/test_cli_smoke.py` | Each `dsim` subcommand renders `--help` without error. |
| `tests/dsim/test_inventory_cli.py` | `dsim inventory --json` round-trips through json + matches `core.inventory.inventory()` output. |
| `tests/dsim/test_obs_preflight_cli.py` | Exit codes: 0 for compatible, 1 for surgery-required, 2 for incompatible. |

### Slice 2 tests

| Path | Coverage |
|---|---|
| `tests/tui/test_app_smoke.py` | `DroneSimApp().run_test()` mounts + renders one frame without raising. |
| `tests/tui/test_widget_kwargs.py` | Every Pane / sub-widget accepts `id=`, `classes=` at construction. |
| `tests/tui/test_pane_navigation.py` | `pilot.press` walks Pane 1 ↓×3 → → Pane 2 ↓×N → → Pane 3 → Esc → ← Esc → q without error. |
| `tests/tui/test_overlapping_ids.py` | Switches between Train and Resume tasks (both have `field-RUN_NAME`) without `DuplicateIds`. |
| `tests/tui/test_subprocess_dispatch.py` | Argv-builder for each task produces the expected command; doesn't actually spawn (mocked). |

### Canary

Single-agent canary (`step 434 / reward 7.3837`) and team canary must remain byte-identical after Slice 1 lands. Slice 2 has no effect on training canaries (no SB3 callback changes).

## Migration / rollout plan

1. **Slice 1 (commands + engine)** lands first as one PR. Tests + canaries green. Drops the old `eval_team` argparse surface; `make eval-team` is removed. Users move to `python -m scripts.eval_team +eval_team=default …` or wait for the TUI.
2. **Documentation update** (in the same PR or immediately after): README.md + CLAUDE.md "CLI surface" sections; brain/index.md + brain/changelog.md updated.
3. **Slice 2 (TUI)** lands as a second PR on the same `feature/ml-infra-part-3` branch (or a separate branch off it). Tests green; manual smoke test against a real eval+train flow.
4. **Merge into develop** with `--no-ff` merge commit (per project convention).
5. **`feature/tui-launcher` retired**: branch + worktree removed after Slice 2 lands. The plan/spec docs in `repo/docs/superpowers/{specs,plans}/2026-05-1{0,1}-tui-launcher-*` are kept as historical record.

## Open questions

1. **Should `[e] edit overrides` in the Train task be implemented in Slice 2?** Three options: (a) full form-with-validation tied to Hydra schema; (b) open `$EDITOR` on a tmpfile pre-filled with `# overrides for blue_v5\n# one key=value per line\n` and parse the diff; (c) drop from Slice 2 — overrides are always reachable via shell. **Recommendation: drop from Slice 2, revisit if it hurts.**
2. **Should `dsim describe-run` regenerate the MODEL.md (default) or only display the existing one?** **Recommendation: regenerate by default (`--no-regen` flag to display existing). Matches the existing `make describe-run` behavior.**
3. **Should obs-preflight be a guard rail in `scripts/train.py`'s `init.mode == pretrain` path?** Currently `check_obs_compat` strict-raises mid-init. The preflight could run earlier and emit a friendly message, then strict-raise the same way for `pretrain` (or auto-suggest `init.mode=warm_start`). **Recommendation: warn-then-raise; do not auto-switch modes (too magical).**
4. **`dsim sweep` subcommand layout** — `dsim sweep create <name>` vs `dsim sweep new <name>`. **Recommendation: `create` (matches W&B's own terminology).**
5. **`models/.cache/` entries** — included in `dsim inventory` by default or behind `--include-cache`? **Recommendation: behind `--include-cache` (default lists only committed-vendored), so the default output is the canonical promoted set.**

## References

- Hydra Part 1 spec: [docs/superpowers/specs/2026-05-13-hydra-config-design.md](docs/superpowers/specs/2026-05-13-hydra-config-design.md)
- W&B Part 2 spec: [docs/superpowers/specs/2026-05-14-wandb-integration-design.md](docs/superpowers/specs/2026-05-14-wandb-integration-design.md)
- Model-doc spec (just merged): [docs/superpowers/specs/2026-05-16-model-doc-generator-design.md](docs/superpowers/specs/2026-05-16-model-doc-generator-design.md)
- blue_v7 spec (just merged): [docs/superpowers/specs/2026-05-18-blue-v7-body-ego-design.md](docs/superpowers/specs/2026-05-18-blue-v7-body-ego-design.md)
- TUI launcher historical work: `feature/tui-launcher` branch (~30 commits, idle since 2026-05-12); the spec/plan docs live on that branch (commit `fa49d33`) and are not on develop. This design retires the branch.
