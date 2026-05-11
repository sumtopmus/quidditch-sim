# TUI Launcher — Design

**Goal.** Replace the single interactive surface in the project (`make demo`'s 4-option stdin prompt) with a full-screen Textual dashboard that drives the day-to-day workflow — demos, training, eval, promotion, lineage, repro, tensorboard — with form-driven parameter pickers that read state from disk, plus a persistent training-progress widget that renders structured progress from a JSON status file the training process emits.

**Why now.** The project has accreted ~22 Makefile targets, most of which take long, hand-typed env-var arguments (`RUN_NAME=...`, `TRIAL=...`, `RED=models/<long_name>/best_model`, …). The 1v1 team workflow especially makes this painful: an `eval-team` invocation needs two model paths. A shareable controller/dashboard fixes the friction, lets a new collaborator discover what the project can do, and gives an "at-a-glance" view of training state.

**Non-goals.**
- Editing `config/training.toml` from inside the TUI. The TUI runs jobs; config edits belong in an editor.
- A fuzzy command palette overlay (`/` hotkey). Tracked as future, out of scope for v1.
- Multi-host / remote runs.
- Persistent "recently used" history across sessions.
- Rendering live SB3 log tables verbatim — the structured progress widget supersedes them; the full subprocess stdout is available in the log overlay if needed.
- Mouse-only navigation. Mouse is supported via Textual but keyboard parity is required for every action.
- A GUI alternative (Tk, etc.). TUI only.

## Design Decisions

### 1. Textual as the UI toolkit; one new pip dep

Textual is the obvious pick: it's the modern Python TUI framework, MIT, by the author of `rich` (already in `requirements.txt`), supports mouse, scrolling, CSS-like theming, and Nerd Font glyphs. The only new pip dep is `textual` itself; `rich` is already pulled in.

Alternatives considered:
- **`prompt_toolkit` + custom layout** — lower-level, more work, lower visual ceiling.
- **`questionary` / `inquirer`** — sequential prompts, no full-screen app; can't host a persistent training widget. Ruled out by the dashboard requirement (Question 2 → answer B).
- **A GUI (Tk / web)** — explicitly rejected; the user wants terminal-native.

### 2. TUI never re-implements logic — only shells out to existing scripts

The TUI is a thin orchestration layer. Each action constructs the same `python scripts/X.py …` or `mjpython scripts/X.py …` invocation the Makefile uses today, runs it as a subprocess, and reads outputs (stdout for logs, the JSON progress file for training). No business logic is duplicated — the entry-point scripts (`scripts/train_ppo.py`, `scripts/train_team_ppo.py`, `scripts/eval_ppo.py`, `scripts/eval_team.py`, `scripts/lineage.py`, demo modules, `tensorboard`) remain the source of truth.

This implies a hard contract: every action's underlying script must remain runnable from the shell without the TUI. We preserve this both because the design depends on it and because non-interactive (CI, paper-repro) invocations need to keep working.

### 3. Structured training progress via an always-on SB3 callback

`core/training/tui_progress_callback.py` defines `TUIProgressCallback(BaseCallback)`. It is wired unconditionally into `scripts/_train_common.build_callbacks` — every future training run (single-agent or team, TUI-launched or shell-launched) writes a status file. This is intentional: the callback is ~80 bytes per write, the cost is negligible, and being unconditional makes the JSON status a reliable feature other tools could consume.

**Write target:** `<run_dir>/tui_progress.json`, where `run_dir` is the existing per-trial directory (`runs/<run_name>/<timestamp>/`).

**Write cadence:** on `_on_step` when `self.num_timesteps % write_every == 0`, where `write_every = max(1, total_timesteps // 500)` so we get ~500 updates per run regardless of length. Empirically this is fast enough to feel "live" (sub-second on multi-million-step runs) without being chatty.

**Atomicity:** write to `tui_progress.json.tmp` then `os.replace` — a partial JSON in the watched file would crash the TUI's parser.

**Schema (v1):**

```json
{
  "schema_version": 1,
  "ts": 1746951234.5,
  "run_name": "ppo_hoop_blue_3",
  "trial": "20260511_010533",
  "kind": "single",
  "learner": null,
  "opponent": null,
  "step": 2300000,
  "total_steps": 10000000,
  "fps": 850,
  "elapsed_sec": 4320,
  "ep_rew_mean": 4.21,
  "ep_len_mean": 1234.0,
  "best_so_far": {"reward": 5.83, "step": 1800000},
  "recent_rewards": [3.8, 4.0, 4.2, 4.1, 4.3, 4.2, 4.4, 4.4, 4.5, 4.3, 4.5, 4.6, 4.4, 4.3, 4.2, 4.2]
}
```

- `kind` ∈ {`single`, `team`}, populated from a constructor arg.
- `learner` / `opponent` are `null` for single-agent, set for team.
- `recent_rewards` is the last 16 `ep_rew_mean` samples (a ring buffer kept on the callback instance).
- `best_so_far` is updated from `EvalCallback.best_mean_reward` if available; falls back to tracking the rolling-mean max ourselves.

`schema_version` exists so we can evolve the file format and have the TUI gracefully degrade if it sees a newer version it doesn't understand.

### 4. Two subprocess slots: one training-exclusive, one auxiliary

The TUI's `process/manager.py` holds at most two live subprocesses:

| Slot | Members | Concurrency |
|---|---|---|
| `training` | `train_ppo`, `train_team_ppo` (incl. `--resume`, `--warm-start`) | At most 1 active. Other train actions disabled in the action tree while this is running (rendered with a lock glyph). |
| `aux` | demos, `eval`, `eval-team`, `tensorboard`, `lineage`, `repro`, `promote`, `list-runs` | At most 1 active. Launching a new aux while one is running prompts "stop current and start new?" — the existing aux is then sent `SIGINT`. |

**Rationale for letting training + aux run concurrently.** Eval/demos/TB don't compete heavily with training for CPU (eval is a single rollout at a time, TB is a daemon serving HTTP). The user explicitly requested this. The MuJoCo viewer takes its own OS window, so it doesn't fight the TUI for the terminal.

**Tensorboard as a special case.** TB is a long-running daemon — distinct from one-shot aux actions. We treat it as occupying the `aux` slot for simplicity but expose a separate "stop tensorboard" hotkey. When TB is the aux process, other aux actions are blocked until the user stops TB (or chooses to interrupt it via the stop prompt). This is acceptable because TB is normally launched and left alone in a corner pane.

### 5. Action catalog — 16 actions, grouped, declarative

Actions are declared as `Action` dataclasses in `tui/actions/`. Each declares: name, glyph, group, fields (list of `FieldSpec`), `build_argv(form_values, scan)` → `list[str]`, and a `requires_mjpython: bool` flag.

Fields are typed with a `kind` and an optional `source` for pickers. The form renderer dispatches off `kind`:

| Kind | Widget | Source values |
|---|---|---|
| `text` | `Input` | — |
| `int` | `Input` (numeric validator) | optional `default` |
| `bool` | `Switch` | `default` |
| `picker:models` | `Select` | `scan.promoted_models()` — entries in `models/<name>/` with `best_model.zip` |
| `picker:runs` | `Select` | `scan.run_names()` — top-level dirs in `runs/` |
| `picker:trials_in_run` | `Select` | `scan.trials_in_run(run_name)` — `runs/<run>/<trial>/`, sorted desc by name (timestamp) |
| `picker:checkpoints_in_trial` | `Select` | `scan.checkpoints(run, trial)` — sorted desc by step |

Pickers that depend on earlier picker values (e.g. `picker:trials_in_run` depends on the `RUN_NAME` field) re-populate when the upstream value changes, via a reactive Textual binding.

**The 14 actions:**

| Group | Action | Glyph (Nerd Font codepoint) | Fields | Underlying invocation |
|---|---|---|---|---|
| Demo | `hover` | `` (nf-fa-gamepad) | — | `mjpython demo/menu.py hover`* |
| Demo | `waypoint` | `` | — | `mjpython demo/menu.py waypoint`* |
| Demo | `takedown` | `` | — | `mjpython demo/menu.py takedown`* |
| Demo | `score-through-tag` | `` | — | `mjpython demo/menu.py score-through-tag`* |
| Train | `train single-agent` | `` (nf-fa-rocket) | `RUN_NAME` text, `PRETRAIN` picker:models (opt) | `python scripts/train_ppo.py [--run-name X] [--pretrain Y/best_model]` |
| Train | `train-team-red` | `` (nf-fa-flag) | `RUN_NAME` text, `WARM_START` picker:models (opt) | `python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue [--run-name] [--warm-start Y/best_model]` |
| Train | `train-team-blue` | `` | `RUN_NAME` text, `RED` picker:models (req) | `python scripts/train_team_ppo.py --learner blue_0 --opponent frozen:<RED>/best_model [--run-name]` |
| Train | `resume` | `` (nf-fa-refresh) | `RUN_NAME` picker:runs, `TRIAL` picker:trials_in_run (default latest), `CHECKPOINT` picker:checkpoints_in_trial (default latest) | `python scripts/train_ppo.py --run-name X --resume <ckpt>` |
| Train | `resume-team` | `` | as `resume` + `LEARNER` text (opt) + `OPPONENT` text (opt) | `python scripts/train_team_ppo.py --resume <ckpt> --run-name X [--learner …] [--opponent …]` |
| Eval | `eval` | `` (nf-fa-crosshairs) | `RUN_NAME` picker:runs, `TRIAL` picker:trials_in_run, `EPISODES` int (def 10), `GUI` bool (def true) | `[mjpython\|python] scripts/eval_ppo.py --model <trial>/best_model --episodes N [--no-render]` |
| Eval | `eval-team` | `` | `RED` picker:models (req), `BLUE` picker:models (req), `EPISODES` int (def 5 if GUI else 100), `GUI` bool (def true), `DETERMINISTIC` bool (def true) | `[mjpython\|python] scripts/eval_team.py --red <RED-spec> --blue <BLUE-spec> [--episodes] [--gui] [--deterministic]` |
| Manage | `tensorboard` | `` (nf-fa-bar_chart) | `RUN_NAME` picker:runs (opt, default all) | `tensorboard --logdir runs[/<run>]` |
| Manage | `promote` | `` (nf-fa-trophy) | `RUN_NAME` picker:runs, `TRIAL` picker:trials_in_run | invoke the same shell logic the `promote` Makefile target does today — moved into `scripts/promote.py` as part of this work (see §10) |
| Manage | `list-runs` | `` (nf-fa-folder_open) | — | `python scripts/list_runs.py` — also extracted from the Makefile (see §10) |
| Manage | `lineage` | `` (nf-fa-link) | `RUN_NAME` picker:runs, `TRIAL` picker:trials_in_run | `python scripts/lineage.py <trial_dir>` |
| Manage | `repro` | `` (nf-fa-refresh) | `MODEL` picker:models | `python scripts/repro.py <model>` — extracted from Makefile (see §10) |

*Demo invocations: today, `demo/menu.py` reads stdin to pick a demo, then `importlib`-imports and calls the module's `main()`. The TUI replaces the stdin step. Simplest path: extend `demo/menu.py` to accept the demo key as a CLI arg, so `mjpython demo/menu.py hover` runs hover without prompting. Falls back to the interactive prompt when no arg is given — keeps the file self-contained for anyone who runs it directly.

`requires_mjpython` is `True` for the four demos and for `eval` / `eval-team` when `GUI=true`. Everything else uses `python`.

### 6. Dashboard layout — three panes + docked training widget

```
┌─ drone-quidditch ────────────────────── conda: uav │ branch: feature/tui-launcher ──┐
│ ┌─ Actions ──────────┐ ┌─ Form ─────────────────────┐ ┌─ State ──────────────────┐  │
│ │  Demo              │ │ ⚑ train-team-blue          │ │  Promoted models          │  │
│ │   hover            │ │                            │ │   • red_v1                │  │
│ │   waypoint         │ │  RUN_NAME  ppo_hoop_blue_3 │ │     ppo_hoop_red_1_…      │  │
│ │   takedown         │ │  RED       red_v1       ▾  │ │   • blue_v1               │  │
│ │   score-through-…  │ │                            │ │     ppo_hoop_blue_1_…     │  │
│ │  Train         ⊘   │ │  [ Run ]   [ Dry-run ]     │ │                           │  │
│ │  Eval              │ │                            │ │  Recent trials            │  │
│ │   eval             │ └────────────────────────────┘ │   ppo_hoop_blue_3         │  │
│ │   eval-team        │                                │     20260511_104210 ● live│  │
│ │  Manage            │                                │   ppo_hoop_blue_2         │  │
│ │   tensorboard      │                                │     20260511_010533       │  │
│ │   promote          │                                │   ppo_hoop_blue_1         │  │
│ │   list-runs        │                                │     20260507_194423       │  │
│ │   lineage          │                                │                           │  │
│ │   repro            │                                │                           │  │
│ └────────────────────┘                                └───────────────────────────┘  │
│ ┌─ ⚑ Training — ppo_hoop_blue_3 (team-blue vs frozen:red_v1) ────────────────────┐   │
│ │ ███████████░░░░░░░░░░░░░░░░░░░░░░░  2.30M / 10.00M    23%    ETA 47m13s        │   │
│ │ ep_rew_mean 4.21 ↑    best 5.83 @ 1.80M    fps 850    elapsed 1h12m            │   │
│ │ ▁▂▂▃▃▄▄▄▅▅▅▆▆▆▇▇▇▇█  (last 16)              TB http://localhost:6006           │   │
│ └────────────────────────────────────────────────────────────────────────────────┘   │
│  q quit │ enter run │ ctrl-c stop │ t toggle TB │ l logs │ ? help                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Pane behavior:**

- **Action tree (left):** Grouped, each group with a header. Glyph per action. Disabled actions (Train group while training is running) render dimmed with a lock glyph (``). Keyboard nav: `↑/↓` move, `enter` select, `space` toggle group expand.
- **Form (center):** Re-renders each time an action is selected. Empty welcome screen when nothing is selected (project name, version, conda env, branch, "press ↑/↓ to navigate"). `tab`/`shift+tab` move between fields; `enter` from any field activates `[ Run ]`. `Run` validates required fields, builds argv via `action.build_argv(form_values, scan)`, hands off to `process.manager`. `Dry-run` shows the constructed argv as a copyable string in the log overlay without spawning.
- **State pane (right):** Two stacked sections — Promoted models (from `models/`) and Recent trials (latest 10 across all runs, sorted desc by trial timestamp). Live trials get a `● live` marker. Updated on filesystem change events (debounced 1s) plus a 5s heartbeat as fallback. A live trial corresponds to one with a fresh `tui_progress.json` (modified in the last 30s).
- **Training widget (docked bottom):** Visible only when the training slot is active. 5 rows. Renders the structured progress JSON; updates every 500ms by re-reading the file (cheap; small file, atomic writes). Sparkline via `rich`'s built-in renderer.
- **Footer:** Hotkey hints. Adapts to current state — e.g. `ctrl-c stop` only shown when something is stoppable.

**Idle vs busy state transitions:**

| State | Action tree | Form pane | State pane | Training widget |
|---|---|---|---|---|
| Idle | All actions enabled | Welcome (or last-selected action's form if user navigated there) | Visible | Hidden |
| Training only | Train-group disabled (lock glyph) | Selected action's form | Visible; live trial highlighted | Visible (docked bottom) |
| Aux only | All enabled, current aux shows ▶ in tree | Selected action's form | Visible | Hidden |
| Training + aux | Train-group disabled, current aux ▶ marked | Selected action's form | Visible | Visible |

### 7. Disk state scanner

`tui/state/scan.py` is a stateless module that walks `models/` and `runs/` on demand. Functions:

- `promoted_models() -> list[PromotedModel]` — entries in `models/<name>/` containing `best_model.zip`. Returns name, path, optional alias (`red_v1`/`blue_v1` resolved from a small lookup table that the user maintains — initially read from a single source on the model dir; if absent, no alias).
- `run_names() -> list[str]` — sorted top-level dirs in `runs/`.
- `trials_in_run(run_name) -> list[Trial]` — `runs/<run_name>/<timestamp>/` entries, desc by name. Each `Trial` carries: name, full path, has_best_model (bool), is_live (mtime of `tui_progress.json` within 30s if present).
- `checkpoints(run, trial) -> list[Checkpoint]` — `.zip` files in `<trial>/checkpoints/`, sorted desc by step. Step parsed from filename (`ppo_<step>_steps.zip`).
- `latest_trial() -> Trial | None` — globally latest trial timestamp across all runs.

Each call is a fresh `os.scandir` — no caching. The scanner is cheap enough (low-hundreds of dirs even for an active project) and avoiding state simplifies the FS-change story. The state-pane widget invokes `scan` on every refresh tick.

**Alias source:** `models/<name>/alias.txt` — a single-line file with the alias (`red_v1`). Optional. If we want to centralize, we can later switch to a top-level `models/aliases.toml`, but the per-dir file is the simpler starting point and survives `make promote` without extra plumbing.

### 8. Process manager

`tui/process/manager.py` holds the two slots and exposes:

```python
class ProcessManager:
    training: ManagedProcess | None
    aux:      ManagedProcess | None

    def start_training(self, argv, env=None, run_dir: Path) -> None: ...
    def start_aux(self, argv, env=None, *, requires_mjpython: bool) -> None: ...
    def stop(self, slot: Literal["training", "aux"]) -> None: ...   # sends SIGINT
    def is_running(self, slot) -> bool: ...
    def tail_logs(self, slot, n: int = 200) -> list[str]: ...
```

Each `ManagedProcess` wraps a `subprocess.Popen` plus a 1000-line ring buffer of merged stdout+stderr (line-buffered via a small reader thread). On exit, captures the return code and surfaces it as a Textual notification (`.notify(...)`) — green for `0`, red otherwise, with the last few stderr lines and a "press `l` for full logs" hint.

**SIGINT semantics.** SB3 catches `KeyboardInterrupt` and writes a final checkpoint before exit — we want that. So `stop("training")` sends SIGINT, not SIGTERM, mirroring what `ctrl-c` in a normal shell would do. For `aux`, we send SIGINT first; if the process is still alive 5s later we escalate to SIGTERM.

**Quit-with-children policy.** On TUI quit with a running training process, prompt: "Training is still running. Quit will leave it running in the background. Stop training and quit, or quit and let it run?" Default: let it run. Either way the JSON status file keeps being written, so re-opening the TUI re-discovers the live process via the state pane's "live trial" detection.

Tensorboard gets the simpler treatment: on TUI quit it is always stopped (it's only useful with a UI). Non-TB aux processes (running eval, demo, or one-shot management actions) are sent SIGINT on TUI quit — they're short-lived enough that orphaning them via a "keep running?" prompt every time would be more friction than value.

### 9. Visual theme — Nerd Font glyphs + Textual CSS

`tui/theme.py` exports:

```python
class Glyphs:
    DEMO       = ""
    TRAIN      = ""
    TRAIN_TEAM = ""
    RESUME     = ""
    EVAL       = ""
    TENSORBOARD = ""
    PROMOTE    = ""
    LIST_RUNS  = ""
    LINEAGE    = ""
    REPRO      = ""
    LOCK       = ""
    LIVE       = "●"
    RED_DRONE  = "9"   # nf-md-robot
    BLUE_DRONE = "9"
    UP_ARROW   = "↑"
    DOWN_ARROW = "↓"
```

ASCII fallback (`--ascii` flag): `Glyphs` re-bound to `[D]`, `[T]`, `[E]`, etc. for users without a Nerd Font.

**CSS (`tui/app.tcss`):** Textual's CSS-like styling. Palette:

```
$bg          = #0f1117
$panel       = #161922
$panel-border = #2a2f3d
$accent      = #a78bfa   /* violet */
$accent-2    = #7c69d6
$red-drone   = #c1432d
$blue-drone  = #2d63c1
$success     = #7ec27e
$warn        = #e0a85a
$dim         = #5a6178
$text        = #e2e4ea
```

Rounded borders on every pane; the training widget uses a thicker border with `$accent` to call attention. The progress bar fills in `$success` when `ep_rew_mean` is trending up over the last 4 samples, `$warn` when flat or down.

### 10. Makefile strip + new entry point

**Remove these targets** (now TUI-only):

```
demo
train, train-team-red, train-team-red-warm, train-team-blue
resume, resume-team
eval, eval-headless, eval-team
tensorboard, lineage, promote, repro, list-runs
```

Also remove the now-unused variable-resolution block (`_LATEST_IN_RUN`, `_LATEST_OVERALL`, `_LATEST_CKPT`, `_RESUME_RUN`, `_TRIAL_DIR`). The trial/checkpoint resolution logic moves to Python (`tui/state/scan.py`), the single source of truth for both pickers and any future scripting.

**Keep:** `help`, `install`, `configs`, `clean`, `test`, `test-fast`, `test-warm`, `camera-test`.

**Add:**

```makefile
ui: ## 🎛  Interactive launcher (TUI dashboard)
	@$(PYTHON) -m tui

demo: ## 🎮 Open TUI on the Demo group (alias)
	@$(PYTHON) -m tui --group demo
```

`make demo` is preserved as a one-liner that opens the TUI pre-positioned on the Demo group — auto-selected by the `--group demo` CLI arg passed to `python -m tui`. Muscle memory survives the strip.

**Promote / list-runs / repro extraction.** Today these are shell logic embedded in Make recipes (file copying, dir listing, config restoration). To remove them from the Makefile cleanly while keeping the behavior, extract each into a small Python script:

- `scripts/promote.py` — accepts a trial dir, copies `best_model.zip`/`info.toml`/`config_snapshot.toml` into `models/<flat-name>/`. Replaces the `promote` Make recipe verbatim. Also callable from shell for one-offs.
- `scripts/list_runs.py` — walks `runs/` and `models/`, prints the same tree the Make target prints today.
- `scripts/repro.py` — copies `models/<name>/config.toml` to `config/training.toml`, with the same "older promote format" warning.

These are deliberately not new abstractions — they are the existing shell logic, one-to-one, in Python. The TUI's `promote` / `list-runs` / `repro` actions shell out to them.

### 11. File inventory

| Action | Path | Why |
|---|---|---|
| Create | `tui/__init__.py` | package marker |
| Create | `tui/__main__.py` | entrypoint; parses `--group`/`--ascii`; runs `DroneSimApp` |
| Create | `tui/app.py` | `DroneSimApp(textual.App)` — composes panes, hotkeys, top-level state |
| Create | `tui/app.tcss` | Textual CSS theme |
| Create | `tui/theme.py` | `Glyphs` class + ASCII fallback |
| Create | `tui/actions/__init__.py` | re-exports `ACTIONS` from registry |
| Create | `tui/actions/base.py` | `Action`, `FieldSpec`, kind enum |
| Create | `tui/actions/demos.py` | 4 demo actions |
| Create | `tui/actions/train.py` | train + resume actions (5 total) |
| Create | `tui/actions/eval.py` | eval + eval-team (2 total) |
| Create | `tui/actions/manage.py` | tensorboard, promote, list-runs, lineage, repro |
| Create | `tui/actions/registry.py` | `ACTIONS: list[Action]` — single source of truth for the catalog |
| Create | `tui/widgets/action_tree.py` | left pane |
| Create | `tui/widgets/action_form.py` | center pane; field-kind dispatch |
| Create | `tui/widgets/state_pane.py` | right pane; promoted models + recent trials |
| Create | `tui/widgets/training_widget.py` | docked bottom; reads JSON via `process/progress.py` |
| Create | `tui/widgets/log_overlay.py` | full-screen modal showing last-N lines of the active slot's ring buffer |
| Create | `tui/process/manager.py` | `ProcessManager` + `ManagedProcess` |
| Create | `tui/process/progress.py` | tails `tui_progress.json`; exposes a reactive `ProgressSnapshot` |
| Create | `tui/state/scan.py` | disk scanner functions |
| Create | `core/training/__init__.py` | new submodule |
| Create | `core/training/tui_progress_callback.py` | `TUIProgressCallback(BaseCallback)` |
| Modify | `scripts/_train_common.py` | append `TUIProgressCallback(run_dir=run_dir, total_timesteps=..., kind=..., learner=..., opponent_spec=...)` to `build_callbacks` return list; thread `total_timesteps` / `kind` / `learner` / `opponent_spec` as kwargs into `build_callbacks` |
| Modify | `scripts/train_ppo.py` | pass `total_timesteps=` and `kind="single"` to `build_callbacks` |
| Modify | `scripts/train_team_ppo.py` | pass `total_timesteps=`, `kind="team"`, `learner=`, `opponent_spec=` |
| Modify | `demo/menu.py` | accept an optional positional arg `[demo-key]`; if given, skip the prompt and run that demo directly |
| Create | `scripts/promote.py` | port of Make `promote` recipe |
| Create | `scripts/list_runs.py` | port of Make `list-runs` recipe |
| Create | `scripts/repro.py` | port of Make `repro` recipe |
| Modify | `Makefile` | strip 15 targets + variable-resolution block; add `ui` + `demo` (alias) |
| Modify | `requirements.txt` | add `textual` |
| Create | `tests/unit/test_tui_actions.py` | argv construction per action, table-driven |
| Create | `tests/unit/test_tui_scan.py` | disk scanner over a tmp fixture tree |
| Create | `tests/unit/test_tui_progress_callback.py` | callback writes schema-conformant JSON |
| Create | `tests/unit/test_tui_progress_tail.py` | `process/progress.py` parses partial / stale / missing files gracefully |
| Modify | `tests/conftest.py` | if any new shared helpers are needed (likely a `tmp_runs_tree` fixture for the scan test) |
| Modify | `README.md` | one-line mention of `make ui` |
| Modify | `CLAUDE.md` | one-line mention so future sessions know `make ui` is the entry point |

No changes to `envs/quidditch/`, `core/world.py`, `core/quadrotor.py`, `core/policies/`. The TUI is strictly additive on top of existing entrypoints, except for the `_train_common.py` callback wire-up and `demo/menu.py` CLI-arg addition.

### 12. Failure modes & tests

| Failure | Test |
|---|---|
| Action argv builder produces wrong arg for missing optional field | unit (`test_tui_actions.py`) — assert `build_argv({...})` for each action against an expected argv list |
| Picker default selection when latest trial has no `best_model.zip` | unit — `scan.trials_in_run` flags `has_best_model=False` and `eval` action's form disables `[ Run ]` |
| `tui_progress.json` written by an old training run with `schema_version > 1` | unit (`test_tui_progress_tail.py`) — graceful "unsupported schema" widget, doesn't crash |
| `tui_progress.json` half-written (partial JSON) | unit — atomic write via `os.replace` is the production path; test reads a deliberately-truncated file and asserts the tail returns the previous good snapshot |
| `tui_progress.json` missing | unit — returns `None`; widget shows "waiting for first checkpoint…" |
| Training subprocess dies with non-zero exit | manual — observe red notification + log overlay shows last lines (smoke-test via running `make ui` → train with bogus pretrain path) |
| User quits TUI while training runs | manual — observe prompt, "let it run" returns to shell with training alive; re-opening TUI sees `● live` on the trial |
| `--ascii` flag used without Nerd Font | manual — glyphs replaced by bracket-letter forms, layout unchanged |
| TUIProgressCallback always-on doesn't break existing tests | run `make test` after change; `tests/integration/test_scoring_canary.py` must still assert `step 434 / reward 7.3837` (the callback writes JSON to disk — it does not alter env state, RNG, or model behavior) |
| `make demo` alias still works | manual — `make demo` opens the TUI with the Demo group pre-selected |

The canary check (last row) is the most important — the TUI work is supposed to be additive and not perturb training numerics. The callback only reads `self.num_timesteps`, `self.model.ep_info_buffer`, and `EvalCallback.best_mean_reward` (read-only), and writes to disk. No RNG consumption, no policy or env mutation.

### 13. Out of scope (deliberately)

- Fuzzy command palette / `/` hotkey
- Per-user persistent recents
- Inline config editing
- Remote / multi-host runs
- Verbatim SB3 log-table rendering
- A GUI variant
- TUI snapshot/pilot tests (Textual offers a pilot harness; deferred — the widgets are presentation-only over tested pure-data functions, where the value of snapshot tests is low compared to the maintenance cost)
- Brain integration (`brain/` lives outside the repo and is per-user)

## Open Questions

None for the TUI itself — the design above is decided. Implementation will surface micro-questions (exact Textual widget APIs, exact field validation messages); those are plan-level, not spec-level.
