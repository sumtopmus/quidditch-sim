# Experiment-Campaign Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code experiment-campaign loop — a controller skill plus worker subagents that run RL experiments, monitor W&B telemetry, early-stop bad runs, analyze results, design the next experiment, and continue (interactive or autonomous), backed by one deterministic telemetry/kill-rule helper.

**Architecture:** A durable **controller skill** (`/run-campaign`) drives the loop in the user's session and dispatches fresh **worker subagents** (`experiment-analyst`, `experiment-monitor`, `experiment-coder`) per job. Deterministic mechanics — reading W&B history and evaluating declared kill-rules — live in pure, unit-tested Python under `core/campaign/`, exposed via a read-only `dsim campaign-status` command. Campaign state lives in a committed `docs/campaigns/<name>/` ledger, with a brain rollup at milestones.

**Tech Stack:** Python 3.13, uv, Typer + Rich (CLI), `wandb.Api` (telemetry, injected for tests), pytest (offline via `WANDB_MODE=disabled`), Claude Code skills/agents (markdown + YAML frontmatter).

**Conventions to follow** (from `CLAUDE.md`):
- Run everything via `uv run …`. Commands fail in the sandbox on `~/.cache/uv` — run with the sandbox disabled.
- Business logic in `core/`; thin Typer command in `dsim/commands/`; wire in `dsim/cli.py`.
- Test layout mirrors source: `tests/<package>/<module>/test_<name>.py`.
- Commit types: `feat:`, `test:`, `docs:`, `chore:`. GPG-sign commits (`-S`). End commit messages with the `Co-Authored-By` trailer.
- `make test-fast` skips `@pytest.mark.slow`.

---

## File Structure

**New Python (deterministic core + CLI):**
- `core/campaign/__init__.py` — package marker.
- `core/campaign/killrules.py` — pure kill-rule evaluator: `evaluate(rows, rules, baselines) -> KillVerdict`.
- `core/campaign/telemetry.py` — thin `wandb.Api` history reader: `read_telemetry(run_path, *, api_factory) -> CampaignTelemetry`.
- `dsim/commands/campaign_status.py` — `dsim campaign-status` Typer command (read-only; prints JSON).
- `dsim/cli.py` — MODIFY: register `campaign-status`.

**New tests:**
- `tests/core/campaign/__init__.py`
- `tests/core/campaign/test_killrules.py`
- `tests/core/campaign/test_telemetry.py`
- `tests/dsim/test_campaign_status_cli.py`
- `tests/test_agent_assets.py` — validates skill/agent frontmatter + campaign templates.

**New Claude Code agent layer (committed in the worktree's `.claude/`):**
- `.claude/skills/run-campaign/SKILL.md` — the controller.
- `.claude/agents/experiment-analyst.md` — report card + ranked next-experiment proposals.
- `.claude/agents/experiment-monitor.md` — gray-zone kill verdict.
- `.claude/agents/experiment-coder.md` — guarded, test-backed code changes.

**New campaign scaffolding:**
- `docs/campaigns/TEMPLATE/goal.md`
- `docs/campaigns/TEMPLATE/ledger.md`
- `docs/campaigns/TEMPLATE/frontier.md`

**Docs:**
- `README.md` — MODIFY: add `campaign-status` to the CLI surface section.

---

## Kill-rule schema (shared contract)

A rules file is JSON. The controller writes one per experiment; `dsim campaign-status --rules <file>` evaluates it; `core/campaign/killrules.py` is the single implementation.

```json
{
  "rules": [
    {"type": "nan", "metrics": ["train/loss", "rollout/ep_rew_mean"]},
    {"type": "floor_at_step", "metric": "eval/success_rate", "min": 0.10, "at_step": 3000000},
    {"type": "collapse_vs_baseline", "metric": "eval/success_rate", "baseline_run": "ent/proj/abc123", "margin": 0.15},
    {"type": "budget", "max_step": 10000000, "max_wallclock_s": 7200}
  ],
  "baseline_runs": ["ent/proj/abc123"]
}
```

History rows are dicts with `_step`, `_runtime`, and metric keys (the shape `wandb` `run.history(pandas=False)` returns). `evaluate` returns a `KillVerdict(kill: bool, reasons: list[KillReason], latest: dict)`.

---

## Task 1: Kill-rule evaluator (pure core)

**Files:**
- Create: `core/campaign/__init__.py`
- Create: `core/campaign/killrules.py`
- Create: `tests/core/campaign/__init__.py`
- Test: `tests/core/campaign/test_killrules.py`

- [ ] **Step 1: Create the package markers**

```bash
mkdir -p core/campaign tests/core/campaign
: > core/campaign/__init__.py
: > tests/core/campaign/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/campaign/test_killrules.py`:

```python
"""Kill-rule evaluator — pure function over synthetic W&B history rows."""
from __future__ import annotations

import math

from core.campaign.killrules import evaluate


def _rows(*steps_metrics):
    """steps_metrics: (step, runtime, {metric: value}) tuples."""
    return [{"_step": s, "_runtime": rt, **m} for s, rt, m in steps_metrics]


def test_no_rules_never_kills():
    v = evaluate(_rows((100, 1.0, {"eval/success_rate": 0.5})), [])
    assert v.kill is False
    assert v.reasons == []
    assert v.latest["_step"] == 100


def test_nan_rule_fires_on_nan_metric():
    rows = _rows((100, 1.0, {"train/loss": math.nan}))
    v = evaluate(rows, [{"type": "nan", "metrics": ["train/loss"]}])
    assert v.kill is True
    assert v.reasons[0].rule == "nan"


def test_nan_rule_silent_when_finite():
    rows = _rows((100, 1.0, {"train/loss": 0.3}))
    v = evaluate(rows, [{"type": "nan", "metrics": ["train/loss"]}])
    assert v.kill is False


def test_floor_at_step_fires_after_gate():
    rows = _rows((3_000_000, 10.0, {"eval/success_rate": 0.04}))
    rule = {"type": "floor_at_step", "metric": "eval/success_rate",
            "min": 0.10, "at_step": 3_000_000}
    v = evaluate(rows, [rule])
    assert v.kill is True
    assert v.reasons[0].rule == "floor_at_step"


def test_floor_at_step_dormant_before_gate():
    rows = _rows((1_000_000, 10.0, {"eval/success_rate": 0.04}))
    rule = {"type": "floor_at_step", "metric": "eval/success_rate",
            "min": 0.10, "at_step": 3_000_000}
    assert evaluate(rows, [rule]).kill is False


def test_collapse_vs_baseline_fires():
    rows = _rows((2_000_000, 10.0, {"eval/success_rate": 0.20}))
    base = _rows((1_000_000, 5.0, {"eval/success_rate": 0.30}),
                 (2_000_000, 9.0, {"eval/success_rate": 0.45}))
    rule = {"type": "collapse_vs_baseline", "metric": "eval/success_rate",
            "baseline_run": "b", "margin": 0.15}
    v = evaluate(rows, [rule], baselines={"b": base})
    assert v.kill is True
    assert v.reasons[0].rule == "collapse_vs_baseline"


def test_collapse_vs_baseline_silent_when_close():
    rows = _rows((2_000_000, 10.0, {"eval/success_rate": 0.40}))
    base = _rows((2_000_000, 9.0, {"eval/success_rate": 0.45}))
    rule = {"type": "collapse_vs_baseline", "metric": "eval/success_rate",
            "baseline_run": "b", "margin": 0.15}
    assert evaluate(rows, [rule], baselines={"b": base}).kill is False


def test_budget_step_and_wallclock():
    rows = _rows((10_000_000, 7200.0, {"eval/success_rate": 0.5}))
    v = evaluate(rows, [{"type": "budget", "max_step": 10_000_000,
                         "max_wallclock_s": 7200}])
    assert v.kill is True
    assert {r.rule for r in v.reasons} == {"budget"}


def test_unknown_rule_type_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown kill-rule"):
        evaluate(_rows((1, 1.0, {})), [{"type": "bogus"}])
```

- [ ] **Step 3: Run tests to verify they fail**

Run (sandbox disabled): `uv run pytest tests/core/campaign/test_killrules.py -v`
Expected: FAIL — `ModuleNotFoundError: core.campaign.killrules`.

- [ ] **Step 4: Implement the evaluator**

Create `core/campaign/killrules.py`:

```python
"""Deterministic kill-rule evaluation over W&B history rows.

Pure functions only — no network, no wandb import.  The controller skill
passes already-fetched history (via dsim campaign-status, which uses
core.campaign.telemetry) so this module stays trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True)
class KillReason:
    rule: str
    detail: str


@dataclass(frozen=True)
class KillVerdict:
    kill: bool
    reasons: list[KillReason]
    latest: Row


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and x != x


def _latest_value(rows: list[Row], metric: str) -> tuple[Any, Any]:
    """Most recent non-null value of metric and the step it occurred at."""
    for row in reversed(rows):
        v = row.get(metric)
        if v is not None:
            return v, row.get("_step")
    return None, None


def _value_at_step(rows: list[Row], metric: str, step: Any) -> Any:
    """Baseline value at the latest baseline step <= step (step-aligned compare)."""
    if step is None:
        return None
    best = None
    for row in rows:
        s, v = row.get("_step"), row.get(metric)
        if s is None or v is None:
            continue
        if s <= step:
            best = v
        else:
            break
    return best


def evaluate(
    rows: list[Row],
    rules: list[dict],
    *,
    baselines: dict[str, list[Row]] | None = None,
) -> KillVerdict:
    """Evaluate declared kill-rules against history rows.

    rows: oldest-first list of W&B history dicts (each with `_step`, `_runtime`).
    rules: list of rule dicts (see kill-rule schema in the plan/spec).
    baselines: run-path -> baseline history rows, for collapse_vs_baseline.
    """
    baselines = baselines or {}
    reasons: list[KillReason] = []
    latest = rows[-1] if rows else {}

    for rule in rules:
        kind = rule["type"]
        if kind == "nan":
            for m in rule["metrics"]:
                val, _ = _latest_value(rows, m)
                if _is_nan(val):
                    reasons.append(KillReason("nan", f"{m} is NaN"))
        elif kind == "floor_at_step":
            m, floor, gate = rule["metric"], rule["min"], rule["at_step"]
            cur_step = latest.get("_step") or 0
            if cur_step >= gate:
                val, step = _latest_value(rows, m)
                if val is not None and val < floor:
                    reasons.append(KillReason(
                        "floor_at_step",
                        f"{m} {val:.3f} < {floor} at step {step}"))
        elif kind == "collapse_vs_baseline":
            m, margin, b = rule["metric"], rule["margin"], rule["baseline_run"]
            val, step = _latest_value(rows, m)
            base_val = _value_at_step(baselines.get(b, []), m, step)
            if val is not None and base_val is not None and val < base_val - margin:
                reasons.append(KillReason(
                    "collapse_vs_baseline",
                    f"{m} {val:.3f} < baseline {base_val:.3f} - {margin} at step {step}"))
        elif kind == "budget":
            max_step, max_wc = rule.get("max_step"), rule.get("max_wallclock_s")
            cur_step, cur_wc = latest.get("_step"), latest.get("_runtime")
            if max_step is not None and cur_step is not None and cur_step >= max_step:
                reasons.append(KillReason("budget",
                                          f"step {cur_step} >= max_step {max_step}"))
            if max_wc is not None and cur_wc is not None and cur_wc >= max_wc:
                reasons.append(KillReason("budget",
                                          f"runtime {cur_wc}s >= max {max_wc}s"))
        else:
            raise ValueError(f"unknown kill-rule type: {kind!r}")

    return KillVerdict(kill=bool(reasons), reasons=reasons, latest=latest)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/core/campaign/test_killrules.py -v`
Expected: PASS (9 tests).

- [ ] **Step 6: Commit**

```bash
git add core/campaign/__init__.py core/campaign/killrules.py \
        tests/core/campaign/__init__.py tests/core/campaign/test_killrules.py
git commit -S -m "$(printf 'feat(campaign): deterministic kill-rule evaluator\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: W&B telemetry reader

**Files:**
- Create: `core/campaign/telemetry.py`
- Test: `tests/core/campaign/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/campaign/test_telemetry.py`:

```python
"""Telemetry reader — uses an injected api_factory, never the network."""
from __future__ import annotations

from core.campaign.telemetry import read_telemetry


class _FakeRun:
    def __init__(self):
        self.summary = {"eval/success_rate": 0.42, "_step": 5_000_000}

    def history(self, keys=None, pandas=True):
        assert pandas is False, "telemetry must request pandas=False"
        return [
            {"_step": 1_000_000, "_runtime": 60.0, "eval/success_rate": 0.20},
            {"_step": 2_000_000, "_runtime": 120.0, "eval/success_rate": 0.42},
        ]


class _FakeApi:
    def __init__(self):
        self.requested = None

    def run(self, path):
        self.requested = path
        return _FakeRun()


def test_read_telemetry_returns_rows_and_summary():
    fake = _FakeApi()
    tel = read_telemetry("ent/proj/abc123", api_factory=lambda: fake)
    assert fake.requested == "ent/proj/abc123"
    assert len(tel.rows) == 2
    assert tel.rows[-1]["eval/success_rate"] == 0.42
    assert tel.summary["_step"] == 5_000_000


def test_read_telemetry_rows_are_plain_dicts():
    tel = read_telemetry("ent/proj/x", api_factory=lambda: _FakeApi())
    assert all(isinstance(r, dict) for r in tel.rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/campaign/test_telemetry.py -v`
Expected: FAIL — `ModuleNotFoundError: core.campaign.telemetry`.

- [ ] **Step 3: Implement the reader**

Create `core/campaign/telemetry.py`:

```python
"""Read W&B run telemetry via wandb.Api.

The only module in core.campaign that touches the network.  wandb.Api is
injected through `api_factory` so unit tests pass a fake and never hit
the wire (WANDB_MODE=disabled does not stub Api reads).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Row = dict[str, Any]

DEFAULT_KEYS: tuple[str, ...] = (
    "eval/success_rate",
    "rollout/ep_rew_mean",
    "train/loss",
    "_step",
    "_runtime",
)


@dataclass(frozen=True)
class CampaignTelemetry:
    rows: list[Row]
    summary: dict[str, Any]


def read_telemetry(
    run_path: str,
    *,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    api_factory: Callable[[], Any] | None = None,
) -> CampaignTelemetry:
    """Fetch history rows + summary for a W&B run.

    run_path: "entity/project/run_id" (or the shortest unambiguous form).
    api_factory: zero-arg callable returning a wandb.Api-like object;
                 defaults to wandb.Api (imported lazily).
    """
    if api_factory is None:
        import wandb

        api_factory = wandb.Api
    api = api_factory()
    run = api.run(run_path)
    rows = [dict(r) for r in run.history(keys=list(keys), pandas=False)]
    summary = dict(run.summary)
    return CampaignTelemetry(rows=rows, summary=summary)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/campaign/test_telemetry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add core/campaign/telemetry.py tests/core/campaign/test_telemetry.py
git commit -S -m "$(printf 'feat(campaign): W&B telemetry reader with injectable api\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: `dsim campaign-status` command

**Files:**
- Create: `dsim/commands/campaign_status.py`
- Modify: `dsim/cli.py`
- Test: `tests/dsim/test_campaign_status_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dsim/test_campaign_status_cli.py`:

```python
"""dsim campaign-status — JSON verdict; telemetry monkeypatched (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from core.campaign.telemetry import CampaignTelemetry
from dsim.cli import app
from dsim.commands import campaign_status as cmd


def _patch_telemetry(monkeypatch, rows, summary=None):
    def fake(run_path, **kwargs):
        return CampaignTelemetry(rows=rows, summary=summary or {})
    monkeypatch.setattr(cmd, "read_telemetry", fake)


def test_status_no_rules_reports_no_kill(monkeypatch):
    _patch_telemetry(monkeypatch,
                     [{"_step": 100, "_runtime": 1.0, "eval/success_rate": 0.5}])
    result = CliRunner().invoke(app, ["campaign-status", "ent/proj/x"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["kill"] is False
    assert out["latest"]["eval/success_rate"] == 0.5


def test_status_with_rules_reports_kill(monkeypatch, tmp_path: Path):
    _patch_telemetry(monkeypatch,
                     [{"_step": 3_000_000, "_runtime": 9.0,
                       "eval/success_rate": 0.04}])
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [
        {"type": "floor_at_step", "metric": "eval/success_rate",
         "min": 0.10, "at_step": 3_000_000}]}))
    result = CliRunner().invoke(
        app, ["campaign-status", "ent/proj/x", "--rules", str(rules)])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["kill"] is True
    assert out["reasons"][0]["rule"] == "floor_at_step"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dsim/test_campaign_status_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: dsim.commands.campaign_status`.

- [ ] **Step 3: Implement the command**

Create `dsim/commands/campaign_status.py`:

```python
"""dsim campaign-status — read-only W&B telemetry + kill-rule verdict as JSON.

Used by the run-campaign controller skill as the deterministic, cheap
monitor tick.  Prints a JSON object: latest metrics, kill bool, reasons,
and the run summary.  Never mutates anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from core.campaign.killrules import evaluate
from core.campaign.telemetry import read_telemetry


def run(
    wandb_run: str = typer.Argument(
        ..., help="W&B run path (entity/project/run_id)"),
    rules: Path = typer.Option(
        None, "--rules", help="JSON file of kill-rules (see campaign spec)"),
) -> None:
    """Print latest telemetry + kill-rule verdict for a W&B run as JSON."""
    tel = read_telemetry(wandb_run)
    rule_list: list[dict] = []
    baselines: dict[str, list] = {}
    if rules is not None:
        spec = json.loads(rules.read_text())
        rule_list = spec.get("rules", [])
        for b in spec.get("baseline_runs", []):
            baselines[b] = read_telemetry(b).rows
    verdict = evaluate(tel.rows, rule_list, baselines=baselines)
    typer.echo(json.dumps({
        "wandb_run": wandb_run,
        "kill": verdict.kill,
        "reasons": [{"rule": r.rule, "detail": r.detail} for r in verdict.reasons],
        "latest": verdict.latest,
        "summary": tel.summary,
    }, indent=2, default=str))
```

- [ ] **Step 4: Register the command in the CLI**

In `dsim/cli.py`, add the import alongside the other `from dsim.commands import …` lines:

```python
from dsim.commands import campaign_status as _campaign_status_cmd
```

And add the registration alongside the other `app.command(...)` lines:

```python
app.command(name="campaign-status")(_campaign_status_cmd.run)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/dsim/test_campaign_status_cli.py tests/dsim/test_cli_help_smoke.py -v`
Expected: PASS (existing help-smoke test still green; 2 new tests pass).

- [ ] **Step 6: Commit**

```bash
git add dsim/commands/campaign_status.py dsim/cli.py tests/dsim/test_campaign_status_cli.py
git commit -S -m "$(printf 'feat(campaign): dsim campaign-status command\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Campaign directory templates

**Files:**
- Create: `docs/campaigns/TEMPLATE/goal.md`
- Create: `docs/campaigns/TEMPLATE/ledger.md`
- Create: `docs/campaigns/TEMPLATE/frontier.md`

- [ ] **Step 1: Create `docs/campaigns/TEMPLATE/goal.md`**

```markdown
# Campaign Goal: <name>

**Objective:** <metric> <comparator> <target>
(e.g. `eval/success_rate` (Blue vs league) > 0.50, judged on honest success
rate de-noised over ≥100 episodes)

**Mode:** interactive | autonomous
**autonomous_allow_code:** true | false   # false => config-only when autonomous

**Budget:**
- max_iterations: <N>
- max_wallclock: <e.g. 8h>
- per_run_step_cap: <e.g. 10_000_000>

**Levers in scope:**
- <reward terms / weights>
- <curriculum knobs>
- <obs blocks>
- <opponent mixture>
- code changes in scope: yes | no

**Baselines** (for collapse-vs-baseline rules):
- <label>: <entity/project/run_id>

**Out of scope / do not touch:**
- <constraints>
```

- [ ] **Step 2: Create `docs/campaigns/TEMPLATE/ledger.md`**

```markdown
# Campaign Ledger: <name>

> Append-only. One report card per iteration, newest at the bottom.

<!-- REPORT CARD TEMPLATE — copy per iteration
## Iteration <N> — <timestamp>

- **W&B run:** <entity/project/run_id> — <url>
- **Hypothesis:** <what this tested and why>
- **Config diff:** <experiment YAML / Hydra overrides; code diff ref if any>
- **Kill-rules:** <declared abort conditions for this run>
- **Outcome:** completed | killed(<rule/judge reason>) | error
- **Key metrics:** honest success_rate <best/final>, mean_reward <…>, step <…>
- **Verdict vs frontier:** better | worse | null (de-noised)
- **Learnings:** <durable takeaways feeding the next design>
-->
```

- [ ] **Step 3: Create `docs/campaigns/TEMPLATE/frontier.md`**

```markdown
# Campaign Frontier: <name>

> Rewritten each iteration. The current state of knowledge.

## Current best
- <run id> — <metric value> — <one-line why>

## Dead ends (do not retry)
- <approach> — <why it failed>

## Open hypothesis queue (ranked)
1. <next experiment> — <expected effect> — needs-code: yes/no
2. <…>
```

- [ ] **Step 4: Commit**

```bash
git add docs/campaigns/TEMPLATE/
git commit -S -m "$(printf 'docs(campaign): ledger/frontier/goal templates\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: `experiment-monitor` worker agent

**Files:**
- Create: `.claude/agents/experiment-monitor.md`

- [ ] **Step 1: Create the agent definition**

Create `.claude/agents/experiment-monitor.md`:

```markdown
---
name: experiment-monitor
description: Gray-zone early-stop judge for a running RL experiment. Reads recent W&B telemetry and returns a continue/kill verdict for cases the deterministic hard rules cannot decide (slow plateau, dead learning signal, suspicious dynamics). Dispatched by the run-campaign controller, never for hard-rule kills.
tools: Bash, Read
model: sonnet
---

You judge whether a *running* RL training experiment is doomed and should be
killed early. The cheap, deterministic hard rules (NaN, success-rate floor at a
step gate, collapse-vs-baseline, budget) are already handled by
`dsim campaign-status` — you are invoked ONLY for the gray-zone judgment calls.

## Inputs (in your dispatch prompt)
- The W&B run path (`entity/project/run_id`).
- The campaign objective metric + target.
- The experiment's hypothesis and declared kill-rules.
- The current best frontier value.

## Procedure
1. Run `dsim campaign-status <run_path>` to get the latest telemetry JSON.
   (Read-only; runs `uv run` under the hood.)
2. Judge against these project-specific lessons:
   - Judge defenders on honest `eval/success_rate`, de-noised over ≥100
     episodes — never length-confounded reward.
   - A flat/dead learning signal can mean a non-reactive opponent (the R4t
     dead-gradient lesson), not a fixable run — recommend kill if the gradient
     is dead and the design can't recover it.
   - A slow but real upward trend is NOT a kill; plateaus below the frontier
     after most of the step budget ARE.
3. Be conservative: when genuinely uncertain, prefer `continue` (let the hard
   budget rule end it) rather than killing a run that might still climb.

## Output (return as your final message — this IS the return value, raw JSON)
{"verdict": "continue" | "kill", "confidence": 0.0-1.0,
 "reason": "<one or two sentences>",
 "latest_success_rate": <number or null>}
```

- [ ] **Step 2: Verify frontmatter parses (covered by Task 9's test)**

No standalone run yet. Proceed.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/experiment-monitor.md
git commit -S -m "$(printf 'feat(campaign): experiment-monitor worker agent\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: `experiment-analyst` worker agent

**Files:**
- Create: `.claude/agents/experiment-analyst.md`

- [ ] **Step 1: Create the agent definition**

Create `.claude/agents/experiment-analyst.md`:

```markdown
---
name: experiment-analyst
description: Post-run analyst and next-experiment designer for an RL campaign. Reads a finished run's full W&B history plus the campaign ledger and frontier, writes a structured report card, and proposes and ranks the next 1-3 experiments with explicit kill-rules. Dispatched once per iteration by the run-campaign controller.
tools: Bash, Read, Grep, Glob
model: opus
---

You are a fresh-context RL research analyst — the "twin" the campaign dispatches
each iteration. Your job: turn one finished experiment into a report card and a
ranked set of next experiments.

## Inputs (in your dispatch prompt)
- The just-finished W&B run path and its outcome (completed/killed/error).
- The campaign directory path (`docs/campaigns/<name>/`).

## Procedure
1. Read `docs/campaigns/<name>/goal.md`, `ledger.md`, and `frontier.md` in full.
2. Run `dsim campaign-status <run_path>` for the latest metrics; read more W&B
   history detail if needed via the same command on related runs.
3. Inspect the experiment's config (the `conf/experiment/*.yaml` and any Hydra
   overrides; `.hydra/config.yaml` under the run dir if present).
4. Judge results on honest `eval/success_rate`, de-noised over ≥100 episodes.
   Compare against the frontier; classify better/worse/null.
5. Propose 1-3 next experiments. For EACH: the hypothesis, the concrete config
   diff (YAML keys / Hydra overrides) or code change needed, whether it needs
   code (`needs_code`), expected effect, and explicit kill-rules (using the
   kill-rule schema: nan / floor_at_step / collapse_vs_baseline / budget).
6. Do NOT repeat anything already in the ledger's dead-ends. Build on learnings.

## Output (return as your final message — this IS the return value, raw JSON)
{
  "report_card": {
    "wandb_run": "...", "hypothesis": "...", "outcome": "...",
    "key_metrics": {"success_rate": <n>, "mean_reward": <n>, "step": <n>},
    "verdict": "better|worse|null", "learnings": ["..."]
  },
  "proposals": [
    {"rank": 1, "hypothesis": "...", "config_diff": "...",
     "needs_code": false, "expected_effect": "...",
     "kill_rules": {"rules": [ ... ]}}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/experiment-analyst.md
git commit -S -m "$(printf 'feat(campaign): experiment-analyst worker agent\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: `experiment-coder` worker agent

**Files:**
- Create: `.claude/agents/experiment-coder.md`

- [ ] **Step 1: Create the agent definition**

Create `.claude/agents/experiment-coder.md`:

```markdown
---
name: experiment-coder
description: Makes a single small, test-backed code change for an RL experiment that config alone cannot express (a new reward term, an env/curriculum lever). Dispatched by the run-campaign controller only when a chosen proposal has needs_code=true. Leaves the suite green or reports failure; never launches training itself.
tools: Bash, Read, Edit, Write, Grep, Glob
model: opus
---

You implement ONE focused code change for an RL experiment and prove it with
tests. You do not launch training and you do not design experiments — the
controller does that.

## Inputs (in your dispatch prompt)
- The proposal: the change to make and why.
- The relevant files (reward stack, env, config schema).

## Procedure (test-driven)
1. Locate the pattern to extend (e.g. an existing reward term in the reward
   stack, an env lever) with Grep/Glob/Read. Follow existing conventions.
2. Write a failing unit test under `tests/<package>/<module>/test_<name>.py`
   that pins the new behavior's semantics (not just shape — assert the sign and
   magnitude of the reward/effect so a subtly wrong term is caught).
3. Run it: `uv run pytest <path> -v` — confirm it fails for the right reason.
4. Implement the minimal change. Add config-schema fields if needed.
5. Run the new test + `uv run make test-fast`. Both must be green.
6. If you cannot make it green, STOP and report the failure — do not weaken the
   test to pass.

## Output (return as your final message — this IS the return value, raw JSON)
{"status": "green" | "blocked",
 "files_changed": ["..."],
 "test_command": "uv run pytest ...",
 "summary": "<what changed>",
 "blocker": "<null or why it's blocked>"}

The controller will NOT launch the run unless status is "green".
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/experiment-coder.md
git commit -S -m "$(printf 'feat(campaign): experiment-coder worker agent\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: `run-campaign` controller skill

**Files:**
- Create: `.claude/skills/run-campaign/SKILL.md`

- [ ] **Step 1: Create the skill**

Create `.claude/skills/run-campaign/SKILL.md`:

```markdown
---
name: run-campaign
description: Run an RL experiment campaign loop — launch a training run, monitor W&B telemetry, early-stop bad runs (hard rules + model judgment), analyze results, design and run the next experiment, repeat. Use when the user wants to start, resume, or drive an experiment campaign (interactive or autonomous/overnight). Reads docs/campaigns/<name>/.
---

# Run Campaign

You are the durable **controller** of an RL experiment campaign. You own the
loop and the ledger; you offload heavy reading/reasoning to fresh worker
subagents so your own context stays lean across many iterations.

## Setup
1. Determine the campaign name. If `docs/campaigns/<name>/` does not exist, copy
   `docs/campaigns/TEMPLATE/` to it and help the user fill `goal.md` (objective,
   mode, budget, levers, baselines). Do not proceed until `goal.md` is complete.
2. Read `goal.md`, `ledger.md`, `frontier.md` in full. Resolve `mode` and
   `autonomous_allow_code`.

## The loop (one iteration)
1. **Choose the experiment.**
   - First iteration: derive it from `goal.md` + `frontier.md`.
   - Later: take the top-ranked proposal from the last analyst output.
   - In **interactive** mode, present the chosen experiment + its kill-rules and
     WAIT for the user to approve / edit / replace it.
   - In **autonomous** mode, take the top-ranked proposal automatically.
2. **Guarded code (if `needs_code`).**
   - If autonomous and `autonomous_allow_code` is false: park this proposal in
     `frontier.md` tagged `needs-code — review required`, fall through to the
     next config-only proposal (or pause if none).
   - Otherwise dispatch `experiment-coder`. If it returns status != "green", do
     NOT launch; record the blocker in the ledger and pick the next proposal.
3. **Write the experiment config** under `conf/experiment/<exp>.yaml` (or assemble
   Hydra overrides) and the kill-rules JSON under
   `docs/campaigns/<name>/rules/<exp>.json`.
4. **Pre-flight:** run `uv run dsim obs-preflight` for the config. Abort the
   iteration on failure (record why).
5. **Launch** training in the background:
   `uv run make train EXP=<exp>` (or `uv run python -m scripts.train …`),
   started with run_in_background. Capture the W&B run path from its early
   output.
6. **Monitor.** On a cadence (ScheduleWakeup, e.g. every few minutes early, then
   coarser), run
   `uv run dsim campaign-status <run> --rules docs/campaigns/<name>/rules/<exp>.json`.
   - If `kill` is true → terminate the background run (record the reason).
   - At coarser intervals, or when telemetry looks ambiguous, dispatch
     `experiment-monitor`; kill if its verdict is "kill" with high confidence.
   - The harness re-wakes you when the background run exits on its own.
7. **Analyze.** When the run ends (completed, killed, or error), dispatch
   `experiment-analyst` with the run path + campaign dir. It returns the report
   card + ranked proposals.
8. **Record.** Append the report card to `ledger.md`; rewrite `frontier.md`
   (current best, dead-ends, ranked open queue).
9. **Milestone rollup.** On a frontier shift or campaign conclusion, write a
   concise rollup into the brain: `brain/changelog.md`, `brain/decisions.md`,
   `brain/models.md`, and `brain/index.md`. Skip for routine iterations.
10. **Continue or stop.**
    - **interactive:** present the report card + ranked proposals; WAIT for the
      user's go/no-go on the next experiment.
    - **autonomous:** check stop conditions (budget exhausted, target met, or N
      consecutive non-improving "dry" iterations). If none, schedule the next
      iteration via ScheduleWakeup and loop. The host must stay awake; remind
      the user to use `caffeinate` for overnight runs.

## Rules
- One ledger writer: you. Workers read; only you append/rewrite campaign files.
- Never kill a run on a telemetry-read failure alone — skip the tick, retry.
- If a worker returns malformed output, retry once; then (interactive) surface
  to the user or (autonomous) log it and pause — never silently guess.
- Judge results on honest `eval/success_rate` de-noised over ≥100 episodes.
- Commit campaign-file updates (`docs/campaigns/<name>/`) at milestones; they are
  version-controlled, not gitignored.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/run-campaign/SKILL.md
git commit -S -m "$(printf 'feat(campaign): run-campaign controller skill\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 9: Agent-asset validation test

**Files:**
- Create: `tests/test_agent_assets.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_assets.py`:

```python
"""Validate the campaign agent layer: frontmatter + templates exist & parse."""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["experiment-monitor", "experiment-analyst", "experiment-coder"]


def _frontmatter(md_path: Path) -> dict:
    text = md_path.read_text()
    assert text.startswith("---\n"), f"{md_path} missing frontmatter"
    _, fm, _ = text.split("---\n", 2)
    return yaml.safe_load(fm) if yaml else {"raw": fm}


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
@pytest.mark.parametrize("name", AGENTS)
def test_agent_frontmatter_has_name_and_description(name):
    fm = _frontmatter(ROOT / ".claude" / "agents" / f"{name}.md")
    assert fm["name"] == name
    assert fm["description"].strip()


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
def test_controller_skill_frontmatter():
    fm = _frontmatter(ROOT / ".claude" / "skills" / "run-campaign" / "SKILL.md")
    assert fm["name"] == "run-campaign"
    assert fm["description"].strip()


def test_campaign_templates_exist():
    tmpl = ROOT / "docs" / "campaigns" / "TEMPLATE"
    for fname in ("goal.md", "ledger.md", "frontier.md"):
        assert (tmpl / fname).is_file(), f"missing template {fname}"
```

- [ ] **Step 2: Run test to verify it passes**

(Tasks 4–8 created the files, so this test should pass on first run — it guards
against later drift.)
Run: `uv run pytest tests/test_agent_assets.py -v`
Expected: PASS. If `pyyaml` is absent, frontmatter tests skip and the template
test still runs.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_assets.py
git commit -S -m "$(printf 'test(campaign): validate agent/skill frontmatter + templates\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 10: Document the CLI surface

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the CLI surface section**

Run: `grep -n "campaign-status\|obs-preflight\|## CLI surface\|inventory" README.md | head`
Locate the `dsim` subcommand list in the "CLI surface" section.

- [ ] **Step 2: Add the campaign-status entry**

Add a bullet to the `dsim` inspection list, matching the surrounding format:

```markdown
- `dsim campaign-status <wandb-run> [--rules rules.json]` — read-only W&B
  telemetry + kill-rule verdict as JSON; the deterministic monitor tick used by
  the `run-campaign` controller skill.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -S -m "$(printf 'docs: document dsim campaign-status in CLI surface\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 11: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the campaign unit tests + CLI help smoke**

Run: `uv run pytest tests/core/campaign tests/dsim/test_campaign_status_cli.py tests/dsim/test_cli_help_smoke.py tests/test_agent_assets.py -v`
Expected: all PASS.

- [ ] **Step 2: Run the fast suite to confirm no regressions**

Run: `uv run make test-fast`
Expected: PASS (no new failures introduced; pre-existing macOS-render skips are
expected per brain Known Issues).

- [ ] **Step 3: Smoke the new command end to end (help only, no network)**

Run: `uv run dsim campaign-status --help`
Expected: usage text showing the `wandb_run` argument and `--rules` option.

- [ ] **Step 4: Final commit if any verification fixups were needed**

```bash
git add -A
git commit -S -m "$(printf 'chore(campaign): verification fixups\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

(Skip this commit if Steps 1–3 were clean.)

---

## Manual validation (human, after the plan)

The loop logic itself (skill + workers + background-task orchestration) is not
unit-testable; validate it with one short real campaign:

1. Create `docs/campaigns/smoke/` from the template with `mode: interactive`,
   `per_run_step_cap: 200_000`, objective on a quick canary.
2. Invoke the `run-campaign` skill; confirm: it launches a background run,
   `dsim campaign-status` reports telemetry, a kill-rule can stop a run, the
   `experiment-analyst` produces a report card, and the ledger/frontier are
   written and committed.
3. Per the project rule **verify behavior before committing the final merge** —
   confirm the real loop with the user before merging `feature/experiment-campaign`
   into `develop`.

---

## Self-Review

**Spec coverage:**
- Controller skill + worker subagents → Tasks 5–8. ✓
- Two modes (interactive/autonomous) + autonomous-code knob → Task 8 skill. ✓
- Early-stop rules + judgment → Tasks 1, 3 (rules) + Task 5 (judgment). ✓
- Telemetry helper → Tasks 2, 3. ✓
- Ledger + brain rollup → Tasks 4, 8. ✓
- Guarded code path → Task 7. ✓
- Campaign goal definition → Task 4 template + Task 8 setup. ✓
- Testing (unit + smoke + e2e) → Tasks 1–3, 9, 11 + manual validation. ✓
- Error handling (crash/unreachable/malformed/budget/red-tests) → Task 8 Rules. ✓

**Type consistency:** `KillVerdict`/`KillReason`/`evaluate` (Task 1) used by Task 3;
`CampaignTelemetry`/`read_telemetry` (Task 2) used by Task 3 and monkeypatched in
its test; kill-rule schema identical across Tasks 1, 3, 4, 6, 8. ✓

**Placeholder scan:** all code/content steps contain full content; the only
`<...>` are inside user-facing template files (intended fill-ins), not plan gaps. ✓
