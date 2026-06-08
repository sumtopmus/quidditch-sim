# Experiment-Campaign Loop — Design

**Date:** 2026-06-08
**Branch:** `feature/experiment-campaign` (worktree off `develop`)
**Status:** Design approved; pending spec review → implementation plan.

## Problem

Running an RL experiment campaign by hand is a tight, repetitive loop: launch a
run, babysit the W&B curves, kill it if it's clearly going nowhere, read the
results, decide what to try next, launch that. The Blue-defender campaign
(concluded 2026-06-07 at a ~30–38% ceiling) showed how much of this is toil that
a model can do, and how easy it is to waste compute on runs that were doomed by
step 3M. We want an agent that owns this loop — runs experiments, monitors
telemetry, stops bad runs early, analyzes results, designs the next experiment,
and carries forward — so a human supervises *direction*, not mechanics.

## Goals

- One **campaign loop**: launch → monitor → early-stop → analyze → design-next →
  continue, driven by a stated goal.
- Two **modes**: *interactive* (stop at each report card for human go/no-go) and
  *autonomous* (self-pace through the loop unattended, e.g. overnight).
- **Early-stopping** that combines cheap deterministic kill-rules with coarser
  model judgment.
- **Accumulated memory** so each iteration builds on the last instead of
  repeating it, with a durable rollup into the project brain at milestones.
- Allow **config experiments and guarded, test-backed code changes**.

## Non-Goals

- Not a standalone Python research daemon and not a remote/cloud orchestrator —
  the substrate is the Claude Code agent layer (training is local MuJoCo/uv).
- Not a hyperparameter-sweep replacement. `dsim sweep` (W&B sweeps) still owns
  dense grid/random search; this loop does *reasoned* sequential design, and may
  *launch* a sweep as one experiment.
- Not a new training/eval stack — it drives the existing `make train` /
  `scripts.train` / `dsim` / W&B surfaces.

## Architecture — controller skill + worker subagents

Two roles mapped onto Claude Code primitives.

### Controller (a skill: `/run-campaign`)

The durable loop driver, invoked in the user's session; the thing the user talks
to in interactive mode. Responsibilities:

- Load `goal.md`; resolve mode and budget.
- Launch each training run as a **background process** (`run_in_background`
  Bash). The harness re-wakes the controller when the process exits (natural
  completion or kill); `ScheduleWakeup` provides mid-run monitor ticks.
- At each monitor tick: run the deterministic **hard kill-rules** (via the
  telemetry helper, §Telemetry); for gray-zone calls, dispatch the
  **monitor-judge** worker. Kill the background run if a rule or the judge says
  so.
- On run end: dispatch the **analyst-designer** worker to produce the report
  card + ranked next-experiment proposals.
- Decide the next experiment — *interactive*: present and wait for the user;
  *autonomous*: take the top-ranked proposal. If the proposal needs code,
  dispatch the **coder** worker (subject to the autonomous-code knob, §Modes).
- Append to the ledger; update `frontier.md`; loop until a stop condition.

The controller stays context-lean by **offloading heavy reading/reasoning to
fresh-context workers** — important for long campaigns. Each analysis/design is a
fresh worker instance: this is the user's "twin spawns twin," with a single
durable supervisor so one bad iteration can't derail the chain and there is one
ledger writer and one place to intervene.

### Workers (`.claude/agents/`)

Dispatched one fresh instance per job via the Agent tool.

- **`experiment-analyst`** — reads the full W&B history for the just-finished run
  together with the campaign ledger/frontier, writes a structured **report
  card**, and proposes & ranks the next 1–3 experiments. Returns structured
  output (schema below).
- **`experiment-monitor`** — invoked only for gray-zone kill decisions. Reads
  recent telemetry and returns a verdict (`continue` / `kill` + reason). Hard
  rules do *not* invoke a model.
- **`experiment-coder`** — only when the next experiment needs source changes.
  Runs the spec→TDD→implement→test flow inside the worktree; the run does not
  launch unless `make test-fast` is green and obs-preflight passes.

## Modes

A `mode` field in `goal.md`.

- **Interactive** — the loop runs and early-stops autonomously, then *stops at
  the report card*: the controller presents the analysis + ranked proposals and
  waits for the user to approve / edit / supply their own next experiment. Human
  drives direction.
- **Autonomous** — same loop, but the controller auto-selects the top-ranked
  proposal and continues, self-pacing via `ScheduleWakeup`, until a stop
  condition. Constraint: training is local, so the **host machine must stay
  awake** (e.g. `caffeinate`); there is no remote-cron path for the training
  itself.

**Stop conditions** (autonomous): budget exhausted (max iterations / wall-clock),
objective target met, or *N* consecutive iterations with no improvement
("dry"), or explicit user interrupt.

**Autonomous-code knob** (`autonomous_allow_code`, default `true` per approval):
when `false`, autonomous mode forbids source edits — any proposal needing code is
parked in `frontier.md` tagged `needs-code — review required` and the loop falls
through to the next config-only proposal (or pauses if none). When `true`
(default), the coder worker may run overnight, guarded only by tests-green +
obs-preflight. Available as a flag without redesign.

## Campaign definition — `goal.md`

Authored by the user (or dictated) at campaign start:

- **objective** — the metric and target, e.g. `eval/success_rate` (Blue vs
  league) `> 0.5`, judged on honest success rate de-noised over ≥100 episodes.
- **levers in scope** — which YAML knobs / reward terms / curriculum / obs
  blocks / opponent mixtures the agent may vary; whether code changes are in
  scope.
- **budget** — max iterations and/or wall-clock and/or cost; per-run step cap.
- **mode** — `interactive` | `autonomous`; `autonomous_allow_code`.
- **baselines** — reference run(s) for "reward collapse vs baseline" rules.

**First real campaign:** the RLlib self-play league — push Blue past the
~30–38% frozen-opponent ceiling via true co-adaptation.

## Early-stopping — rules + judgment

**Hard rules** (cheap, every monitor tick, deterministic, no model):

- `NaN` in loss/reward/grad (the gSDE failure mode).
- `eval/success_rate` below a declared floor at a declared step-gate
  (e.g. `< 0.10` at 3M steps).
- Reward / objective collapse below a baseline run's trajectory at matched steps.
- Wall-clock or step budget for the run exceeded.

Each experiment's report card commits to its specific kill-rules **up front**, so
the monitor is just evaluating declared, auditable conditions.

**Model judgment** (coarser ticks, `experiment-monitor` worker): slow plateaus,
suspicious dynamics, dead learning signal (the R4t dead-gradient lesson —
non-reactive opponent ⇒ flat gradient). Always judged on **honest
`success_rate`, de-noised over ≥100 episodes**, never length-confounded reward.

## State — ledger + brain rollup

Self-contained campaign directory in the worktree, **committed at milestones**
(not gitignored), mirroring the retrospective convention:

```
docs/campaigns/<name>/
  goal.md        # objective, levers, budget, mode, baselines
  ledger.md      # append-only report cards, one per iteration
  frontier.md    # current best + dead-ends + open-hypothesis queue (rewritten each iter)
```

Each twin reads the whole directory on startup. **Brain rollup only at
milestones** (frontier shift, campaign conclusion) → `brain/changelog.md`,
`brain/decisions.md`, `brain/models.md`, `brain/index.md`. Fast experiment churn
stays out of curated long-term memory.

### Report-card schema (per iteration, in `ledger.md`)

- `iteration`, `timestamp`, `wandb_run_id` / URL
- `hypothesis` — what this experiment tested and why
- `config_diff` — the experiment YAML / Hydra overrides (and code diff ref, if any)
- `kill_rules` — the declared abort conditions for this run
- `outcome` — `completed` | `killed(<rule/judge reason>)` | `error`
- `key_metrics` — final/best honest `success_rate`, mean reward, step reached
- `verdict` — better / worse / null vs frontier, de-noised
- `learnings` — durable takeaways feeding the next design

## Glue code

Minimal, in the Claude Code layer (no daemon), all read-only or thin:

- **`dsim campaign-status <wandb-run>`** — read-only helper that queries
  `wandb.Api` and prints the latest history rows + summary as JSON, so hard-rule
  checks are deterministic and cheap. Pure function over W&B history;
  unit-testable against fake history.

Everything else reuses existing surfaces (`make train`, `scripts.train`, `dsim`,
`wandb.Api`).

## Error handling

- **Training process crashes** (non-zero exit, not a kill): captured as
  `outcome: error`; the analyst sees the tail of stderr and treats it as a
  failed experiment (propose a fix or move on), not a loop-fatal event.
- **W&B unreachable at a monitor tick**: skip the tick, retry next wake-up; do
  not kill a run on telemetry-read failure alone.
- **Worker returns malformed/empty output**: controller retries once; on second
  failure, in interactive mode it surfaces to the user, in autonomous mode it
  logs to the ledger and pauses the campaign (does not silently guess).
- **Budget reached mid-run**: let the current run finish its monitor cycle or
  hit its step cap; do not orphan a background process.
- **Coder worker leaves tests red**: the run never launches; the proposal is
  marked blocked in `frontier.md`.

## Testing

- Unit tests for `dsim campaign-status` and the kill-rule evaluator — pure
  functions over synthetic W&B history (NaN, floor-at-gate, collapse-vs-baseline,
  budget). Offline (`WANDB_MODE=disabled`).
- Smoke test that the skill + agent definitions load and dispatch (dry-run, no
  real training).
- End-to-end validation: one short real campaign with a tiny per-run step budget
  in interactive mode, confirming the full loop (launch → monitor → report →
  approve → next) and the ledger/frontier writes.

## Open questions (resolved)

- Ledger committed, not gitignored. ✓ (user)
- Code edits allowed in both modes; config-only-when-autonomous available as an
  off-by-default knob. ✓ (user)
