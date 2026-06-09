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
