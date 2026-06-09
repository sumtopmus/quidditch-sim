# Smoke-campaign runbook (manual validation)

A step-by-step to validate the `run-campaign` loop end-to-end on your own
machine, where you have **W&B auth + a display (or video disabled)**. Nothing in
this campaign was launched in the build container — it has no W&B credentials
and no OpenGL context, so the online `campaign-status` tick can't run there.

Goal: confirm every stage of the loop works once, cheaply, before pointing the
controller at a real research campaign.

---

## 0. Prerequisites (once)

```bash
uv sync                      # environment
wandb login                  # or: export WANDB_API_KEY=...   (online telemetry is required)
```

`dsim` invocation note: this repo sets `package = false` in `pyproject.toml`, so
there is **no bare `dsim` script**. Invoke it as a module everywhere:

```bash
uv run python -m dsim <cmd>          # e.g. uv run python -m dsim campaign-status --help
```

(The Makefile and the skill/agents use this form. Bare `dsim ...` only works if
you separately install the entry point, e.g. `uv tool install` from this repo.)

---

## 1. Start the controller

In Claude Code:

```
/run-campaign smoke
```

The controller will read `docs/campaigns/smoke/{goal,ledger,frontier}.md`,
resolve `mode: interactive` + `autonomous_allow_code: false`, and propose
iteration 1 from the frontier queue: **smoke_canary**. It will show the chosen
experiment + kill-rules and **wait for your approval**.

The experiment config and rules are already written, so you can approve as-is:
- config: `conf/experiment/smoke_canary.yaml` (single-agent, 200k steps, video off)
- rules:  `docs/campaigns/smoke/rules/smoke_canary.json`

---

## 2. Pre-flight (expected: skipped)

`smoke_canary` uses `init: scratch` — there is no parent checkpoint, so the
`obs-preflight` step does not apply (it checks parent↔child obs-spec compat for
`pretrain`/`warm_start`). The controller should note this and move on. Preflight
gets exercised later when you run a real `pretrain` campaign.

---

## 3. Launch training (background)

```bash
uv run make train EXP=smoke_canary
# or:  uv run python -m scripts.train +experiment=smoke_canary
```

From the early stdout, capture the **W&B run path**: `entity/project/run_id`
(printed as the "View run at ..." link; project defaults to `drone-quidditch`,
run name `_smoke_canary_<timestamp>`). The controller needs this for the monitor
tick. On a GPU box 200k steps is a few minutes; on CPU, longer — shrink
`trainer.total_timesteps` if needed.

---

## 4. Monitor tick (the deterministic check)

```bash
uv run python -m dsim campaign-status <entity/project/run_id> \
    --rules docs/campaigns/smoke/rules/smoke_canary.json
```

Expect a JSON object with `latest` metrics, `kill: false` (healthy run),
`reasons: []`, and the run `summary`. Re-run on a cadence; once `_step` reaches
200k the `budget` rule flips `kill: true` (which coincides with the run's natural
end — that's fine; it validates the budget rule fired).

### 4a. Prove the kill path on demand

Point the same command at the **force-kill** rules file, which has an
impossibly high success-rate floor gated at the first eval:

```bash
uv run python -m dsim campaign-status <entity/project/run_id> \
    --rules docs/campaigns/smoke/rules/smoke_canary_forcekill.json
```

Once at least one `eval/success_rate` point at `_step >= 50k` is logged, expect
`kill: true` and `reasons[0].rule == "floor_at_step"`. In a live loop the
controller would terminate the background run here and record the reason.

---

## 5. Analyze

When the run ends (completed/killed/error), the controller dispatches the
`experiment-analyst` subagent with the run path + `docs/campaigns/smoke/`. It
returns a report card + ranked proposals as JSON. (The analyst also calls
`campaign-status`, so it needs the same W&B auth.)

---

## 6. Record + commit

The controller (only the controller writes campaign files):
- appends the iteration-1 report card to `docs/campaigns/smoke/ledger.md`,
- rewrites `docs/campaigns/smoke/frontier.md` (current best / dead-ends / queue),
- commits `docs/campaigns/smoke/`.

For the smoke, skip the `brain/` milestone rollup (goal.md says so).

---

## Acceptance checklist (the loop is validated when all are true)

- [ ] Controller scaffolds/reads the campaign and proposes iteration 1, waiting
      for approval (interactive gate works).
- [ ] `make train EXP=smoke_canary` launches in the background and a W&B run path
      is captured.
- [ ] `campaign-status ... --rules smoke_canary.json` returns telemetry JSON with
      `kill: false` on a healthy run.
- [ ] `campaign-status ... --rules smoke_canary_forcekill.json` returns
      `kill: true` with a `floor_at_step` reason (kill path proven).
- [ ] On run end, `experiment-analyst` returns a well-formed report card + ranked
      proposals.
- [ ] Controller appends the report card to `ledger.md`, rewrites `frontier.md`,
      and commits `docs/campaigns/smoke/`.
- [ ] Second iteration is offered (or the campaign stops on budget) — your choice
      to continue or end the smoke here.

---

## Caveats (smoke only — don't carry into a real campaign)

- `n_eval_episodes: 20` is far below the ≥100-episode de-noising bar; smoke
  success-rate numbers are noisy and not a real verdict.
- The force-kill rules file is a validation prop, not a judgment rule.
- Video is disabled; you won't get episode clips for this run.

## Cleanup

The smoke W&B run can be deleted from the W&B UI afterward. To reset the campaign
for a re-run, restore the empty `ledger.md`/`frontier.md` from
`docs/campaigns/TEMPLATE/` (or `git checkout` them) and delete the smoke run dir
under `runs/_smoke_canary_*`.
