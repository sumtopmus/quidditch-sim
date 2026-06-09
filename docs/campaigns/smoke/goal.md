# Campaign Goal: smoke

> Purpose: a tiny end-to-end **validation** campaign for the run-campaign loop
> itself — not a real research goal. It exercises every stage (scaffold →
> preflight → launch → monitor/kill → analyze → ledger/frontier) on the cheapest
> single-agent canary so the mechanics can be verified before a real campaign.

**Objective:** `eval/success_rate` (single-agent fly-through-the-hoop) trends
upward and clears > 0.20, judged on honest success rate over the eval episodes.
(For a real campaign de-noise over ≥100 episodes; the smoke uses far fewer to
stay fast — see the runbook caveat.)

**Mode:** interactive
**autonomous_allow_code:** false   # config-only; no code changes in the smoke

**Budget:**
- max_iterations: 2
- max_wallclock: 1h
- per_run_step_cap: 200_000

**Levers in scope:**
- `trainer.lr`, `trainer.total_timesteps`
- `eval.eval_freq_steps`, `eval.n_eval_episodes`
- `curriculum` start (fixed vs random)
- code changes in scope: no

**Baselines** (for collapse-vs-baseline rules):
- none (smoke campaign; no baseline run)

**Out of scope / do not touch:**
- `conf/experiment/canary_single.yaml` and `canary_team.yaml` — pinned for the
  integration scoring tests; never edit (use `smoke_canary.yaml` instead).
- `brain/` rollups — skip for the smoke (no milestone worth recording).
- Don't reuse a real research W&B project; let the run log to the default
  project under a `_smoke_*` run name.
