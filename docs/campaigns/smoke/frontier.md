# Campaign Frontier: smoke

> Rewritten each iteration. The current state of knowledge.

## Current best
- (none — smoke is validation-only; iteration 1 `eval/success_rate` is
  identically 0.0, not a real policy. The value of this campaign was proving the
  loop mechanics + surfacing/fixing two monitoring-path defects, not a number.)

## Dead ends (do not retry)
- (none)

## Open hypothesis queue (ranked)
1. **smoke_canary_lr** — iteration 1 was flat (success_rate 0.0, approx_kl ~1e-4,
   explained_variance ~0.20: the policy barely moved at lr 5e-5). Bump
   `trainer.lr` 5e-5 → 1e-3 to confirm the learning signal is live; expect
   success_rate to lift off 0.0 by the 150k eval. Kill-rules: nan(train/loss);
   floor_at_step `eval/success_rate ≥ 0.02 @ 150k`; budget `max_step 200k`. —
   needs-code: no
2. **smoke_canary_lr_long** — lr 1e-3 + `trainer.total_timesteps` 200k → 300k for
   more optimization budget. ONLY if the campaign relaxes `per_run_step_cap`
   (currently 200k in goal.md); otherwise keep at 200k. — needs-code: no

## Loop-validation status (the smoke's actual purpose)
- ✅ Interactive approval gate, launch, healthy monitor tick (`kill:false`),
  forced-abort path (`kill:true` / `floor_at_step`), analyst report card,
  ledger/frontier write — all exercised end-to-end on real W&B telemetry.
- ✅ Two real defects found & fixed (env `is_success`; telemetry per-key fetch),
  each with regression tests. See ledger iteration 1.
