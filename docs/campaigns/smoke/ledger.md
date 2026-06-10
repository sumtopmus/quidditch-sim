# Campaign Ledger: smoke

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

## Iteration 1 — 2026-06-10 12:48

- **W&B run:** `gridcom/drone-quidditch/_smoke_canary_20260610_124818` — https://wandb.ai/gridcom/drone-quidditch/runs/_smoke_canary_20260610_124818
- **Hypothesis:** train the single-agent fly-through-hoop policy 200k steps from
  scratch at the default LR (5e-5); expect `eval/success_rate` to climb off the
  floor as the policy learns to approach the hoop.
- **Config diff:** `conf/experiment/smoke_canary.yaml` (env=simple, obs=simple,
  reward=single_agent, init=scratch, curriculum=fixed_start, n_envs=4,
  total_timesteps=200k, lr=5e-5, eval_freq=50k, n_eval_episodes=20, video off).
  Two code fixes also shipped this iteration — see Learnings — in
  `envs/quidditch/simple_env.py` and `core/campaign/telemetry.py`.
- **Kill-rules:** `rules/smoke_canary.json` — nan(train/loss, rollout/ep_rew_mean);
  floor_at_step `eval/success_rate ≥ 0.02 @ 150k`; budget `max_step 200k /
  max_wallclock 1800s`.
- **Outcome:** completed (ran to natural 200k end). In a live controller loop it
  would have been early-stopped: the floor_at_step rule was breached at the 150k
  eval (success_rate 0.0 < 0.02). It only finished because 200k takes ~30s here,
  faster than a realistic monitor cadence.
- **Key metrics:** honest success_rate **0.0** (best=final), mean_reward ≈ **−20.29**,
  step **200000**. (Smoke: n_eval_episodes=20 — noise, not a real verdict.)
- **Verdict vs frontier:** **null** (first iteration; frontier empty; numbers are
  mechanics-validation noise).
- **Learnings:**
  - Loop mechanics validated end-to-end: scaffold → approve (interactive gate) →
    launch → monitor (healthy `kill:false` pre-150k) → forced abort
    (`kill:true`, `floor_at_step`) → analyze → record. Healthy→kill transition
    observed live on real telemetry (kill:false@110k → kill:true floor+budget@150k/200k).
  - **Defect #1 (fixed):** the single-agent simple env emitted `info['scored']`
    but not `info['is_success']`, so SB3's `EvalCallback` never logged
    `eval/success_rate` — the campaign objective. Fixed by mirroring
    `scored → is_success` in `simple_env.py` (+ regression tests).
  - **Defect #2 (fixed):** `core/campaign/telemetry.py` requested a never-logged
    key (`rollout/ep_rew_mean`) in one `history(keys=…)` call; wandb returns
    **0 rows if any requested key is absent**, silently blinding every kill-rule
    (vacuous `kill:false`). Fixed by fetching each key independently and merging
    by `_step` (+ regression test). The kill-rule loop is only trustworthy now
    that telemetry returns real rows AND the objective metric is logged.
  - Policy barely moved at lr 5e-5 (approx_kl ~1e-4, explained_variance ~0.20) →
    the next lever is the LR bump (queued).
