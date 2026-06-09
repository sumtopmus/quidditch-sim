# Campaign Frontier: smoke

> Rewritten each iteration. The current state of knowledge.

## Current best
- (none yet — first iteration not run)

## Dead ends (do not retry)
- (none yet)

## Open hypothesis queue (ranked)
1. **smoke_canary** — train the single-agent fly-through-hoop policy for 200k
   steps from scratch at the default LR; expect `eval/success_rate` to climb off
   the floor as the policy learns to approach the hoop. — needs-code: no
2. **smoke_canary_lr** — if iteration 1 is flat, bump `trainer.lr` (e.g. 1e-3)
   to confirm the learning signal is live. — needs-code: no
