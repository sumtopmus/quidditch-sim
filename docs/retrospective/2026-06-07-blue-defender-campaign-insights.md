# Blue Defender Campaign — Insights Retrospective

**Scope:** Everything learned while trying to train a Blue *defender* that prevents the trained Red *attacker* (R4t) from scoring, in the 1v1 MuJoCo drone-quidditch env. Covers the pre-session state (`HANDOFF.md`) and the 2026-06-05 → 06-07 experiment session (the `blue_sde_*` / `blue_std_*` runs).

**Audience:** an agent or developer picking this up cold. Read `HANDOFF.md` for the Red lineage and env-gotcha detail; this doc is the *why-it's-hard* and *what-we-learned* record.

> **Note (preserved on `develop`).** The work this retrospective describes lived on the experimental branch `exp/retraining`, which was **abandoned**. Its training runs (`runs/...`), uncommitted env/reward/config changes, and the `SuccessRateEvalCallback` commit (`7b644c3`) did **not** merge to `develop` — this document is the surviving record of the campaign's findings. Run-dir and `team_env.py`/config paths below refer to that abandoned branch and won't exist on `develop`; treat them as historical references, not live artifacts.

---

## TL;DR

- **Red is solved.** `R4t` = `runs/red_team_random/20260604_152427/best_model.zip` scores from ~75% of random starts. It is the frozen opponent every Blue run trains against. Obs spec **must** be `DUEL_V1_BODY` (22-d, n_stack=1).
- **Blue beats a predictable attacker but not the trained one.** Standard-policy ladder reaches **100%** prevention vs a fixed-start beeline Red and **~80%** vs a random-start beeline. Against the full-speed **R4t it robustly caps at ~30–38% honest prevention**, and *nothing cheap moves it past there.*
- **The single most important correction this session:** **R4t was trained only to *score* against a *hover* Blue — it is NOT reactive to a defending Blue.** So the ceiling was never "an evasive adversary"; it was a **dead learning signal** (pure-R4t episodes are mostly losses → no gradient) plus a hard interception/physical limit.
- **What broke the dead-signal wall:** a **beeline + R4t mixture** opponent (fictitious-curriculum bridge). It restored a live gradient and produced the first *stable, genuinely-learned* defender vs R4t (32% objective, no erosion), up from a noisy ~25% that never learned.
- **What did NOT break the ~30–38% ceiling:** policy type (gSDE vs standard), reward density (dense/sparse/high/**positional shot-denial**), start geometry, take-down threshold, entropy, opponent speed, mixture ratio, full dense→sparse anneal. The ceiling looks like a property of the **observation + drone dynamics**, not reward/curriculum.
- **Net deliverables:** an honest, hack-free defender that stops ~1/3 of a 75% scorer; a committed `SuccessRateEvalCallback`; two TDD'd env curriculum levers (`red_start_x_max`, `red_action_scale`); a working mixture-bridge recipe; and a clear case that the next gain requires a *structural* change (self-play, richer/predictive obs, or a recurrent policy).

---

## 1. The setup

- **Env:** `QuidditchTeamEnv` (PettingZoo ParallelEnv) → `OpponentControlledEnv` (SB3 single-learner shim). Red attacker tries to fly through the hoop at `(2, 0, 2)`; Blue defender starts at `(1, 0, 1.5)` facing −x. Arena radius 3 m. Action = normalized delta-setpoint `[dx, dy, dyaw, dz]`.
- **Learner policy:** SB3 PPO, MLP, `DUEL_V3_BODY_EGO` obs (25-d body-frame ego: `vec_to_goal`, `vec_to_hoop`, `opp_pos_rel`, `opp_vel_rel` rotated into Blue's body frame, world `lin_pos` anchor, `closing_rate`), `n_stack=3` (75-d).
- **Opponent (R4t):** a frozen PPO checkpoint fed its native `DUEL_V1_BODY` (22-d) obs through `OpponentControlledEnv`.
- **The objective:** Blue prevents R4t from scoring, at **random start, full speed, full disc**. Measured by `eval/success_rate` = honest prevention (Red didn't score **AND** Blue didn't self-crash).

## 2. Red lineage (solved) — for context

Training Red *from scratch in the team env collapsed* (the hovering Blue sits in Red's scoring lane and the TakeDown penalty drives Red to −20). The fix that worked is **simple-env-first then warm-start into the team env**:

`R1s` simple/fixed → `R2s` simple/random → `R3t` team/fixed (warm-start 16→22, hover Blue) → **`R4t` team/random** (hover Blue). lr 5e-5, ~10M/stage. R4t scores ~75%. This is the project's proven-stable Red recipe; don't retrain Red in the team env from scratch.

## 3. The Blue defender problem (the hard part)

R4t is a strong, fast, *non-reactive* scorer that flies a learned trajectory to the hoop. Defending it is genuinely hard, and the campaign repeatedly hit reward-hacks and learning failures before this session, then a hard performance ceiling during it.

### 3.1 Pre-session failure modes (from `HANDOFF.md`)

- **Stalling (reward-hack):** dense per-step shaping (`team_v4_cone`, ~0.1/step × long episodes ≈ +200) dwarfs the −10 score concession → Blue learns to *delay and farm shaping*, not prevent (37% prevention).
- **Suicide (reward-hack):** an early `TerminalScoreOutcome` rewarded *any* episode end without a Red score → Blue learned to **self-crash at ~step 197** for the bonus. Fixed by treating a defender self-crash as a concession (`defender_crash_flags`) + an honest `is_success` that excludes it.
- **Eval reward is length-confounded** (corr ≈ 0.88 with episode length) → judge Blue by `eval/success_rate`, never reward.
- **`best_model` (reward-selected) ≠ best defender** → a high-reward staller gets saved.
- **gSDE attempt:** introduced to fix an action-std blow-up; it bounded σ but the gentle squashed actions were too soft to intercept (0%) and it NaN-crashed.

## 4. This session's experiment arc (chronological)

### 4.1 gSDE: validated high-shaping, then abandoned
- New reward `team_v8_prevention` (dense Blue shaping ×3 over v7, outcome anchors fixed). `blue_sde_1` (gSDE, beeline fixed) reached **100% prevention by 800k** — proving the earlier "gSDE 0%" was a *reward-too-weak* problem, not gSDE — then **NaN-crashed @2.7M**.
- `blue_sde_2` (gSDE, random beeline, v7): peaked **60% then degraded + NaN @7.8M** — *worse* than the non-sde baseline (~80%).
- **Conclusion: gSDE is a net negative here.** It underperforms the standard policy *and* NaN-crashes chronically (lowering lr only delays it: 2.7M @3e-4 → 7.8M @1e-4). The action-std blow-up it "fixes" is **cosmetic for the deployed deterministic-mean policy**. Dropped gSDE.

### 4.2 `SuccessRateEvalCallback` (committed `7b644c3`)
Selects `best_model` by the tuple `(eval/success_rate, mean_reward)` — honest prevention dominates, reward breaks ties — so a length-confounded staller can never be saved (closes the `best_model`≠best-defender hole). Degrades to pure-reward selection when the env has no `is_success` (single-agent). TDD: 7 decision-logic tests + a wiring test. **The only commit so far.**

### 4.3 Standard-policy ladder (the keeper path)
- `blue_std_1` (scratch, beeline fixed, v8): **rock-stable 100%** 1.4M→10M, no NaN.
- `blue_std_2` (pretrain, beeline random, v8): peak 100%, **~80% sustained** (volatile).
- The standard Gaussian policy learns fast/aggressively and never NaNs — strictly better than gSDE here.

### 4.4 The pure-R4t wall (rung 3, four takes)
`blue_std_3` vs frozen R4t was run four times, each adding a lever:

| take | lever added | sustained prevention |
|---|---|---|
| 1 | full-disc, sparse v6 | ~24% (peak 60% then **eroded to 20%**) |
| 2 | + left-half Red start | ~23% |
| 3 | + take-down thr 0.5 + `ent_coef`=0 | ~25% |
| 4 | + Red slowed 50% | ~24% |

**All plateaued ~25%, with no learning curve** (vs the clean 0→100% curves on beeline). The tell: beeline (non-reactive, predictable) is learnable; pure-R4t is not, under any of these knobs.

### 4.5 The correction that reframed everything
**R4t was trained only to score vs a *hover* Blue (`ZeroOpponent`) — it is not reactive to a defending Blue.** So the ~25% ceiling wasn't an evasive adversary dodging Blue; it was a **learning-signal problem**: pure-R4t episodes are mostly "you lost," giving PPO no gradient → the policy drifts around ~25–35%.

### 4.6 The mixture bridge (the unlock)
`blue_std_4_mixture_red`: opponent = **50/50 `beeline_red` + frozen R4t**. The beeline half (which Blue beats ~80%) keeps a live positive gradient while R4t episodes expose the target. Result: **stable learning, no erosion.** De-noised (50 eps each): beeline 66%, R4t@train-cond 48%, **R4t@objective (full speed, full disc) 32%.** First rung-3/4 attempt that *learned and held* vs R4t.

### 4.7 Annealing the bridge (plateau)
Progressively shift the mix toward R4t and Red toward full speed, pretraining each rung from the last:

| rung | opponent mix / Red speed / start | R4t objective (de-noised) |
|---|---|---|
| `std_4` | 0.5 beeline / 0.5 R4t, speed 0.5, left | 32% |
| `std_5` | 0.3 / 0.7, speed 0.75, left | **38% (best)** |
| `std_6` | 0.15 / 0.85, speed 1.0, left | 32% |
| `std_7` | 0.1 / 0.9, speed 1.0, **full disc** | 29% |

The objective **plateaued ~30–38%**; more anneal steps don't help.

### 4.8 Positional shot-denial reward (structural reward pivot — also plateaus)
`blue_std_8_positional` (`team_v9_positional`): make `GoalSideCone` (occupy the goal-side of the Red→hoop axis, *interpose* to deny the shot — no contact needed) the **primary dense signal** (scale 0.05 vs 0.0005 in v6) + stronger `HoopAnchor`. Single-variable change vs std_5. Result: **R4t objective 29%** — same ceiling. Changing *what* we reward (contact take-down → positional) doesn't break it either.

## 5. The robust ceiling: what was tried vs what moved it

Across the whole session the full-speed-R4t objective sat at **~30–38%**, regardless of:

| dimension | variants tried | effect on objective |
|---|---|---|
| Policy | gSDE, **standard** | gSDE worse + NaN; standard ≈ same ceiling |
| Reward | v4 dense, v6 sparse, v8 high, **v9 positional** | ±a few pts; none break it |
| Red start | full disc, left-half (x≤0) | ~none |
| Take-down threshold | 1.0, 0.5 | ~none |
| Entropy | 0.01, 0 | ~none (didn't even stop the erosion) |
| Opponent speed | 1.0, 0.75, 0.5 | ~none |
| Opponent mixture | pure R4t, 0.5/0.7/0.85/0.9 R4t | mixture *enabled learning*; ratio ~none on ceiling |
| Curriculum | full dense→sparse + opponent anneal | got to the ceiling, didn't exceed |

**Interpretation:** this is a ceiling of the **observation + drone dynamics**, not of reward shaping or curriculum. The behavioral signature throughout: Blue *delays* R4t (long episodes) but rarely lands the take-down contact / fails to physically block the small hoop in time.

## 6. Durable insights (the takeaways worth keeping)

1. **A non-reactive frozen opponent that mostly wins is a *dead* learning signal.** If most episodes are losses, PPO gets ~no gradient and the policy drifts/erodes (even with `ent_coef=0`). **Mix in a winnable opponent** (here: beeline) to keep a live gradient — fictitious-curriculum / league logic, cheaply realized with the existing `MixtureOpponent`. This was *the* unlock.
2. **Judge defenders by honest `eval/success_rate`, never reward.** Reward is length-confounded; a staller out-scores a decisive defender. Bake honesty into the *metric* (`is_success` excludes self-crash) *and* the *checkpoint selection* (`SuccessRateEvalCallback`).
3. **Eval at n=5 episodes is too noisy to read.** "Peak 60/80%" single evals are mostly noise; `best_model` gets chosen on them. **De-noise with 50–100 episodes and break the number down by opponent component** (beeline-only vs R4t-only, train-condition vs objective-condition). The training-time `success_rate` curve is a blend; the component eval is the real signal.
4. **gSDE was a net negative for this task.** It underperformed the standard policy *and* NaN-crashed chronically. The action-std blow-up it addresses is cosmetic for the deployed deterministic-mean policy — don't pay for a fix to a non-problem.
5. **Reward/curriculum knobs have a ceiling.** Once a defender is genuinely *learning* (live gradient), turning reward density, start geometry, take-down threshold, entropy, or opponent speed each moves the objective only ±5 pts. A robust plateau across many independent knobs is a strong sign the limit is *structural* (obs / dynamics / policy class), not a hyperparameter.
6. **"Slow the opponent" ≠ "make it learnable."** Halving R4t's speed barely helped (32% → ~32%) because the problem wasn't speed; it was the dead gradient. Diagnose *why* a lever should help before spending a run on it.
7. **Single-variable changes + a held-out objective eval are essential** in a long campaign. Stacking knobs (as we did toward the end) makes attribution impossible; keep one clean objective-condition eval (`R4t, full speed, full disc, 100 eps`) as the north-star number across all runs.
8. **Curriculum levers belong in the env as config, default-off.** `red_start_x_max` (constrain Red's start region) and `red_action_scale` (scale Red's speed) are clean, TDD'd `TeamConfig` knobs threaded through `_build_team_cfg` + `conf/env/team.yaml`, canary-safe at their defaults. This is the pattern for adding difficulty axes without forking the env.

## 7. Methodology / reproducibility notes

- **Run from the worktree root**, toolchain = **uv**, **sandbox disabled** (uv needs `~/.cache/uv`). Silence the stale conda env with `env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run ...`. `ls` is aliased to `eza` (`ls -dt` broken) — resolve latest run dir with `find runs/<name> -mindepth 1 -maxdepth 1 -type d | sort | tail -1`.
- **W&B** online (project `gridcom/drone-quidditch`); each run also writes `evaluations.npz` (`timesteps`, `results`, `ep_lengths`, `successes`).
- **De-noised objective eval recipe:** build a single-opponent eval env via `TeamEnvFactory(...).build_eval_env()` (with the run's obs blocks / `n_stack` / `team_cfg`, overriding `red_action_scale` / `red_start_x_max` per condition), load `best_model`, run `evaluate_policy(..., n_eval_episodes=100, deterministic=True)` with an `is_success`-collecting callback. Report beeline-only, R4t@train-condition, and **R4t@full-speed-full-disc** separately.
- **Each run ≈ 30–40 min** for 10–12M steps on the dev laptop.

## 8. Key artifacts

- **Frozen Red:** `runs/red_team_random/20260604_152427/best_model.zip` (R4t, `DUEL_V1_BODY` 22-d).
- **Best Blue defender so far:** `runs/blue_std_5_mixture_anneal/20260606_212638/best_model.zip` (~38% objective; honest, hack-free).
- **Committed:** `SuccessRateEvalCallback` (`scripts/callbacks.py`, commit `7b644c3`, GPG-signed).
- **Uncommitted code (entangled with the broader campaign env diff in `team_env.py`):** `red_start_x_max` + `red_action_scale` + `_apply_action` refactor + threading + tests. A clean commit needs the **team canary re-pinned** first (`RED_START_Z=0.5` shifted its fingerprint).
- **Configs (uncommitted):** rewards `team_v8_prevention`, `team_v9_positional`; experiments `blue_std_1..8`, `blue_sde_1..3`.

## 9. Open questions / what's next

The objective ceiling (~30–38%) looks structural. Candidate next directions, roughly in order of expected payoff vs cost:

1. **Self-play / co-training (chosen direction).** Stop treating R4t as a fixed target; let Red and Blue co-evolve. Lowest-infra option that fits the SB3 + frozen-opponent stack is **alternating self-play with a past-checkpoint league** (train Blue vs a mixture of recent Reds, freeze, train Red vs a mixture of recent Blues, repeat) — reuses the `MixtureOpponent` + frozen + pretrain machinery. True simultaneous MARL would need an RLlib/custom-PPO rewrite.
2. **Richer / predictive observation.** Does Blue actually have enough information to *anticipate* R4t's trajectory and pre-position? Candidates: explicit future-Red prediction features, hoop-relative shot-line geometry, longer frame stack, or a recurrent (LSTM) policy for trajectory memory.
3. **Physical capability check (cheap diagnostic).** Add a `blue_action_scale > 1.0` (mirror of `red_action_scale`) to test whether Blue is simply too slow/un-agile to reach interception points. If a faster Blue jumps the ceiling, the limit was physical, not perceptual.
4. **Reconsider the success criterion.** 100% prevention of a strong 75% scorer on a clear line may be geometrically infeasible; a *score-rate reduction* target (e.g., 75% → <30% scored, i.e. >70% prevention) is the realistic frame. We currently achieve ~38% prevention (≈ 47% scored).

---

*Compiled 2026-06-07 from `worktrees/exp/retraining/HANDOFF.md` and the 2026-06-05 → 06-07 session. Run-by-run numbers are de-noised eval (`is_success` over 50–100 episodes) unless noted as training-time `success_rate`.*
