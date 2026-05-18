# MODEL: ppo_hoop_blue_1_ppo_hoop_blue_1_20260507_194423

**Status:** promoted  ·  **Git:** `<legacy>`
**W&B:** `wandb://ppo_hoop_blue_1:prod` (v0, aliases: latest, prod, ppo_hoop_blue_1, v0)

## Summary

drone_0 learner, init=scratch, obs DUEL_V1_BODY × n_stack=1.

_(legacy migrated config — trainer / reward / opponent / curriculum not recorded in .hydra/config.yaml; see run_info.toml for legacy training metadata.)_

## Lineage

- **init mode:** `scratch` — no parent

## Obs spec

**Name:** `DUEL_V1_BODY` (22-d)  ·  **n_stack:** 1  ·  **Input dim:** 22

| Slot | Block | Dim | Frame | Notes |
|------|-------|-----|-------|-------|
| 0:3 | ang_vel | 3 | body |  |
| 3:6 | ang_pos | 3 | body |  |
| 6:9 | lin_vel | 3 | body |  |
| 9:12 | lin_pos | 3 | world |  |
| 12:15 | unit_to_goal | 3 | world | unit vector toward hoop (red) or midpoint (blue) |
| 15:16 | signed_dist_norm | 1 |  | (pos - hoop)·hoop_normal / ARENA_RADIUS |
| 16:19 | opp_pos_rel | 3 | world |  |
| 19:22 | opp_vel_rel | 3 | body_mixed | legacy: each velocity in its own body frame |

## Reward stack

_(legacy migrated config — reward stack composition not recorded in .hydra/config.yaml; see run_info.toml for legacy reward magnitudes / Phase-2 narrative.)_

## Env config

- **Opponent:** `(none)`  ·  **Learner:** `drone_0`
- **Curriculum:** `(unknown)`

## Training hyperparams

_(legacy migrated config — trainer hyperparams not recorded in .hydra/config.yaml; see run_info.toml for legacy `[training].*` fields.)_

## Eval results

- **completed_steps:** 10,002,432  ·  **wall_clock:** (unknown)
- **model_kind:** `(unknown)`

## W&B

- **Project:** `gridcom/drone-quidditch`
- **Run id:** `ppo_hoop_blue_1_ppo_hoop_blue_1_20260507_194423`
- **Artifact:** `ppo_hoop_blue_1:v0`  (aliases: latest, prod, ppo_hoop_blue_1, v0)
