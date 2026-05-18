# MODEL: ppo_hoop_blue_4_ppo_hoop_blue_4_20260511_202612

**Status:** promoted  ·  **Git:** `<legacy>`
**W&B:** `wandb://ppo_hoop_blue_4:prod` (v0, aliases: latest, prod, ppo_hoop_blue_4, v0)

## Summary

drone_0 learner, init=scratch, obs DUEL_V2_WORLD × n_stack=3.

_(legacy migrated config — trainer / reward / opponent / curriculum not recorded in .hydra/config.yaml; see run_info.toml for legacy training metadata.)_

## Lineage

- **init mode:** `scratch` — no parent

## Obs spec

**Name:** `DUEL_V2_WORLD` (25-d)  ·  **n_stack:** 3  ·  **Input dim:** 75

| Slot | Block | Dim | Frame | Notes |
|------|-------|-----|-------|-------|
| 0:3 | ang_vel | 3 | body |  |
| 3:6 | ang_pos | 3 | body |  |
| 6:9 | lin_vel | 3 | body |  |
| 9:12 | lin_pos | 3 | world |  |
| 12:15 | unit_to_goal | 3 | world | unit vector toward hoop (red) or midpoint (blue) |
| 15:18 | vec_to_hoop | 3 | world | HOOP_CENTER - learner_pos, not normalized |
| 18:21 | opp_pos_rel | 3 | world |  |
| 21:24 | opp_vel_rel | 3 | world |  |
| 24:25 | closing_rate | 1 |  | -d‖opp - learner‖/dt |

## Reward stack

_(legacy migrated config — reward stack composition not recorded in .hydra/config.yaml; see run_info.toml for legacy reward magnitudes / Phase-2 narrative.)_

## Env config

- **Opponent:** `(none)`  ·  **Learner:** `drone_0`
- **Curriculum:** `(unknown)`

## Training hyperparams

_(legacy migrated config — trainer hyperparams not recorded in .hydra/config.yaml; see run_info.toml for legacy `[training].*` fields.)_

## Eval results

- **completed_steps:** 20,004,864  ·  **wall_clock:** (unknown)
- **model_kind:** `(unknown)`

## W&B

- **Project:** `gridcom/drone-quidditch`
- **Run id:** `ppo_hoop_blue_4_ppo_hoop_blue_4_20260511_202612`
- **Artifact:** `ppo_hoop_blue_4:v0`  (aliases: latest, prod, ppo_hoop_blue_4, v0)
