# Drone Quidditch Sim

A reinforcement learning project that trains quadcopter drones to fly through a goal hoop — and, in the team variant, to chase down an opposing drone — in a simplified analogue of Quidditch. Built on [MuJoCo](https://mujoco.org/) physics with a vendored Crazyflie 2 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), and trained with [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) PPO (new API stack) + Ray Tune. Experiments are composed with [Hydra](https://hydra.cc/) and tracked in [Weights & Biases](https://wandb.ai/).

**Current status:** The single-drone "fly through a hoop" milestone is reached, and work has moved to **1v1 team play** (a Blue drone defending its hoop against a Red attacker). The stack runs on MuJoCo (migrated from PyFlyt/PyBullet on 2026-04-24): a generic `core/` infrastructure layer (`World` + per-drone `Quadrotor` views, MJCF composition primitives) plus a sport-specific `envs/quidditch/` package holding both `QuidditchSimpleEnv` (single-agent) and `QuidditchTeamEnv` (PettingZoo 1v1). The project tooling was rebuilt in three ML-infra passes: **Hydra config** (TOML/argparse retired → `conf/` group tree + `make train EXP=<name>`), **W&B integration** (TensorBoard retired → online experiment tracking + a model-artifact registry), and a **`dsim` Typer CLI** for inspection and dispatch. The toolchain is **uv**, not conda. Dynamics canary: `make test` runs a pytest suite whose scoring fingerprint asserts `SCORED at step 434 / total reward 7.3837`.

---

## Contents

- [What it does](#what-it-does)
- [Game rules](#game-rules)
- [Observation & action space](#observation--action-space)
- [Reward function](#reward-function)
- [Team play (1v1)](#team-play-1v1)
- [Requirements](#requirements)
- [Installation](#installation)
- [CLI surface](#cli-surface)
- [Workflow](#workflow)
- [Configuration (Hydra)](#configuration-hydra)
- [Project layout](#project-layout)
- [Architecture notes](#architecture-notes)
- [Roadmap](#roadmap)

---

## What it does

A single drone spawns inside a 3 m-radius circular arena. It must learn to fly forward and pass through a 50 cm-diameter hoop mounted 2 m above the ground, 2 m from the arena center. The environment is a Gymnasium-compatible custom env (`QuidditchSimpleEnv`) wrapping a MuJoCo simulator.

Physics steps at 240 Hz; the controller and the RL agent step at 120 Hz (one control update = two `mj_step` calls). A 120-second episode is 14 400 control steps.

---

## Game rules

| Constant | Value |
|---|---|
| Arena radius | 3 m (6 m diameter) |
| Arena wall height | 4.5 m |
| Hoop diameter | 0.5 m |
| Hoop center | (2, 0, 2) m |
| Hoop orientation | plane perpendicular to ground, outward normal [1, 0, 0] |
| Drone start | random uniform inside the arena (curriculum), or (0, 0, 0) |
| Episode length | 120 s env default (curriculum YAMLs set 30 s for fast iteration) |
| Flight mode | mode 7 — position setpoint [x, y, yaw, z], cascaded PID |

**Scoring:** every hoop has an invisible cylindrical "score tube" defined in MJCF, centred on the hoop plane along the outward normal (half-length 0.1 m). The drone carries an invisible probe sphere. Each step `mujoco.mj_geomDistance` reports whether the probe penetrates the tube. The env registers a score when the drone enters the tube from the arena-center side (signed distance < 0) and exits on the outside (signed distance > 0); entering and retreating doesn't score, and the score-tube and probe geoms are non-colliding (`contype=0 conaffinity=0`) so they can't push the drone around.

**Termination:** score, crash (z < 0.05 m after a 30-step take-off grace), out-of-bounds (> 3 m from origin), or episode timeout.

---

## Observation & action space

**Observation (16-dim, continuous):**

| Dims | Field |
|---|---|
| [0:3] | body-frame angular velocity (rad/s) |
| [3:6] | ground-frame Euler attitude (rad) |
| [6:9] | body-frame linear velocity (m/s) |
| [9:12] | ground-frame position (m) |
| [12:15] | unit vector from drone to hoop center |
| [15] | signed distance to hoop plane / arena radius |

**Action (4-dim, continuous, normalized to [−1, 1]):**

Normalized delta applied to the running position setpoint each step:

| Dim | Axis | Scale |
|---|---|---|
| 0 | Δx | ±0.2 m |
| 1 | Δy | ±0.2 m |
| 2 | Δyaw | ±0.5 rad |
| 3 | Δz | ±0.1 m |

The setpoint is clamped to arena bounds (xy) and altitude [0.01, 4.0] m after each update.

---

## Reward function

| Event | Reward |
|---|---|
| Each step | −(dist_to_hoop / arena_radius) × 0.01 |
| Fly through hoop | +10 |
| Crash or out-of-bounds | −20 |

The per-step distance penalty provides a dense gradient toward the hoop. The +10 bonus dominates accumulated step penalty, so the agent is incentivised to score quickly. The crash/OOB penalty was bumped from −2 to −20 on 2026-04-17 after a curriculum run found the agent learned that crashing was cheaper than timing out.

The reward function above is the single-agent stack (`conf/reward/single_agent.yaml`). Rewards are assembled as a `RewardStack` of small composable terms (`envs/quidditch/rewards/terms.py` — `ScoreEvent`, `CrashEvent`, `HoopDistancePenalty`, `HoopAnchor`, `ZeroSumDistMirror`, `TagEntryPulse`, `ProximityGradedTag`, `ClosingVelInTagZone`, `TakeDown`, `InterceptShaping`, `GoalSideCone`), so a new reward shaping is a YAML edit, not a code change. Team stacks live in `conf/reward/team_v*.yaml` — see [Team play](#team-play-1v1).

---

## Team play (1v1)

The second milestone pits two drones against each other: a **Red** attacker trying to score through the hoop and a **Blue** defender trying to stop it. This runs through `QuidditchTeamEnv` (a PettingZoo `ParallelEnv`), trained directly as a multi-agent environment by RLlib (new API stack) — each side is a policy, and scripted opponents are mounted as frozen `RLModule`s (`envs/quidditch/rllib_modules.py`).

**Scoring & events:**

- **Tag** — Blue "tags" Red by closing inside a 0.3 m proximity sphere (`envs/quidditch/tagging.py`), with a 1 s post-exit cooldown. Tags drive Blue's reward (entry pulse + proximity-graded + closing-velocity bonus).
- **Take-down** — a drone-vs-drone contact above `CRASH_VEL_THR = 1.0 m/s` (`envs/quidditch/crash.py`) is a decisive crash; Blue ramming Red earns +20 / Red −20.
- **Score** — Red flying through the hoop scores +10 (zero-summed against Blue).
- **Crash aftermath** — under `--gui`, drone-drone-crash termination is deferred a few seconds (`crash_aftermath_seconds`) and the loser's motors are cut so the impact is legible on video; training defaults to 0 s (immediate termination, byte-identical canary).

**Opponents** (`conf/opponent/`, classes in `envs/quidditch/opponents.py`): `beeline_red` / `beeline_blue` (scripted straight-line), `intercepter_blue` (predicts and cuts off the target), and `none`. Frozen self-play snapshots are managed inside the RLlib league, not via these scripted-opponent configs.

**Observations** are richer than the single-agent 16-d vector and are composed from named blocks (`conf/obs/*.yaml`, `obs.blocks`): `duel_v1_body` (22-d, opponent in body frame, unstacked), `duel_v2_world` (25-d, opponent in world frame — the default), and `duel_v3_body_ego` (25-d, fully body-frame ego-centric). The two current specs are frame-stacked (`obs.n_stack = 3`, so the policy sees a 75-d input). Because the obs layout is a persisted, named structure, loading a checkpoint validates its obs spec against the current one (`core/obs_compat.py`) and refuses a silent mismatch.

---

## Requirements

- macOS (Apple Silicon tested; Linux should work with minor adjustments)
- [uv](https://docs.astral.sh/uv/)
- Python 3.13 (pinned in `.python-version`; uv will download it on first sync)

---

## Installation

```bash
# 1. Clone this repo
git clone <this-repo-url>

# 2. Sync the uv environment (Python 3.13 + MuJoCo + Ray RLlib + imageio)
make install

# 3. Sanity-check the env
make test           # full pytest suite — scoring canary asserts "SCORED at step 434 / total reward 7.3837"
make test-fast      # unit tests only (skips @pytest.mark.slow integration runs)
```

`make install` is idempotent — `uv sync` creates `.venv/` on first run and updates it on subsequent runs based on `uv.lock`. Run anything in the venv with `uv run <cmd>` (or `make`/`dsim`, which wrap it). Configuration lives in the git-tracked `conf/` tree — there is no local config to seed.

> **macOS OpenMP:** training spawns multiple Ray env-runner processes; on macOS, multiple copies of `libomp` can coexist across Python distributions and cause a duplicate-init abort. `KMP_DUPLICATE_LIB_OK=TRUE` is propagated to the Ray workers via `ray.init(runtime_env=...)` (see `rllib/runtime.py`), so the guard reaches every env-runner. No user action required.
>
> **macOS viewer:** the interactive MuJoCo passive viewer needs `mjpython` (it owns the Cocoa main thread). Use `uv run mjpython demo/menu.py` for demos; headless rendering and training use plain `python`.

---

## CLI surface

Three command surfaces, by purpose:

| Purpose | Surface | Examples |
|---|---|---|
| Inspection / read-only / one-shot dispatch | **`dsim`** (Typer) | `dsim inventory`, `dsim obs-preflight --parent X --child-obs Y`, `dsim lineage --target ...`, `dsim list-runs`, `dsim resume <run-name>` (RLlib resume not yet wired — fails fast), `dsim promote <run-name>`, `dsim describe-run <run-name>`, `dsim obs-specs` |
| Composable runs (training, eval) | **Hydra apps** (`python -m scripts.X`) | `python -m scripts.train +experiment=rllib_league_step5`, `python -m scripts.eval_team +eval_team=default +learner=red learner.uri=<checkpoint-dir>` |
| Chores (install, test, train one-off, TUI) | **`make`** | `make install`, `make test`, `make test-fast`, `make train EXP=rllib_league_step5`, `make tui` |

Inspection and dispatch went to `dsim` for discoverability (`dsim --help` lists everything). Hydra apps cover anything with composable configs. Make is reserved for true chores plus the one-off `make train EXP=X` muscle-memory shortcut.

### Replacing removed `make` targets

| Old | New |
|---|---|
| `make demo` | TUI -> Demo task (Slice 2); or `mjpython demo/menu.py <key>` directly |
| `make camera-test CAM=<x>` | TUI -> Camera Test (Slice 2); or `python demo/camera_test.py --cam <x>` |
| `make eval` | Retired — the single-agent eval path was removed with the SB3 path (Step 6) |
| `make eval-team LEARNER=… BLUE=… RED=… GUI=1` | `python -m scripts.eval_team +eval_team=default +learner=<side> learner.uri=<checkpoint-dir>` |
| `make resume RUN_NAME=…` | `dsim resume <run-name>` |
| `make lineage RUN_NAME=…` | `dsim lineage --target models/ppo_hoop_<name>_*/best_model` |
| `make promote RUN_NAME=…` | `dsim promote <run-name>` |
| `make list-runs` | `dsim list-runs` |
| `make obs-specs` | `dsim obs-specs` |
| `make describe-run RUN_NAME=…` | `dsim describe-run <run-name>` |
| `make sweep SWEEP=…` / `sweep-agents ID=… N=…` | Retired — hyperparameter sweeps run via Ray Tune (`tune` config group) inside `scripts/train.py` |

---

## Workflow

### Train

Training is a Hydra app. The daily entrypoint is `make train EXP=<name>`, where `<name>` is a file under `conf/experiment/` — each experiment YAML pins the group choices (env / obs / reward / opponent / init / curriculum) and any hyperparameter overrides for one run.

```bash
make train EXP=blue_v7                          # train the blue_v7 experiment
make train EXP=blue_v7 OVERRIDES="seed=7 trainer.lr=1e-4"   # ad-hoc overrides
python -m scripts.train +experiment=blue_v7     # the same thing, no make wrapper

ls conf/experiment/                             # list available experiments
```

`init.mode` is `scratch` only — the RLlib league always starts from a fresh policy (the SB3 `pretrain`/`resume`/`warm_start` modes and parent loading were retired in Step 6).

Training artifacts land in `runs/<run_name>/<timestamp>/`:
- `tune/<trial>/checkpoint_NNNNNN/` — Ray Tune checkpoint dirs (the best one is logged as the `:latest` W&B artifact)
- `videos/` — multi-camera episode clips, logged to W&B
- `.hydra/{config,overrides,hydra}.yaml` — the fully-composed config (Hydra-written; the config snapshot)
- `.hydra/meta.yaml` — git hash, parent lineage edge, and final eval stats
- `MODEL.md` — auto-generated human-readable spec sheet (obs spec, reward stack, hyperparams, eval results)
- `wandb/` — local W&B client state

**Experiment tracking is W&B** (TensorBoard was retired). Runs log online by default to project `drone-quidditch`; set `WANDB_MODE=offline` to defer upload or `WANDB_MODE=disabled` to skip it (the test suite runs disabled). The run id matches the on-disk dir basename (`<run_name>_<timestamp>`).

### Evaluate

`scripts.eval_team` loads an RLlib checkpoint and runs the native
main_red-vs-main_blue battery:

```bash
# 1v1 team: load a checkpoint dir, run the RLlib head-to-head battery
python -m scripts.eval_team +eval_team=default \
    +learner=red learner.uri=runs/rllib_league_step5/<ts>/tune/<trial>/checkpoint_NNN \
    eval.n_episodes=20
```

`learner.uri` is an RLlib checkpoint dir (or a run dir whose latest checkpoint
is auto-discovered). Eval prints per-side score / prevention / take-down rates,
mean episode length, and the terminal-cause buckets. (The SB3 single-agent
eval and the standalone battery script were retired in Step 6; the battery
now runs inside `eval_team` via `rllib/eval_battery.py`.)

### Inspect, promote, resume — `dsim`

```bash
dsim inventory                 # what's on disk: runs, promoted models, caches
dsim list-runs                 # all training runs + promoted models
dsim describe-run <run-name>   # render a run's MODEL.md spec sheet
dsim obs-specs                 # block-by-block layout of every named obs spec
dsim obs-preflight --parent <uri> --child-obs <name>   # will a load/surgery succeed?
dsim lineage --target <path-or-uri>                    # walk the parent chain
dsim resume <run-name>         # (RLlib Tuner.restore not yet wired — fails fast)
dsim promote <run-name>        # promote a run's best checkpoint into models/ + alias the W&B artifact
```

Promoting copies the run's best checkpoint (an RLlib `checkpoint/` dir, or a legacy `best_model.zip`) + `.hydra/` + `MODEL.md` into `models/<run-name>/` (git-tracked) and moves the `:prod` alias on the W&B artifact. Then commit the new `models/<run-name>/` directory.

### Demos & camera

```bash
uv run mjpython demo/menu.py        # interactive demo picker (hover, waypoint, takedown, score-through-tag)
python demo/camera_test.py          # headless hover render through the fixed scene camera
```

The fixed scene camera (`eye` + `lookat`) and the per-drone follow cameras live in `conf/camera/default.yaml`; the same config drives both the live viewer pose and the offscreen renderer used for training videos.

---

## Configuration (Hydra)

There is no TOML config and no local working copy — everything is the git-tracked `conf/` group tree, composed at run time by [Hydra](https://hydra.cc/). The default stack is set in `conf/config.yaml`; experiment YAMLs override individual groups.

| Group | Choices (`conf/<group>/`) | What it sets |
|---|---|---|
| `algo` | `ppo_rllib` | RLlib PPO hyperparameters + `total_timesteps` |
| `multiagent` | `red_solo`, `red_blue_selfplay`, `red_blue_league` | policy mapping + per-module specs |
| `tune` | `default` | Ray Tune run config (checkpointing, W&B logger) |
| `league` | `disabled`, `default` | self-play snapshot league (PFSP) |
| `env` | `team` | team physics thresholds (`team_env_params`) |
| `obs` | `simple`, `duel_v1_body`, `duel_v2_world`, `duel_v3_body_ego` | obs blocks + frame-stack depth |
| `reward` | `single_agent`, `team_v1…v4_*` | the `RewardStack` of terms |
| `opponent` | `none`, `beeline_red/blue`, `intercepter_blue` | scripted sparring partner (team only) |
| `curriculum` | `fixed_start`, `random_start` | start randomisation + episode length |
| `init` | `scratch` | always scratch (SB3 parent-loading modes retired in Step 6) |
| `eval` | `default`, `fast` | eval cadence + episode count |
| `wandb` | `default` | W&B project / tags / verbosity |
| `learner` | `blue`, `red` | which side the eval-team learner controls |
| `local` | (gitignored) | optional per-machine hardware tunings |

Group defaults are validated against `@dataclass` schemas in `config_schema.py` at compose time. Opponents and reward terms are `_target_`-instantiated, so adding one is a YAML edit plus a class — no entrypoint changes.

```yaml
# conf/config.yaml — the default stack (a team run on the RLlib new stack)
defaults:
  - env: team
  - obs: duel_v2_world
  - reward: team_v2
  - opponent: beeline_red
  - eval: default
  - init: scratch
  - curriculum: random_start
  - wandb: default
  - algo: ppo_rllib       # RLlib PPO hyperparameters
  - multiagent: red_solo  # policy mapping / module specs
  - tune: default         # Ray Tune run config
  - league: disabled      # self-play snapshot league (off by default)
```

**Tuning notes carried from prior runs:**
- A large entropy bonus caused instability — it dominated the sparse hoop reward. `algo.entropy_coeff` is kept at 0.01 and annealed toward 0 (`entropy_coeff_schedule`).
- Too few policy updates per run stalls learning; the RLlib path tunes the rollout/update balance via `algo.train_batch_size_per_learner` and `algo.num_epochs`.
- Training parallelises rollouts across Ray env-runners (`algo.num_env_runners`); the learner trains on the aggregated batch (`algo.train_batch_size_per_learner`).

---

## Project layout

```
repo/
├── core/                       generic infra (no Quidditch knowledge)
│   ├── position_controller.py  cascaded PID (mode 7, cf2x gains from PyFlyt's cf2x.yaml)
│   ├── world.py                World — owns MjModel/MjData/viewer/renderer/step loop
│   ├── quadrotor.py            Quadrotor — per-drone view bound to a World
│   ├── mjcf/                   composition primitives (fragment, document, meshes, camera)
│   ├── drone/cf2x.py           cf2x_assets() + cf2x_fragment(prefix, ...)
│   ├── obs_compat.py           obs-spec compatibility check / preflight
│   ├── eval_core.py            eval result dataclasses + terminal-cause buckets
│   ├── rllib_checkpoint.py     RLlib checkpoint discovery/detection
│   ├── rllib_eval.py           RLlib head-to-head battery runner
│   ├── eval_report.py          battery report rendering
│   ├── inventory.py            on-disk run/model discovery
│   ├── lineage.py              parent-chain walkers (local + W&B DAG)
│   ├── promote.py              run → models/ promotion logic
│   ├── run_context.py          run-dir / config locating
│   └── run_listing.py          list-runs backing logic
├── envs/                       sport-specific
│   ├── __init__.py
│   └── quidditch/
│       ├── constants.py        ARENA_RADIUS, HOOP_*, TAG_*, CRASH_VEL_THR — single source of truth
│       ├── scene.py            arena_wall_fragment, hoop_fragment
│       ├── scoring.py          GeomDistanceScorer (hoop scoring)
│       ├── tagging.py          TagDistanceScorer (proximity tag, team play)
│       ├── crash.py            CrashDetector (drone-drone / wall take-downs)
│       ├── obs_spec.py         named ObsBlock/ObsSpec, YAML-driven composition
│       ├── opponents.py        scripted opponents (Beeline*, Intercepter*, Zero)
│       ├── rllib_env.py        make_team_env — registers "quidditch_team" for RLlib
│       ├── rllib_modules.py    learned + scripted RLModules (new API stack)
│       ├── rewards/            RewardStack + composable reward terms
│       ├── simple_env.py       QuidditchSimpleEnv (single-agent; obs-spec unit tests)
│       └── team_env.py         QuidditchTeamEnv — PettingZoo ParallelEnv (1v1)
├── scripts/                    Hydra apps + run-lifecycle helpers
│   ├── train.py                RLlib + Ray Tune training entrypoint (canonical)
│   ├── eval_team.py            1v1 RLlib head-to-head eval (loads a checkpoint)
│   ├── promote.py / lineage.py model promotion + lineage backing scripts
│   ├── _artifact_io.py         W&B artifact registry (resolve_parent, log_*_artifact)
│   └── _render_model_doc.py    MODEL.md generator
├── dsim/                       Typer CLI (inspection + dispatch); `dsim --help`
│   ├── cli.py                  command wiring
│   └── commands/               inventory, list-runs, describe-run, lineage, resume,
│                               promote, obs-specs, obs-preflight
├── conf/                       Hydra config group tree (git-tracked)
│   ├── config.yaml             default group choices
│   ├── env/ obs/ reward/ opponent/ curriculum/ init/ eval/ wandb/
│   ├── algo/ multiagent/ tune/ league/   RLlib + Ray Tune config groups
│   ├── experiment/             one YAML per run (rllib_red_skeleton, rllib_selfplay, rllib_league*)
│   ├── learner/ eval_team/
│   └── camera/default.yaml     scene + per-drone follow cameras
├── demo/                       hover / waypoint / takedown / score-through-tag + menu
├── assets/cf2x/               Menagerie cf2 visual + collision meshes (Apache 2.0)
├── docs/superpowers/           per-feature spec + plan docs
├── models/                     promoted models — vendored, git-tracked (.cache/ is gitignored)
├── runs/                       training artifacts (gitignored)
├── tests/                      pytest suite (mirrors source layout)
├── config_schema.py           @dataclass schemas validated at Hydra compose time
├── Makefile                    chores only (install, clean, test*, tui, train)
├── pyproject.toml
└── uv.lock
```

---

## Architecture notes

**Simulator:** [MuJoCo](https://mujoco.org/) (pip `mujoco>=3.0`). Replaced PyFlyt/PyBullet on 2026-04-24 — the new stack has working offscreen rendering on Apple Silicon and ships visual + collision meshes the previous setup never had.

**Drone model:** Crazyflie 2 (cf2x), 27 g, arm length 0.028 m. Visual meshes vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `bitcraze_crazyflie_2/` (commit `affef0836947b64cc06c4ab1cbf0152835693374`, Apache 2.0). Inertia tensor adopted from Menagerie (`IXX = IYY = 2.3951e-5`, `IZZ = 3.2347e-5` kg·m²). Motor coefficients (`THRUST_COEF`, `TORQUE_COEF`, `MAX_RPM`) kept from PyFlyt's `cf2x.yaml` because Menagerie's `<motor gear=...>` actuator model doesn't directly map to our PID's RPM-squared thrust formulation — adopting it would require rewriting the controller.

**Controller:** pure-Python cascaded PID in [core/position_controller.py](core/position_controller.py), implementing PyFlyt mode 7 (position setpoint → desired velocity → desired thrust + attitude → motor PWMs). Gains match PyFlyt's `cf2x.yaml` exactly. Physics steps at 240 Hz; control steps at 120 Hz (one PID update = two `mj_step` calls).

**Gymnasium env:** [envs/quidditch/simple_env.py](envs/quidditch/simple_env.py). The env owns no MuJoCo state directly — it constructs a `core.world.World` on first `reset()` from a list of `SceneFragment` objects (cf2x assets + cf2x fragment + arena wall + hoop) and runs through a `Quadrotor` view bound to that world.

**Scene composition:** [core/mjcf/](core/mjcf/) provides `SceneFragment` (MJCF chunks + binary asset bytes) and `build_mjcf(opts, fragments)` (string concat under one `<mujoco>` root). No file `<include>` directives, no temp files — all MJCF assembly happens in memory at sim-init via `MjModel.from_xml_string(xml, assets=...)`. Adding a second drone is a one-line fragment append — which is exactly how the team env builds its 1v1 scene.

**Team env:** [envs/quidditch/team_env.py](envs/quidditch/team_env.py) — `QuidditchTeamEnv` is a PettingZoo `ParallelEnv` with Red and Blue sharing one `MjModel` (two `Quadrotor` views; two MjModels can't share a contact world). RLlib trains it directly as a multi-agent env: `envs/quidditch/rllib_env.py:make_team_env` registers it as `"quidditch_team"`, and `envs/quidditch/rllib_modules.py` provides the learned and scripted `RLModule`s. Collisions are enabled here (the 32 cf2 collision hulls are opt-in via `with_collision_meshes=True` / `with_collisions=True`, bitmask `1`; hoop and arena-wall stay phase-through on bit `0`); the single-drone path leaves them off and stays byte-identical.

**Obs specs:** [envs/quidditch/obs_spec.py](envs/quidditch/obs_spec.py) — obs layouts are named `ObsSpec`s composed from `ObsBlock`s, declared as `blocks: [...]` lists in `conf/obs/*.yaml` and built at env construction. The spec persists with each checkpoint, so loading validates the parent's obs spec against the current one ([core/obs_compat.py](core/obs_compat.py)) and strict-refuses a silent mismatch.

**Hoop scoring:** [envs/quidditch/scoring.py](envs/quidditch/scoring.py) — `GeomDistanceScorer.overlaps()` returns an `(N drones × M hoops)` boolean matrix from `mujoco.mj_geomDistance` between each drone's invisible probe sphere and each hoop's invisible score tube. The earlier signed-distance plane crossing was replaced because `mj_geomDistance` is the right primitive for "is this geom inside that geom" and doesn't involve the contact solver. (A first MuJoCo-era attempt used a `<contact><pair>` with `solimp="0 0 ..."` to read overlap from contact reports; it pinned the drone at the tube boundary with ~0.1 N residual force.)

**Config & experiment tracking:** runs are composed by Hydra from the `conf/` group tree (validated against `@dataclass` schemas in `config_schema.py`) and tracked in W&B. The fully-resolved config is saved to each run's `.hydra/`, and promoted models are registered as W&B artifacts (`<run_name>:prod`) whose lineage DAG is walkable via `dsim lineage`. Opponents and reward terms are `_target_`-instantiated, so the config tree is the extension surface.

**Camera:** the fixed scene camera (`eye` + `lookat`) and per-drone follow cameras live in `conf/camera/default.yaml`, loaded directly via PyYAML (cameras aren't an experiment axis, so they're outside Hydra composition). The same config drives the live viewer pose and the offscreen renderer for training videos. `python demo/camera_test.py` renders a hover flight through the scene camera without launching a training run.

**macOS OpenMP:** `rllib/runtime.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and propagates it to every Ray env-runner via `ray.init(runtime_env={"env_vars": ...})` (and `tests/conftest.py` sets it for the suite), suppressing the libomp double-init abort that can occur on macOS Apple Silicon when worker processes load multiple `libomp` copies.

---

## Roadmap

| Phase | Status |
|---|---|
| 1 — Foundation, hover smoke test (PyFlyt) | Done — superseded by MuJoCo migration |
| 2 — Game design (arena, hoop, reward constants) | Done |
| 3 — Gymnasium env (`QuidditchSimpleEnv`) | Done |
| 4–5 — PPO training + evaluation/promote workflow | Done |
| 6 — MuJoCo migration (sim, controller, scoring) | Done (2026-04-24 → 2026-05-01 refactor + Menagerie cf2 drop-in) |
| 7 — Single-agent retrain on MuJoCo dynamics | Done — hoop milestone reached |
| 8 — Team play: 1v1 Red attacker vs Blue defender (`QuidditchTeamEnv`) | In progress — opponent ladder + Blue policy iteration |
| ML infra — Hydra config, W&B tracking, `dsim` CLI | Done |
| — | Multi-agent self-play / opponent league |
| — | Multiple hoops |
| — | Carryable quaffle |
| — | Seeker / snitch dynamic |
