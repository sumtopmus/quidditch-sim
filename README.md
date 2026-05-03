# Drone Quidditch Sim

A reinforcement learning project that trains a quadcopter drone to fly through a goal hoop in a simplified analogue of Quidditch. Built on [MuJoCo](https://mujoco.org/) physics with a vendored Crazyflie 2 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), and trained with Stable-Baselines3 PPO.

**Current status:** Migrated to MuJoCo (was PyFlyt/PyBullet; the simulator swap landed 2026-04-24). The drone/scene refactor split the old single-module simulator into a generic `core/` infrastructure layer (`World` + per-drone `Quadrotor` view, MJCF composition primitives) and a sport-specific `envs/quidditch/` subpackage. The cf2x drone now uses Menagerie's visual meshes; 32 collision hulls are vendored alongside, flag-gated for the upcoming multi-agent work. Hoop scoring switched from a signed-distance plane crossing to a geometric `mj_geomDistance` query against an invisible score tube. Dynamics canary `make check-sim` reproducibly prints `SCORED at step 431 / total reward 7.3827`. Pre-MuJoCo trained models are not compatible — retrain from scratch is the next training-side step.

---

## Contents

- [What it does](#what-it-does)
- [Game rules](#game-rules)
- [Observation & action space](#observation--action-space)
- [Reward function](#reward-function)
- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
- [Training configuration](#training-configuration)
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
| Episode length | 120 s default (template config sets 30 s for fast iteration) |
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

---

## Requirements

- macOS (Apple Silicon tested; Linux should work with minor adjustments)
- [Conda / Miniforge](https://github.com/conda-forge/miniforge)
- Python 3.11 (pinned in `environment.yml`)

---

## Installation

```bash
# 1. Clone this repo
git clone <this-repo-url>

# 2. Create the conda environment (Python 3.11 + MuJoCo via pip + SB3 + imageio)
make install

# 3. Sanity-check the env
make check-sim      # headless (fast) — should print "SCORED at step 431 / total reward 7.3827"
make check-gui      # opens MuJoCo viewer for visual inspection
```

`make install` is idempotent — it creates the environment on first run and updates it on subsequent runs. It also seeds `config/training.toml` and `config/camera.toml` from `templates/` if those local files don't exist yet.

> **macOS OpenMP:** training spawns multiple SB3 envs via `SubprocVecEnv`, and conda ships multiple copies of `libomp` on Apple Silicon. `train_ppo.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time to suppress the duplicate-init abort. No user action required.

---

## Workflow

All day-to-day tasks go through `make`. Run `make help` for the full list.

### Train

```bash
make train                         # train with default config (ppo_hoop run name)
make train RUN_NAME=my_experiment  # custom run name
```

Training artifacts land in `runs/<RUN_NAME>/<timestamp>/`:
- `PPO_1/` — TensorBoard logs
- `checkpoints/` — model snapshots every 50 k steps
- `videos/` — episode clips through the fixed scene camera every 100 k steps
- `best_model.zip` — best checkpoint by eval reward
- `config_snapshot.toml` — exact training config used (for `make repro`)
- `info.toml` — run metadata and finish timestamp

```bash
make tensorboard                   # launch TensorBoard across all runs
make tensorboard RUN_NAME=ppo_hoop # just one run name
make list-runs                     # list all trials and promoted models
```

### Evaluate

```bash
make eval                          # open MuJoCo viewer, 10 episodes (latest trial)
make eval-headless                 # headless, 50 episodes
make eval TRIAL=20260416_190850    # specific trial
make eval EPISODES=25              # custom episode count
```

Prints: score rate, crash rate, timeout rate, mean reward ± std, mean steps-to-score.

### Promote a model

When a training run looks good, promote it to `models/` (tracked in git):

```bash
make promote TRIAL=20260416_190850
# → copies best_model.zip + config + info into models/ppo_hoop_20260416_190850/
```

Then commit:
```bash
git add models/ppo_hoop_20260416_190850
git commit -m "model: promote 20260416_190850 best model"
```

### Reproduce a run

```bash
make repro MODEL=ppo_hoop_20260416_190850
# → restores config/training.toml from the promoted model's snapshot
make train
```

### Camera setup

The fixed scene camera used for both `make camera-test` and per-checkpoint training videos is configured in `config/camera.toml` (`eye` + `lookat`). Edit, then `make camera-test` re-renders a hover flight through it to `runs/camera_test/hover_camera_test.mp4` plus a still PNG of the last frame. Fast iteration before kicking off a real training run.

### Utilities

```bash
make clean          # remove __pycache__ and .pyc files
```

---

## Training configuration

All PPO hyperparameters live in [config/training.toml](config/training.toml) (gitignored — copied from [templates/training.toml](templates/training.toml) on first `make install`). CLI flags `--timesteps`, `--n-envs`, `--lr`, `--seed` override individual values at runtime.

```toml
[training]
run_name        = "ppo_hoop"
total_timesteps = 2_000_000
n_envs          = 8          # SubprocVecEnv (parallel)
seed            = 42         # integer ≥ 0 for reproducible runs; -1 to disable

[training.ppo]
n_steps    = 1024
batch_size = 256
n_epochs   = 10
lr         = 3e-4
gamma      = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef   = 0.01

[training.eval]
eval_freq_steps  = 50_000
n_eval_episodes  = 10

[training.callbacks]
checkpoint_freq_steps = 50_000
video_freq_steps      = 100_000
video_fps             = 20

[env]
randomise_start = false      # start at (0,0,0) for the first milestone; flip to true for curriculum
episode_seconds = 30.0       # short episodes for fast iteration; default env value is 120
```

**Key tuning notes from prior runs (PyFlyt-era; expected to carry over to MuJoCo, but not yet re-validated):**
- `ent_coef = 0.05` caused instability — entropy bonus dominated sparse hoop reward. Keep it at 0.01.
- `n_steps = 2048` with 8 envs gave only 122 policy updates per 2 M steps. Halving to 1024 doubled update frequency and stabilised learning.
- Training uses `SubprocVecEnv` (n parallel workers) for the training env and `DummyVecEnv` (1 worker) for evaluation — the type-mismatch warning from SB3 is intentional and suppressed.

---

## Project layout

```
repo/
├── core/                       generic infra (no Quidditch knowledge)
│   ├── position_controller.py  cascaded PID (mode 7, cf2x gains from PyFlyt's cf2x.yaml)
│   ├── world.py                World — owns MjModel/MjData/viewer/renderer/step loop
│   ├── quadrotor.py            Quadrotor — per-drone view bound to a World
│   ├── mjcf/                   composition primitives
│   │   ├── fragment.py         SceneFragment dataclass + merge()
│   │   ├── document.py         build_mjcf(world_opts, fragments) → str
│   │   ├── meshes.py           torus, arena-wall, marker mesh data
│   │   └── camera.py           xyaxes derivation + load_camera_config
│   └── drone/
│       └── cf2x.py             cf2x_assets() + cf2x_fragment(prefix, ...)
├── envs/                       sport-specific
│   ├── __init__.py             re-exports QuidditchSimpleEnv
│   └── quidditch/
│       ├── constants.py        ARENA_RADIUS, HOOP_*, single source of truth
│       ├── scene.py            arena_wall_fragment, hoop_fragment
│       ├── scoring.py          GeomDistanceScorer
│       └── simple_env.py       QuidditchSimpleEnv
├── scripts/
│   ├── check_env.py            sanity checks (make check-sim / make check-gui)
│   ├── train_ppo.py            SB3 PPO training entry point
│   ├── eval_ppo.py             evaluation script
│   └── callbacks.py            checkpoint + video callbacks
├── demo/
│   ├── menu.py                 interactive demo selector (make demo)
│   ├── hover_demo.py           hover smoke test (Quidditch arena)
│   ├── waypoint_demo.py        triangular waypoint flight (empty scene)
│   └── camera_test.py          headless render of hover through fixed cam
├── assets/
│   └── cf2x/                   Menagerie cf2 visual + collision meshes (Apache 2.0)
├── config/                     gitignored — local working configs
│   ├── training.toml
│   └── camera.toml
├── templates/                  git-tracked defaults; copied on `make install`
│   ├── training.toml
│   └── camera.toml
├── models/                     promoted models (tracked in git)
├── runs/                       training artifacts (gitignored)
├── Makefile
├── environment.yml
└── requirements.txt
```

---

## Architecture notes

**Simulator:** [MuJoCo](https://mujoco.org/) (pip `mujoco>=3.0`). Replaced PyFlyt/PyBullet on 2026-04-24 — the new stack has working offscreen rendering on Apple Silicon, ships visual + collision meshes the previous setup never had, and removes the PyBullet conda-binary gotcha entirely.

**Drone model:** Crazyflie 2 (cf2x), 27 g, arm length 0.028 m. Visual meshes vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `bitcraze_crazyflie_2/` (commit `affef0836947b64cc06c4ab1cbf0152835693374`, Apache 2.0). Inertia tensor adopted from Menagerie (`IXX = IYY = 2.3951e-5`, `IZZ = 3.2347e-5` kg·m²). Motor coefficients (`THRUST_COEF`, `TORQUE_COEF`, `MAX_RPM`) kept from PyFlyt's `cf2x.yaml` because Menagerie's `<motor gear=...>` actuator model doesn't directly map to our PID's RPM-squared thrust formulation — adopting it would require rewriting the controller.

**Controller:** pure-Python cascaded PID in [core/position_controller.py](core/position_controller.py), implementing PyFlyt mode 7 (position setpoint → desired velocity → desired thrust + attitude → motor PWMs). Gains match PyFlyt's `cf2x.yaml` exactly. Physics steps at 240 Hz; control steps at 120 Hz (one PID update = two `mj_step` calls).

**Gymnasium env:** [envs/quidditch/simple_env.py](envs/quidditch/simple_env.py). The env owns no MuJoCo state directly — it constructs a `core.world.World` on first `reset()` from a list of `SceneFragment` objects (cf2x assets + cf2x fragment + arena wall + hoop) and runs through a `Quadrotor` view bound to that world.

**Scene composition:** [core/mjcf/](core/mjcf/) provides `SceneFragment` (MJCF chunks + binary asset bytes) and `build_mjcf(opts, fragments)` (string concat under one `<mujoco>` root). No file `<include>` directives, no temp files — all MJCF assembly happens in memory at sim-init via `MjModel.from_xml_string(xml, assets=...)`. This makes "add a second drone" a one-line append rather than an XML rewrite, in preparation for Phase 2 multi-drone.

**Multi-drone forward-compat:** the `World` / `Quadrotor` split exists because two MjModels can't share a contact world — multi-drone must share a single `MjModel`, with one `Quadrotor` view per drone. The 32 cf2 collision hulls vendored alongside the visuals are opt-in: pass `with_collision_meshes=True` to `cf2x_assets()` and `with_collisions=True` to each `cf2x_fragment(prefix=...)` to enable drone-drone + drone-floor collisions (bitmask `1`; hoop and arena-wall stay phase-through on bit `0`). Default-off keeps the single-drone path byte-identical.

**Hoop scoring:** [envs/quidditch/scoring.py](envs/quidditch/scoring.py) — `GeomDistanceScorer.overlaps()` returns an `(N drones × M hoops)` boolean matrix from `mujoco.mj_geomDistance` between each drone's invisible probe sphere and each hoop's invisible score tube. The earlier signed-distance plane crossing was replaced because `mj_geomDistance` is the right primitive for "is this geom inside that geom" and doesn't involve the contact solver. (A first MuJoCo-era attempt used a `<contact><pair>` with `solimp="0 0 ..."` to read overlap from contact reports; it pinned the drone at the tube boundary with ~0.1 N residual force.)

**Camera:** the fixed scene camera (`eye` + `lookat` in `config/camera.toml`) drives both the live MuJoCo viewer pose and the offscreen renderer used by the per-checkpoint training video callback. `make camera-test` renders a hover flight through this camera to mp4 + a still PNG of the last frame, so iterating on camera angle doesn't require launching a training run.

**macOS OpenMP:** `train_ppo.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time to suppress the libomp double-init abort that conda environments produce on Apple Silicon when SB3's `SubprocVecEnv` spawns workers.

---

## Roadmap

| Phase | Status |
|---|---|
| 1 — Foundation, hover smoke test (PyFlyt) | Done — superseded by MuJoCo migration |
| 2 — Game design (arena, hoop, reward constants) | Done |
| 3 — Gymnasium env (`QuidditchSimpleEnv`) | Done |
| 4 — PPO training on PyFlyt | Done — results obsolete on new dynamics |
| 5 — Evaluation, promote workflow | Done — promote/repro pipeline carries over |
| 6 — MuJoCo migration (sim, controller, scoring) | Done (2026-04-24 → 2026-05-01 refactor + Menagerie cf2 drop-in) |
| 7 — Retrain on MuJoCo dynamics | **Next** |
| 8 — Multi-drone (collision flags ready) | Backlog |
| — | Multiple hoops |
| — | Carryable quaffle |
| — | Opposing drone (multi-agent self-play) |
| — | Seeker / snitch dynamic |
