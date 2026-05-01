# Drone Quidditch Sim

A reinforcement learning project that trains a quadcopter drone to fly through a goal hoop in a simplified analogue of Quidditch. Built on [PyFlyt](https://github.com/jjshoots/PyFlyt) (PyBullet physics) and trained with Stable-Baselines3 PPO.

**Current status:** First milestone complete. Training has converged — run `20260416_190850` achieved `mean_reward = 8.65 ± 0.01` at 1.95 M steps. Next step: promote and evaluate.

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

A single drone spawns at the center of a 3 m-radius circular arena. It must learn to fly forward and pass through a 50 cm-diameter hoop mounted 2 m above the ground, 2 m from the arena center. The environment is a Gymnasium-compatible custom env (`QuidditchSimpleEnv`) wrapping the PyFlyt simulator.

The physics run at 20 Hz (PyFlyt's QuadX default), giving 2 400 steps per 120-second episode.

---

## Game rules

| Constant | Value |
|---|---|
| Arena radius | 3 m |
| Hoop diameter | 0.5 m |
| Hoop center | (2, 0, 2) m |
| Hoop orientation | plane perpendicular to ground, outward normal [1, 0, 0] |
| Drone start | (0, 0, 0) — ground level, arena center |
| Episode length | 120 s (2 400 steps at 20 Hz) |
| Flight mode | PyFlyt mode 7 — position setpoint [x, y, yaw, z] |

**Scoring:** The drone must approach the hoop from behind (negative signed distance), enter the aperture, and travel at least 10 cm through the plane. Hovering at the boundary without fully crossing does not score — a two-phase crossing detector prevents that exploit.

**Termination:** Episode ends on score, crash (z < 0.05 m after 30-step grace), out-of-bounds (> 3 m from origin), or time limit.

---

## Observation & action space

**Observation (16-dim, continuous):**
- Drone state from PyFlyt (position, velocity, angular velocity, orientation)
- Unit vector from drone to hoop center
- Signed distance to hoop plane

**Action (4-dim, continuous, normalized to [−1, 1]):**
Normalized delta applied to the running position setpoint each step:

| Dim | Axis | Scale |
|---|---|---|
| 0 | Δx | ±0.2 m |
| 1 | Δy | ±0.2 m |
| 2 | Δyaw | ±0.5 rad |
| 3 | Δz | ±0.1 m |

Altitude is clamped to [0.01, 4.0] m after each update.

---

## Reward function

| Event | Reward |
|---|---|
| Each step | −(dist\_to\_hoop / 3) × 0.01 |
| Fly through hoop | +10 |
| Crash or out-of-bounds | −2 |

The per-step distance penalty provides a dense gradient toward the hoop. The +10 bonus clearly dominates any accumulated step penalty, so the agent is strongly incentivised to score quickly rather than hover in place.

---

## Requirements

- macOS (Apple Silicon tested; Linux should work with minor adjustments)
- [Conda / Miniforge](https://github.com/conda-forge/miniforge)
- Python 3.11 (enforced — see [architecture notes](#architecture-notes))

---

## Installation

```bash
# 1. Clone this repo
git clone <this-repo-url>

# 2. Create the conda environment (installs PyBullet, PyFlyt, SB3, imageio, etc.)
make install

# 3. Sanity-check the env
make check-sim      # headless (fast)
make check-gui      # opens PyBullet GUI for visual inspection
```

`make install` is idempotent — it creates the environment on first run and updates it on subsequent runs.

> **Why not `pip install pybullet`?** On macOS 26 (Tahoe), PyBullet's PyPI wheels are incompatible and the source build fails on the new SDK headers. The conda-forge binary works. Always `conda install pybullet` before `pip install` anything else.

> **Never run `pip install "stable-baselines3[extra]"`** — it pulls in numpy ≥ 2.4 which conflicts with PyFlyt's (stale) metadata constraint and breaks the environment. Install extras individually if you need them.

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
- `videos/` — rendered episode clips every 100 k steps
- `best_model.zip` — best checkpoint by eval reward
- `config_snapshot.toml` — exact config used (for `make repro`)
- `info.toml` — run metadata and finish timestamp

```bash
make tensorboard                   # launch TensorBoard across all runs
make tensorboard RUN_NAME=ppo_hoop # just one run name
make list-runs                     # list all trials and promoted models
```

### Evaluate

```bash
make eval                          # open PyBullet GUI, 10 episodes (latest trial)
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

### Utilities

```bash
make clean          # remove __pycache__ and .pyc files
```

---

## Training configuration

All PPO hyperparameters live in [config/training.toml](config/training.toml). CLI flags `--timesteps`, `--n-envs`, `--lr`, `--seed` override individual values at runtime.

```toml
[training]
total_timesteps = 2_000_000
n_envs          = 8          # SubprocVecEnv (parallel)
seed            = 42

[ppo]
n_steps    = 1024
batch_size = 256
n_epochs   = 10
lr         = 3e-4
gamma      = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef   = 0.01

[eval]
eval_freq_steps = 50_000
n_eval_episodes = 10

[callbacks]
checkpoint_freq_steps = 50_000
video_freq_steps      = 100_000
video_fps             = 20
```

**Key tuning notes from prior runs:**
- `ent_coef = 0.05` caused instability on this task — the entropy bonus dominated the sparse hoop reward. Keep it at 0.01.
- `n_steps = 2048` with 8 envs gave only 122 policy updates per 2 M steps. Halving to 1024 doubled update frequency and stabilised learning.
- Training uses `SubprocVecEnv` (8 workers) for the training env and `DummyVecEnv` (1 worker) for evaluation — the type mismatch warning from SB3 is intentional and suppressed.

---

## Project layout

```
quidditch-sim/
├── envs/
│   ├── __init__.py
│   └── quidditch_simple_env.py   ← QuidditchSimpleEnv (canonical Gymnasium env)
├── config/
│   └── training.toml             ← PPO hyperparameters
├── scripts/
│   ├── check_env.py              ← sanity checks (SB3 checker, zero policy, scripted)
│   ├── train_ppo.py              ← SB3 PPO training entry point
│   ├── eval_ppo.py               ← evaluation script
│   └── callbacks.py              ← checkpoint + video recording callbacks
├── demo/
│   ├── menu.py                   ← interactive demo selector (`make demo`)
│   ├── hover_demo.py             ← hover smoke test (Quidditch arena)
│   ├── waypoint_demo.py          ← triangular waypoint flight (empty scene)
│   └── camera_test.py            ← record hover through fixed cam → mp4
├── models/                       ← promoted models (tracked in git)
│   └── ppo_hoop_YYYYMMDD_HHMMSS/
│       ├── best_model.zip
│       ├── config.toml           ← config snapshot (for make repro)
│       └── run_info.toml
├── runs/                         ← training artifacts (gitignored)
│   └── ppo_hoop/
│       └── YYYYMMDD_HHMMSS/
│           ├── PPO_1/            ← TensorBoard logs
│           ├── checkpoints/
│           ├── videos/
│           ├── best_model.zip
│           ├── info.toml
│           └── config_snapshot.toml
├── Makefile
├── environment.yml
└── requirements.txt
```

---

## Architecture notes

**Simulator:** [PyFlyt](https://github.com/jjshoots/PyFlyt) provides quadcopter flight dynamics on top of PyBullet. We use `QuadXDrone` in flight mode 7 (position setpoint), which abstracts away low-level motor control and lets the RL agent reason at the level of desired position deltas.

**Gymnasium env:** `QuidditchSimpleEnv` in [envs/quidditch_simple_env.py](envs/quidditch_simple_env.py) wraps PyFlyt. It handles observation construction, reward shaping, episode termination, and 3D visualization (torus mesh hoop, cylinder mesh arena wall).

**Hoop crossing detection:** Two-phase to prevent boundary exploitation:
1. **Arm:** drone crosses from negative to positive signed distance while inside the hoop aperture → arm flag set.
2. **Confirm:** drone travels ≥ 0.1 m past the plane (HOOP_CROSSING_MARGIN). Retreating resets the arm.

**Visualization:** Hoop rendered as an inline torus mesh (no external URDF). Arena wall is a single open-cylinder mesh. PyBullet noise and startup messages are suppressed via `ctypes` FD redirection with a `fflush`-before-restore to avoid buffered output flooding the terminal at exit.

**Python 3.11 requirement:** Python 3.14 creates an unsatisfiable numpy version conflict between PyFlyt (`numpy<2`) and pandas (`numpy>=2.3.3`). 3.11 has no such conflict. PyFlyt's `numpy<2` metadata constraint is stale — it works fine with numpy 2.4.4 at runtime.

**macOS OpenMP:** `train_ppo.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time to suppress the libomp double-init abort that conda environments produce on Apple Silicon.

---

## Roadmap

| Phase | Status |
|---|---|
| 1 — Foundation (PyFlyt setup, hover smoke test) | Done |
| 2 — Game design (arena, hoop, reward constants) | Done |
| 3 — Gymnasium env (`QuidditchSimpleEnv`) | Done |
| 4 — PPO training (converged at 1.95 M steps) | Done |
| 5 — Evaluation (promote model, measure metrics) | **Next** |
| 6 — Curriculum (randomize hoop position) | Backlog |
| — | Multiple hoops |
| — | Carryable quaffle |
| — | Opposing drone (multi-agent self-play) |
| — | Seeker / snitch dynamic |
