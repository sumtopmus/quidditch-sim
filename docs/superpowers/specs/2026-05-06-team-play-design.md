# Team Play (Phase 2): asymmetric Red-attacker vs Blue-defender, single hoop

**Date:** 2026-05-06
**Status:** Design — pending implementation plan
**Branch:** `feature/team-play`

## 1. Summary

Phase 2 of Drone Quidditch. Adds a second drone (Blue, defender) to the existing single-drone, single-hoop env. Red attacks, Blue defends. Both drones are mechanically identical; team roles are configuration. Built so Phase 3 (mirror-symmetric two-hoop) is a config flip, not a rewrite.

Three new game mechanics:
- **Soft tag** — proximity reward for Blue when Blue's probe enters a 0.3 m sphere around Red. Entry pulse + per-step duration reward, gated by a 1 s post-exit cooldown on the entry pulse.
- **Hard crash** — drone-vs-drone contact above 1.5 m/s relative velocity terminates the episode. Asymmetric attribution: Blue +20, Red −20 (mirrors team roles).
- **Wall collisions** — arena wall becomes physically collidable via a ring of 16 box geoms; high-velocity wall hits crash with the same 1.5 m/s threshold as floor and drone-drone.

Training is single-agent from SB3's perspective: a thin wrapper exposes one side of the multi-agent env to PPO while a frozen "opponent" (scripted, zero, or loaded checkpoint) drives the other. Phase 2a trains Red against scripted Blue; Phase 2b freezes Red and trains Blue against it; Phase 2c flips back. PettingZoo `ParallelEnv` is the underlying interface so multi-agent training (RLlib, league play) is a future swap of the wrapper, not the env.

## 2. Goals & non-goals

**Goals:**
- Two-drone, single-hoop env with stable single-side training via SB3 PPO.
- Reward shape that produces visibly distinct attacker and defender behavior without scripting.
- Asymmetric architecture that role-swaps via config (Phase 3 prerequisite).
- Deterministic canary scenario — analog of `make check-sim` for the team env.

**Non-goals (deferred):**
- Symmetric two-hoop play (Phase 3).
- Self-play or league training where both policies update simultaneously (Phase 4).
- Snitch / quaffle / multi-drone teams (Phase 3+).
- Switching RL libraries (SB3 stays for v1).
- 1v2, 2v2, or any non-1v1 team size for v1.

## 3. Game design

**Setup.** Same arena (3 m radius, 6 m diameter, 2 m wall height). One hoop at (2, 0, 2) m, plane perpendicular to ground, outward normal +x. One Red drone, one Blue drone.

**Roles.** Red attacks (scoring through the hoop earns reward). Blue defends (preventing scoring + tagging Red earns reward).

**Starting positions.**
- Red: ground level. Random position in arena (when `randomise_start=true`, matches existing env) or fixed at (0, 0, 0).
- Blue: hovering at **(1.0, 0.0, 1.5)**, yaw = π (facing arena center). One meter in front of the hoop center along the −x axis, slightly below hoop height. Blue's start is fixed across episodes for v1.

**Action.** Same as single-agent env: 4-d normalized delta-setpoint `[dx, dy, dyaw, dz] ∈ [-1, 1]^4`, scaled by `[0.2, 0.2, 0.5, 0.1]`, applied to per-step position setpoint. Mode 7. Both drones use identical control.

**Episode termination events** (any one ends the episode for both agents):
- Red scores through the hoop (correct direction crossing).
- Either drone goes out of bounds (`|drone_xy| > ARENA_RADIUS`).
- Either drone crashes into floor (z < 0.05 after takeoff grace) or wall (high-velocity contact).
- Drone-vs-drone contact at relative velocity > 1.5 m/s.
- Episode timeout (30 s — `episode_seconds` from existing config).

**Tag events** (do not terminate; reward pulse only):
- **Tag entry** — Blue's probe enters the 0.3 m tag sphere around Red, gated by 1 s cooldown after the previous exit.
- **Tag duration** — per simulation step while Blue is inside Red's tag sphere.

**Reward table** (per agent, per step). Sign convention: every interaction event has equal-and-opposite Red/Blue magnitudes. Solo events (floor, OOB, wall by one drone) penalize only the crasher.

| Event | Blue reward | Red reward | Episode |
|------|------|------|------|
| Distance shaping (always on) | −(\|blue_pos − midpoint\| / R)·0.01 | −(\|red_pos − hoop_center\| / R)·0.01 | continue |
| Red scores | −10 | +10 | end |
| Tag entry (cooldown-gated) | +5 | −5 | continue |
| Tag duration (per step in zone) | +0.02 | −0.02 | continue |
| Drone-vs-drone contact, \|v_rel\| > 1.5 m/s | +20 | −20 | end |
| Drone-vs-drone contact, \|v_rel\| ≤ 1.5 m/s | 0 | 0 | continue |
| Red floor / OOB / wall crash | 0 | −20 | end |
| Blue floor / OOB / wall crash | −20 | 0 | end |
| 30 s timeout | shaped only | shaped only | end |

`midpoint = α·red_pos + (1−α)·hoop_center`, default α = 0.5 (tunable via config).

**Magnitude commentary:**
- Score: +10 for Red is the headline reward.
- Drone-drone crash dominates at ±20 — decisive.
- Tag entry +5 is a sharp learning signal; tag duration +0.02/step accumulates up to roughly +12 over a 30 s glued episode (intentionally exceeds entry pulse — persistent harassment should pay more than hit-and-run).
- Distance shaping is small (~0.01/step magnitude) — there to provide gradient, not to dominate.

## 4. Architecture

```
repo/
├── core/
│   ├── drone/cf2x.py                     ← unchanged (already supports prefix + with_collisions)
│   └── policies/                         ← NEW directory
│       └── warm_start.py                 ← NEW: warm_start_ppo() — input-layer augmentation
├── envs/
│   └── quidditch/
│       ├── constants.py                  ← extend: TAG_RADIUS, CRASH_VEL_THR, etc.
│       ├── scene.py                      ← extend: arena_wall_fragment(with_collisions=False)
│       ├── scoring.py                    ← unchanged (N×M scorer is already team-ready)
│       ├── tagging.py                    ← NEW: TagDistanceScorer
│       ├── crash.py                      ← NEW: CrashDetector
│       ├── simple_env.py                 ← unchanged (single-agent canary stays)
│       ├── team_env.py                   ← NEW: QuidditchTeamEnv(ParallelEnv)
│       └── opponents.py                  ← NEW: Opponent Protocol + scripted/frozen impls
└── scripts/
    ├── _train_common.py                  ← NEW: shared SB3 setup factored out of train_ppo.py
    ├── train_team_ppo.py                 ← NEW: team-env training entry
    ├── eval_team.py                      ← NEW: head-to-head eval
    └── check_team_env.py                 ← NEW: deterministic canary (`make team-check`)
```

**Module responsibilities:**

- **`team_env.QuidditchTeamEnv`** (PettingZoo `ParallelEnv`): owns the World, two Quadrotor views, scorers/detectors, per-pair tag state machines. Returns dicts keyed by `agent_id` from `step()`/`reset()`.
- **`opponents.OpponentControlledEnv`** (gym.Env): wraps the team env, picks one agent as the learner, drives the other from an `Opponent` instance. Only object SB3 sees.
- **`opponents.Opponent`** (Protocol): stateless contract — `reset()`, `act(obs) -> action`. Implementations: `BeelineRed`, `BeelineBlue`, `IntercepterBlue`, `ZeroOpponent`, `FrozenPolicyOpponent`, `MixtureOpponent`.
- **`tagging.TagDistanceScorer`**: per-pair `mj_geomDistance(probe, tag_sphere)` — same pattern as `GeomDistanceScorer` for the hoop. Returns booleans; the entry/exit/cooldown state machine lives in the env.
- **`crash.CrashDetector`**: iterates `data.contact[:ncon]` each step, classifies pairs as `drone-drone | drone-floor | drone-wall`, computes relative velocity along contact normal for drone-drone events.
- **`core.policies.warm_start.warm_start_ppo`**: builds a fresh PPO with the input layer augmented from a loaded checkpoint's weights — copies `W_old` into `W_new[:, :old_dim]` and small-init initializes `W_new[:, old_dim:]`.

**Symmetry & role-swap:** the env never branches on "am I Red or Blue." It looks up `attacker_id` and `defender_id` from config and applies the reward table. Phase 3 mirror mode = construct two `QuidditchTeamEnv` instances (or one with two attacker/defender pairs) where each agent is attacker for its target hoop and defender for the opponent's. No env code changes.

## 5. MJCF additions

**Per drone fragment** (added to `cf2x_fragment` or via a sibling factory `tag_sphere_fragment(prefix, rgba)`):
- `{prefix}_tag_sphere` — sphere geom, radius 0.3 m, child of drone body, `contype=0 conaffinity=0`, `group="3"` (hidden by default), translucent rgba (red `1 0 0 0.15` for Red, blue `0 0 1 0.15` for Blue) for debug visibility.

**Drone collision flip:**
- `cf2x_assets(with_collision_meshes=True)` once per scene (covers both drones).
- `cf2x_fragment(prefix, with_collisions=True)` per drone. Activates the 32 vendored collision hulls on `contype=1 conaffinity=1`.

**Arena wall — new collidable ring:** `arena_wall_fragment(radius, height, with_collisions=False)` extended:
- Visible mesh stays as-is (`contype=0 conaffinity=0`).
- When `with_collisions=True`, add 16 thin box geoms arranged in a ring at `radius = ARENA_RADIUS`, each spanning 22.5° of arc:
  - Half-extents approximately `(ARENA_RADIUS · sin(11.25°), 0.025, height/2)` ≈ `(0.585, 0.025, 1.0)` m.
  - Positioned at `(R·cos(θ_i), R·sin(θ_i), height/2)` for θ_i = i·22.5° + 11.25°, i ∈ {0, …, 15}.
  - Each box rotated to be tangent to the perimeter (yaw = θ_i + π/2).
  - `contype=1 conaffinity=1`, `group="3"` (hidden in viewer).

The 16-segment count is a deliberate trade-off — enough segments that drones don't notice the polygonal approximation at 3 m radius, few enough that contact-pair iteration is cheap. Number is tunable; if behavior issues appear, bump to 32.

**World fragment list at construction:**

```python
fragments = [
    cf2x_assets(with_collision_meshes=True),
    cf2x_fragment(prefix="red_0",  with_collisions=True),
    cf2x_fragment(prefix="blue_0", with_collisions=True),
    tag_sphere_fragment("red_0",  rgba=(1.0, 0.0, 0.0, 0.15)),
    tag_sphere_fragment("blue_0", rgba=(0.0, 0.0, 1.0, 0.15)),
    arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
    hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
]
```

## 6. Observation space

**22-dim per agent**, layout chosen so first 16 dims match the existing single-agent env exactly (warm-start from single-agent checkpoint is weight surgery, not a custom features extractor):

```
slots  0:12  self_state          ang_vel(3) + euler(3) + lin_vel_body(3) + lin_pos(3)
slots 12:15  unit_goal_vec       attacker: (hoop_center − self_pos) / |hoop_center − self_pos|
                                 defender: (midpoint    − self_pos) / |midpoint    − self_pos|
slot     15  signed_dist_to_hoop_plane / ARENA_RADIUS    (always the same signal regardless of role)
slots 16:19  opp_pos_rel         opp_pos     − self_pos      (world frame, NOT unit-normalized)
slots 19:22  opp_vel_rel         opp_lin_vel − self_lin_vel  (world frame)
```

All "relative" terms are world-frame; `self_lin_vel` is body-frame (matches existing convention from `simple_env._obs`). The defender's `unit_goal_vec` reuses the same 3 obs slots as the attacker's — a single policy network architecture handles either role by config (relevant for self-play later, where both sides can share an architecture).

**Critical for warm-start:** slots 0:15 use *exactly* the same encodings as `simple_env._obs` — `unit_to_*` is unit-normalized (matching `unit_to_hoop = vec_to_hoop / (dist + 1e-8)` from the existing env), `signed_dist_norm` is signed projection along `HOOP_OUTWARD_NORMAL` divided by `ARENA_RADIUS`. The contractual freeze on slots 0:15 is what makes warm-start a weight-surgery operation rather than a behavioral retrain.

**Slot 15 is the same signal for both roles** — `signed_dist((self_pos − hoop_center), HOOP_OUTWARD_NORMAL) / ARENA_RADIUS`. For the attacker this is the existing scoring-relevant cue ("am I on the inside or outside of the hoop plane?"). For the defender it's positional context ("am I in front of the hoop, behind it, or level with the plane?"). Same value, useful to both.

**The 16-slot prefix is contractually frozen** to match `simple_env.py`. A comment in `simple_env._obs` should reference this dependency.

## 7. Episode lifecycle

**Per-step sequence in `team_env.step(actions)`:**

```
1. Apply each agent's delta to its setpoint; clamp to arena bounds and altitude limits.
2. Assign setpoints to each Quadrotor; world.step() advances both drones.
3. Read positions + velocities for both drones.
4. Run scorers and detectors:
   a. hoop_scorer.overlaps()       — (1×1) bool: did Red's probe overlap hoop tube?
   b. tag_scorer.in_zone()         — (Blue probe inside Red tag sphere?)
                                     and the symmetric (Red probe inside Blue tag sphere?)
                                     v1 only uses the first; symmetric is for Phase 3.
   c. crash_detector.events()      — {solo: {prefix→bool}, drone_drone: (vrel, contact_pt)|None,
                                       wall: {prefix→(vrel)|None}}
5. Update per-agent state machines:
   a. Hoop crossing state          — same as simple_env (entry side, exit side).
   b. Per-pair tag state           — IDLE / IN_ZONE / COOLDOWN (see below).
6. Compute terminations and rewards from the table.
7. step_count += 1; check timeout (truncation if no terminal event).
8. Return dicts: obs / rewards / terminations / truncations / infos, keyed by agent_id.
```

**Tag state machine** (per attacker–defender pair; v1 has one pair):

```
state: TagState ∈ {IDLE, IN_ZONE, COOLDOWN}
cooldown_ticks: int = 0
COOLDOWN_TICKS = round(TAG_COOLDOWN_SECONDS / step_period)   # 1 s @ 30 Hz = 30

each step, given in_zone_now (bool from TagDistanceScorer):
  IDLE:
    if in_zone_now:
      fire entry_pulse (Blue +5, Red -5)
      state ← IN_ZONE
  IN_ZONE:
    fire duration_reward (Blue +0.02, Red -0.02)
    if not in_zone_now:
      state ← COOLDOWN
      cooldown_ticks ← COOLDOWN_TICKS
  COOLDOWN:
    if in_zone_now:
      fire duration_reward (no entry pulse — that's what the cooldown gates)
    cooldown_ticks -= 1
    if cooldown_ticks ≤ 0:
      state ← IDLE if not in_zone_now else IN_ZONE_QUIET
```

`IN_ZONE_QUIET` is conceptually IN_ZONE without retroactively firing the entry pulse — implemented as setting state back to IN_ZONE directly (no pulse fire on this step). This prevents a Blue camping in zone from earning a fresh entry pulse every cooldown cycle.

**Crash detector** (`envs/quidditch/crash.py`):

```python
class CrashDetector:
    def __init__(self, world: World, drone_prefixes: list[str]):
        self._model = world.model
        self._data  = world.data
        # Cache: geom_id → {drone_prefix} or "floor" or "wall"
        self._geom_owner: dict[int, str] = {}
        self._build_owner_map(drone_prefixes)

    def events(self) -> dict:
        """Returns:
          {
            "solo_floor": {prefix: True}          # drone hit floor (above takeoff grace)
            "wall": {prefix: vrel_along_normal}   # drone hit wall (one row per drone in contact)
            "drone_drone": (a_pref, b_pref, |vrel|, contact_pt) or None
          }
        """
        # Iterate self._data.contact[:self._data.ncon].
        # For each contact (geom1, geom2), look up owners.
        # For drone-drone: compute |v_rel · contact_normal| at contact point,
        #   keep the largest magnitude across all contact pairs this step.
        # For drone-wall and drone-floor: same, per drone.
```

OOB stays detected env-side from positions (`np.linalg.norm(drone_xy) > ARENA_RADIUS`) — defensive, since collidable walls *should* prevent OOB but tunneling at high velocity is theoretically possible.

**Termination and reward composition.** Multiple events can fire in one step. The env applies *all* their reward effects, and terminates the episode if *any* terminal event fires. `terminated` is set to True for both agents when terminal — simpler than per-agent terminals and matches `score → reset` semantics. `truncated` only fires on timeout with no terminal event.

**Takeoff grace.** `TAKEOFF_GRACE_STEPS = 30` (1 s at 30 Hz control) per drone, individually. Red gets it (spawns on ground); Blue's grace counter starts at 0 (spawns hovering). Solo-floor-crash detection is gated by each drone's grace counter.

**Reset.** First `reset()` builds the World from fragments and constructs the two `Quadrotor` views, the scorers, and the `CrashDetector`. Subsequent resets reuse the world (matches `simple_env`'s pattern) — sets new positions, zeroes velocities and setpoints, resets state machines.

## 8. Opponent ABC and warm-start

**Opponent contract** (`envs/quidditch/opponents.py`):

```python
class Opponent(Protocol):
    """Drives a single agent in a QuidditchTeamEnv.

    Stateless from the env's perspective. Called once per env step with the
    agent's own observation; returns a normalized 4-d action.
    """
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...   # shape (4,), in [-1, 1]
```

**Reference implementations:**

- `ZeroOpponent`: returns zeros — agent holds initial setpoint.
- `BeelineRed`: steers along `obs[12:15]` (`unit_to_hoop_center`), clipped to action scale. Concretely: `action = clip([obs[12], obs[13], 0.0, obs[14]], -1, 1)` — x and y components into translation slots, z component into altitude slot, no yaw. No additional state.
- `BeelineBlue`: identical code shape; steers along `obs[12:15]` (`unit_to_midpoint`), clipped to action scale. The midpoint is computed env-side from the role config's `α`, so the scripted policy needs no knowledge of `α` or `hoop_center` — it just follows the defender's pre-computed unit goal vector.
- `IntercepterBlue`: computes a lookahead target `red_pos + lookahead · red_velocity` from `obs[16:19]` (relative red position) and `obs[19:22]` (relative red velocity), then steers toward it. Lookahead capped to a max distance (default 0.5 m); falls back to `BeelineBlue` behavior if cap is exceeded.
- `FrozenPolicyOpponent(model_path, deterministic=False)`: wraps a loaded SB3 PPO checkpoint.
- `MixtureOpponent([(weight, opp), ...])`: picks one opponent per episode (weighted random) — supports league-style training without changing the wrapper.

All scripted policies read the same observation vector the learnable agent gets. No privileged information.

**Spec-string parsing** (`envs.quidditch.opponents.from_spec`):

```
beeline_blue                                → BeelineBlue()
beeline_red                                 → BeelineRed()
intercepter_blue:lookahead=0.5              → IntercepterBlue(lookahead=0.5)
zero                                        → ZeroOpponent()
frozen:path/to/best_model.zip               → FrozenPolicyOpponent(path)
mixture:0.5*beeline_blue,0.5*frozen:path    → MixtureOpponent([(0.5, ...), (0.5, ...)])
```

**Wrapper** (`envs/quidditch/opponents.py`):

```python
class OpponentControlledEnv(gym.Env):
    """Reduces a QuidditchTeamEnv to a single-agent Gym env from `learner_id`'s view.

    Each step:
      1. Wrapper queries `opponent.act(opp_obs)` for the frozen side.
      2. Wrapper builds the action dict {learner_id: learner_action, opp_id: opp_action}.
      3. Calls team_env.step(actions).
      4. Returns only learner_id's obs/reward/term/trunc/info to SB3.
    """
    def __init__(self, team_env: QuidditchTeamEnv, *, learner_id: str, opponent: Opponent):
        ...
```

This is the only SB3-facing class. Everything underneath is multi-agent.

**Warm-start mechanism** (`core/policies/warm_start.py`):

```python
def warm_start_ppo(
    old_checkpoint: Path,
    new_env: VecEnv,
    *,
    new_input_dim: int,         # 22
    old_input_dim: int = 16,
    new_dim_init_scale: float = 0.01,
    **ppo_kwargs,
) -> PPO:
    """Build a fresh PPO whose network weights are copied from `old_checkpoint`,
    with the input layer augmented from `old_input_dim` to `new_input_dim`.

    Slots [0:old_input_dim] of the new obs MUST match the old obs layout exactly.
    Extra `new_input_dim - old_input_dim` columns of the input weight matrix are
    initialized with N(0, new_dim_init_scale²). Other policy heads, intermediate
    layers, and biases transfer unchanged.

    Caller passes ppo_kwargs (n_steps, batch_size, lr, etc.) that the new PPO
    will use; the loaded checkpoint's hyperparameters are NOT inherited.
    """
```

Implementation sketch:

```python
old_model = PPO.load(old_checkpoint)
new_model = PPO("MlpPolicy", new_env, **ppo_kwargs)
old_sd = old_model.policy.state_dict()
new_sd = new_model.policy.state_dict()
for key in old_sd:
    if old_sd[key].shape == new_sd[key].shape:
        new_sd[key] = old_sd[key]
    elif key.endswith(".weight") and old_sd[key].dim() == 2 \
         and old_sd[key].shape[0] == new_sd[key].shape[0]:
        # Input layer: (hidden, old_input_dim) → (hidden, new_input_dim).
        new_sd[key][:, :old_input_dim] = old_sd[key]
        # Columns old_input_dim: stay at fresh init (small Gaussian via init_scale).
new_model.policy.load_state_dict(new_sd)
return new_model
```

Only Red has a warm-startable single-agent ancestor. Blue trains from scratch in Phase 2b regardless.

## 9. Training entry & config

**`scripts/train_team_ppo.py`** is a new sibling to existing `train_ppo.py`. Shared SB3 plumbing factored into `scripts/_train_common.py`.

**CLI:**

```
python scripts/train_team_ppo.py
    --learner   {red_0, blue_0}
    --opponent  <opponent_spec>
    [--warm-start <path-to-old-zip>]      # Red-only, optional
    [--config    config/training.toml]
    [--timesteps N] [--n-envs N] [--lr X] [--seed N]
```

**`config/training.toml` extensions** (everything else stays unchanged):

```toml
[training]
run_name        = "team_red_v1"
total_timesteps = 5_000_000
n_envs          = 8
seed            = 42

[training.ppo]
n_steps    = 1024
batch_size = 256
n_epochs   = 10
lr         = 3e-4         # back to 3e-4 for fresh team training
gamma      = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef   = 0.01

[training.team]                           # NEW
learner          = "red_0"
opponent_spec    = "beeline_blue"
warm_start_from  = ""                     # empty = no warm-start

[training.team.warm_start]                # NEW
new_dim_init_scale = 0.01

[env]
randomise_start  = true                   # randomises Red's start; Blue's start is fixed
episode_seconds  = 30.0
mode             = "team"                 # NEW: "single" | "team" — selects env class

[env.team]                                # NEW
red_prefix       = "red_0"
blue_prefix      = "blue_0"
hoop_prefix      = "hoop_0"
midpoint_alpha   = 0.5                    # Blue shaping target weighting
tag_radius       = 0.3
tag_cooldown_s   = 1.0
crash_vel_thr    = 1.5
walls_collide    = true

[training.callbacks.video]
grid        = true
cells       = ["south", "east", "top", "tpv"]
cell_width  = 960
cell_height = 540
tpv_target  = "red_0"                     # NEW: which drone the TPV cam follows
```

`[env.team]` values are also constructor kwargs to `QuidditchTeamEnv` so unit-test setups don't need a toml round-trip.

**Eval script `scripts/eval_team.py`:**

```
python scripts/eval_team.py
    --red    <opponent_spec>
    --blue   <opponent_spec>
    --episodes 100
    [--render | --video <path.mp4>]
```

Outputs match summary: Red score rate, mean tag count per episode, mean tag duration, drone-drone crash rate, mean episode length, per-side reward distributions. No SB3 dependency — both sides are `Opponent` instances driving the team env directly.

**Run directory layout** (matches existing pattern):

```
runs/team_red_v1/20260506_120000/
├── PPO_1/                       # SB3 TB logs
├── checkpoints/                 # periodic .zip
├── videos/                      # grid recordings, every Nth eval
├── best_model.zip
├── final_model.zip              # only present if training completes cleanly
└── run_info.toml                # seed, timesteps, opponent_spec, warm_start_from, git SHA
```

`run_info.toml` is the reproducibility record — `learner`, `opponent_spec`, and `warm_start_from` all logged.

**Makefile additions:**

```make
train-team-red:           ## Phase 2a: train Red against scripted Blue
	$(PYTHON) scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue

train-team-red-warm:      ## Phase 2a (warm-start): train Red from single-agent best
	$(PYTHON) scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue \
	    --warm-start models/$(WARM_START)/best_model.zip

train-team-blue:          ## Phase 2b: train Blue against frozen Red
	$(PYTHON) scripts/train_team_ppo.py --learner blue_0 \
	    --opponent frozen:models/$(RED)/best_model.zip

eval-team:                ## Head-to-head: RED=... BLUE=... [EPISODES=100]
	$(PYTHON) scripts/eval_team.py --red $(RED) --blue $(BLUE) --episodes $(or $(EPISODES),100)

team-check:               ## Deterministic team-env canary
	$(PYTHON) scripts/check_team_env.py
```

## 10. Verification

**Single-agent canary stays unchanged.** `simple_env.py` is not touched. `make check-sim` continues to print `SCORED at step 434 / total reward: 7.3837`. If that line moves, something in `core/` was modified inadvertently.

**New canary: `make team-check`** (`scripts/check_team_env.py`). Scripted full-episode run, both sides hand-coded and deterministic:

- Red plays `BeelineRed()` (fly straight at hoop center).
- Blue plays `BeelineBlue()` with α=0.5 (fly toward midpoint).
- Seed 42, Red's start fixed at (0, 0, 0) (`randomise_start=false` for canary).
- Run for full 30 s episode (or until terminated). Print every event:

```
[step    1]  red=(0.000,0.000,0.000)  blue=(1.000,0.000,1.500)  in_zone=False  contacts=0
[step  124]  TAG_ENTRY                                          blue +5.000  red −5.000
[step  125]  TAG_DURATION                                       blue +0.020  red −0.020
[step  127]  TAG_EXIT
[step  187]  CRASH drone_drone vrel=2.341 m/s  termination      blue +20.000  red −20.000
EPISODE END  steps=187  red_total=−6.81  blue_total=+18.43
```

Exact step numbers, vrel values, and totals are the fingerprint, locked in once at first stable build of the team env and held thereafter.

**Per-component check scripts:**

- `make team-check-mjcf` — build the team-env MJCF, assert: 2 drones, 2 tag-spheres (`contype=0`), 16 wall boxes (`contype=1`), expected `ngeom`.
- `make team-check-tag` — drives Red and Blue through scripted positions to exercise the tag state machine: enter → duration → exit → cooldown → re-enter (no entry pulse during cooldown) → cooldown ends → exit → re-enter (entry pulse fires). Asserts exact `(event, agent, reward)` sequence.
- `make team-check-crash` — three sub-cases: (a) Red into wall at 2.0 m/s → crash, Red −20; (b) Blue into Red at 0.5 m/s → contact, no crash; (c) Blue into Red at 2.0 m/s → crash, Blue +20 / Red −20.
- `make team-check-warm` — load single-agent `best_model.zip` via `warm_start_ppo`, run warm-started policy in team env with `obs[16:]` zeroed, assert action distribution on a fixed batch matches single-agent policy's actions within 1e-3 (warm-start preserves behavior on the old obs subspace).
- `make team-check-reset` — two episodes with the same seed produce identical event sequences.

**Reward-symmetry runtime assertion** in `team_env.step`:

```python
if __debug__:
    interaction_events = [score, tag_entry, tag_duration, drone_drone_crash]
    if any_fired(interaction_events):
        assert r_red + r_blue == sum_of_interaction_event_magnitudes_with_signs, \
            "interaction reward asymmetry detected"
```

Cheap sanity check that catches reward-table edits where Red's penalty doesn't match Blue's bonus. Disabled in optimized runs (`-O`); fires during tests and dev. Solo events (one drone's floor / wall / OOB) are explicitly asymmetric — the assertion skips them.

**Quality gate before Phase 2b:** after Phase 2a (Red trained vs `BeelineBlue`), `eval_team.py --red frozen:.../best_model.zip --blue beeline_blue --episodes 100` should show **Red score rate ≥ 80%**. If Red can't beat the hand-coded defender, we are not ready to put it against a learnable adversary.

## 11. Phasing / build sequence

This design implements through Phase 2c. Phase 3 (mirror) is enabled by it but is out of scope.

**Build order suggested for the implementation plan:**

1. **Foundation** — `arena_wall_fragment(with_collisions=...)` ring of boxes; `tag_sphere_fragment`; flip drone collision flags. Validate with `make team-check-mjcf` and a manual viewer session.
2. **Detectors** — `CrashDetector`, `TagDistanceScorer`. Validate with `make team-check-crash` and `make team-check-tag` (which also exercises the env's tag state machine, so this depends on (3)).
3. **`QuidditchTeamEnv`** — full env class with reward composition, state machines, lifecycle. Validate with `make team-check`.
4. **Opponents + wrapper** — `Opponent` Protocol, scripted impls, `OpponentControlledEnv`, `from_spec` parser.
5. **Training entry** — `_train_common.py`, `train_team_ppo.py`, config schema, run dir. Validate by running a 10k-step smoke training and confirming TB logs are written.
6. **Warm-start** — `core/policies/warm_start.py`. Validate with `make team-check-warm`.
7. **Eval** — `eval_team.py`. Run a head-to-head between two `BeelineX` opponents for sanity (no NaNs, expected match summary fields populated).
8. **Phase 2a training run** — `make train-team-red`. Goal: Red beats `BeelineBlue` with score rate ≥ 80% over 100 eval episodes.
9. **Phase 2b training run** — `make train-team-blue` against frozen Red from (8). Goal: Blue tag rate increases meaningfully over a `BeelineBlue` baseline.
10. **Phase 2c training run** (optional) — flip back, retrain Red against frozen Blue from (9).

Steps 1–7 are pure engineering. Steps 8–10 are training cycles, each potentially a multi-day session.

## 12. Open questions / future work

- **Self-play / league sampling** — `MixtureOpponent` is in v1 but actually composing a league-of-checkpoints curriculum is deferred. Phase 4.
- **Symmetric two-hoop mode** — Phase 3. The env's role-by-config design supports it; needs another `hoop_fragment` and a reward composition that handles "I am attacker for my target hoop AND defender for opponent's hoop." Estimated as moderate work, not a rewrite.
- **PettingZoo → RLlib migration** — when both policies need to update simultaneously. The env's `ParallelEnv` interface is the right starting point; the migration is the wrapper, not the env.
- **Per-agent termination** — the v1 design terminates the whole episode when any agent terminates. If "Red OOB but Blue keeps flying" turns out to provide useful training signal (e.g., for Blue's solo navigation), revisit.
- **Tag radius and crash threshold tuning** — both values (0.3 m, 1.5 m/s) are first-pass guesses. Watch behavior in early Phase 2a runs and adjust.
- **Midpoint α tuning** — start at 0.5; if Blue gets repeatedly beaten on cuts to the hoop, push toward hoop-side (e.g., α = 0.4 weights toward Red, so target is closer to hoop).
- **Warm-start Blue from a hover-trained policy?** — currently Blue trains from scratch. A pre-trained "stable hover" policy might bootstrap Phase 2b. Probably not worth the engineering for v1.
