# `blue_v7` — body-frame ego-centric obs + proactive intercept reward — Design

**Goal.** Train a new Blue defender (`ppo_hoop_blue_7`) with two structural changes versus `blue_v4`:
1. **Obs:** Replace world-frame relative vectors with body-frame ego-centric ones. The four blocks `unit_to_goal` (→ `vec_to_goal`, not normalized), `vec_to_hoop`, `opp_pos_rel`, `opp_vel_rel` all move into the Blue learner's body frame. `signed_dist_norm` is dropped (replaced by `vec_to_hoop` body, like `DUEL_V2_WORLD` already did). The "goal" point shifts from the existing `midpoint_alpha = 0.5` lerp (50% Red / 50% hoop) to `α = 0.6` (60% Red / 40% hoop) — closer to the threat.
2. **Reward:** Drop Blue's `HoopDistancePenalty(blue → midpoint)`. Keep `HoopAnchor(blue→hoop)` and `ZeroSumDistMirror(red→hoop)` (defender still must keep close to the hoop and benefit from Red being far). Add a new term, `InterceptShaping`, that activates only when Red is within `1.5 m` of the hoop and rewards Blue for closing on a short-horizon predicted future-Red point (`future_red = red_pos + 0.5 s · red_vel_world`).

Frozen Red opponent = `red_v1` (= `ppo_hoop_red_1_20260506_103058`). Red is unchanged — its 22-d `DUEL_V1_BODY` obs is what `OpponentControlledEnv` already feeds frozen opponents.

**Why now.** Two motivations.
- `blue_v4` defends from world-frame positions; the policy must internally compose drone yaw into every relative vector before reacting. Body-frame egocentric obs removes that latent step and matches the literature for quadrotor RL (where ego-frame inputs typically learn faster and generalize better across spawn poses, which we use under `random_start`).
- The existing Blue shaping (`HoopDistancePenalty + HoopAnchor + ZeroSumDistMirror`) trains Blue to *hover* near the midpoint or hoop; it does not directly reward intercepting Red's trajectory. The new `InterceptShaping` term gives gradient toward proactive interception when Red is threatening, while keeping the static "stay near hoop" pressure (`HoopAnchor`) so Blue still defaults to a useful defensive position when Red is far.

**Non-goals.**
- Re-training Red. Red is the existing frozen `red_v1`. We are not changing Red's obs, reward, or weights.
- Promoting `blue_v7`. Promotion is a separate user decision (`make promote`) after evaluating the trained checkpoint.
- Adding body-frame variants of every obs constant for *all* roles. Red continues to use `DUEL_V1_BODY`. Only Blue's spec is body-frame.
- Generic per-agent obs-spec selection in `QuidditchTeamEnv`. The refactor exposes a single `learner_id` + `learner_spec` pair (the non-learner gets `DUEL_V1_BODY` for back-compat with frozen Red). A future spec can generalize to "any agent → any spec" if needed.
- Keeping `OpponentControlledEnv._augment_learner_obs` around. The refactor folds the existing world-frame augmentation into `team_env`'s per-agent packer (under `learner_spec=DUEL_V2_WORLD`) so OCE becomes a pure pass-through for both `v2` and `v3` learners. `blue_v4`'s 25-d obs is reproduced byte-identically by the new in-env packer; the augmenter's logic moves, it does not disappear.
- Sweeping the new tuning knobs (`scale`, `lookahead_s`, `activation_dist`). Starting values are committed to `conf/reward/team_v3_intercept.yaml`; later sweeps can use the existing wandb sweep machinery.

## Design Decisions

### 1. New ObsSpec — `DUEL_V3_BODY_EGO` (25-d, body-frame)

New constants in [envs/quidditch/obs_spec.py](../../../envs/quidditch/obs_spec.py):

```python
VEC_TO_GOAL_BODY = ObsBlock(
    "vec_to_goal", dim=3, frame="body",
    notes="goal point - learner_pos, rotated into learner body frame; "
          "goal = α·red_pos + (1-α)·hoop_center (α from TeamConfig.midpoint_alpha)",
)
VEC_TO_HOOP_BODY    = ObsBlock("vec_to_hoop",  dim=3, frame="body")
OPP_POS_REL_BODY    = ObsBlock("opp_pos_rel",  dim=3, frame="body")
OPP_VEL_REL_BODY_EGO = ObsBlock(
    "opp_vel_rel", dim=3, frame="body",
    notes="(opp_vel_world - learner_vel_world) rotated into learner body frame; "
          "distinct from OPP_VEL_REL_BODY (body_mixed)",
)
```

Composed spec:

```python
DUEL_V3_BODY_EGO = ObsSpec((
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS,
    VEC_TO_GOAL_BODY,                      # replaces UNIT_TO_GOAL (world, normalized)
    VEC_TO_HOOP_BODY,                      # replaces VEC_TO_HOOP (world)
    OPP_POS_REL_BODY,                      # replaces OPP_POS_REL (world)
    OPP_VEL_REL_BODY_EGO,                  # replaces OPP_VEL_REL_WORLD
    CLOSING_RATE,                          # unchanged scalar
))
```

Registered in `SPEC_BY_NAME["DUEL_V3_BODY_EGO"]`. Dim = 25. New `conf/obs/duel_v3_body_ego.yaml`:

```yaml
name: DUEL_V3_BODY_EGO
n_stack: 3
```

`n_stack=3` matches `blue_v4` so the learning curves are comparable.

**Frame convention.** "body" here means the Blue learner's body frame. The rotation from world to body uses `R_wb = data.xmat[blue_body_id].reshape(3, 3)`, where MuJoCo's `xmat` stores the body's world-frame orientation (i.e. columns are body axes in world coords, so `v_world = R_wb @ v_body` and `v_body = R_wb.T @ v_world`).

**Helper.** A free function in `obs_spec.py`:

```python
def world_to_body(vec_world: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector from world frame into a body frame defined by R_wb,
    where R_wb is body-to-world (MuJoCo data.xmat convention)."""
    return (R_wb.T @ vec_world).astype(np.float32)
```

Pure function, no MuJoCo imports — keeps `obs_spec.py` dependency-free for legacy callers.

### 2. New reward term — `InterceptShaping`

New dataclass in [envs/quidditch/rewards/terms.py](../../../envs/quidditch/rewards/terms.py):

```python
@dataclass
class InterceptShaping:
    """Closing-velocity reward on Red's short-horizon predicted position,
    active only when Red is within `activation_dist` of the hoop.

    Mirrors ClosingVelInTagZone in structure (zero-floored closing rate
    weighted by `scale`), but gated by dist_red_to_hoop instead of tag_during.
    Non-zero-sum: only the defender is rewarded — Red is not penalised on
    this signal (Red already has its own HoopDistancePenalty pulling it
    toward the hoop).
    """
    scale: float
    lookahead_s: float          # informational; the env consumed this when
                                # populating future-red distances on StepState.
    activation_dist: float
    defender: str               # "blue_0"

    def compute(self, state: StepState) -> dict[str, float]:
        out = {a: 0.0 for a in state.agent_ids}
        if state.dist_red_to_hoop >= self.activation_dist:
            return out
        d, d_prev = state.dist_def_to_future_red, state.dist_def_to_future_red_prev
        closing = (d_prev - d) / state.step_period
        out[self.defender] += self.scale * max(0.0, closing)
        return out
```

Starting params: `scale = 0.05`, `lookahead_s = 0.5`, `activation_dist = 1.5`.

**StepState additions** ([envs/quidditch/rewards/stack.py](../../../envs/quidditch/rewards/stack.py)):

```python
dist_def_to_future_red:      float = 0.0
dist_def_to_future_red_prev: float = 0.0
```

Defaulted to 0 so single-agent + existing tests don't need to populate them. Team env computes both each step (see Decision 4).

**Why store `lookahead_s` on the term even though the env computes the distances.** Two reasons: (1) the YAML stays self-documenting — a reader sees the prediction horizon next to the activation gate and scale; (2) a future `__post_init__` check can warn if the YAML's `lookahead_s` doesn't match the value the env was constructed with (we wire that check in a follow-up — not part of this spec).

### 3. New reward stack — `conf/reward/team_v3_intercept.yaml`

Forked from `team_v2.yaml`. Two changes:

1. `HoopDistancePenalty` becomes Red-only: `agent_to_target: {red_0: hoop}`. (Old form had both Red and Blue mapped; the Blue entry — `blue_0: midpoint` — is removed.)
2. New `InterceptShaping` entry inserted after `HoopAnchor`, before `ScoreEvent`.

Full file:

```yaml
_target_: envs.quidditch.rewards.stack.RewardStack
terms:
  - _target_: envs.quidditch.rewards.terms.TagEntryPulse
    magnitude: 5.0
    gainer: blue_0
    loser:  red_0
  - _target_: envs.quidditch.rewards.terms.ProximityGradedTag
    max_reward: 0.05
    gainer: blue_0
    loser:  red_0
  - _target_: envs.quidditch.rewards.terms.ClosingVelInTagZone
    scale: 0.05
    gainer: blue_0
    loser:  red_0
  - _target_: envs.quidditch.rewards.terms.HoopDistancePenalty
    scale: 0.01
    agent_to_target:
      red_0: hoop
  - _target_: envs.quidditch.rewards.terms.ZeroSumDistMirror
    scale: 0.01
    agents: [blue_0]
  - _target_: envs.quidditch.rewards.terms.HoopAnchor
    scale: 0.005
    agents: [blue_0]
  - _target_: envs.quidditch.rewards.terms.InterceptShaping
    scale: 0.05
    lookahead_s: 0.5
    activation_dist: 1.5
    defender: blue_0
  - _target_: envs.quidditch.rewards.terms.ScoreEvent
    magnitude: 10.0
    scorer: red_0
    zero_sum_opponent: blue_0
  - _target_: envs.quidditch.rewards.terms.TakeDown
    aggressor_reward: 20.0
    victim_penalty: -20.0
    aggressor: blue_0
    victim: red_0
  - _target_: envs.quidditch.rewards.terms.CrashEvent
    magnitude: -20.0
    agent_to_crash_flags:
      red_0:  [red_floor, red_wall_crash,  red_oob]
      blue_0: [blue_floor, blue_wall_crash, blue_oob]
```

### 4. `team_env.py` refactor — per-agent obs builder

Today `QuidditchTeamEnv._build_agent_obs` always packs `DUEL_V1_BODY`. `OpponentControlledEnv._augment_learner_obs` then re-packs into `DUEL_V2_WORLD` for the learner. That two-stage pattern is awkward for body-frame learner obs because the augmenter would need the body rotation matrix, which it currently doesn't touch.

**New approach.** `QuidditchTeamEnv.__init__` takes two new kwargs:

```python
def __init__(
    self,
    *,
    cfg: TeamConfig | None = None,
    render_mode: str | None = None,
    reward_stack: RewardStack | None = None,
    learner_id: str | None = None,        # NEW: which agent OCE will use as the learner
    learner_spec: ObsSpec | None = None,  # NEW: spec for the learner; non-learner gets DUEL_V1_BODY
) -> None:
```

When `learner_id` is `None` (e.g. tests that don't wrap in OCE — the canary), both agents get `DUEL_V1_BODY` (current behavior preserved). When set, `_build_agent_obs(learner_id)` packs `learner_spec`; the other agent still gets `DUEL_V1_BODY`. The env's `observation_spaces[learner_id]` is updated to match `learner_spec.dim`.

`_build_agent_obs` dispatches on the agent id:

```python
def _build_agent_obs(self, agent_id: str) -> np.ndarray:
    spec = self._learner_spec if agent_id == self._learner_id else DUEL_V1_BODY
    return self._pack_agent_obs(agent_id, spec)

def _pack_agent_obs(self, agent_id: str, spec: ObsSpec) -> np.ndarray:
    # Computes all needed values for `spec`'s blocks. The set of blocks the
    # env knows how to fill is finite (DUEL_V1_BODY ∪ DUEL_V2_WORLD ∪ DUEL_V3_BODY_EGO).
    ...
```

The packer reads from `self_q.state()`, `opp_q.state()`, plus `data.xmat[learner_body_id]` and `data.qvel[*_dofadr]` when needed. The free-joint dofadr lookups are cached on first reset (same pattern as the existing OCE).

**Closing rate** (the only stateful obs feature today) moves from OCE to the team env, keyed on the learner. The env owns `_prev_dist_to_opp` (one per learner; non-learner doesn't have one). This eliminates the OCE-side stateful cache.

### 5. `OpponentControlledEnv` simplification

After Decision 4 *and the v2-augmentation migration described under Non-goals*, OCE no longer needs `_augment_learner_obs`, `_cache_dofadrs`, or `_prev_dist_to_opp` — the team env emits the learner's full obs (whatever shape `learner_spec` says — `DUEL_V1_BODY` / `DUEL_V2_WORLD` / `DUEL_V3_BODY_EGO`) and OCE is a pure pass-through. `OpponentControlledEnv.__init__` is updated:

```python
self.observation_space = team_env.observation_space(learner_id)  # whatever team_env decided
self.action_space      = team_env.action_space(learner_id)
```

The opponent still receives the team env's per-agent obs as before (always `DUEL_V1_BODY` for the non-learner, so frozen Red checkpoints keep loading without surgery).

`FrameStackWrapper` is unchanged.

**`blue_v4` regression check.** Moving the v2 packing into `team_env._pack_agent_obs(blue_0, DUEL_V2_WORLD)` must produce a byte-identical 25-d vector to what `OpponentControlledEnv._augment_learner_obs` produces today. Verified by:
- `tests/envs/quidditch/test_augmented_obs.py` is rewritten to construct `QuidditchTeamEnv(learner_id="blue_0", learner_spec=DUEL_V2_WORLD)` directly (no OCE) and assert the same five properties (25-d shape, `vec_to_hoop = HOOP - blue_pos`, closing-rate sign, world-frame `opp_vel_rel`, FrameStack doubling). Equivalence-by-property, not byte-for-byte against the old implementation — but the math is identical, so any deviation is a bug.
- A new fixture in `test_team_env_v3.py` round-trips `models/ppo_hoop_blue_4_20260511_202612/best_model` through one `predict()` call under the new code path. If the obs shape changes, SB3 raises; if the math drifts, the action distribution shifts. We just assert it runs without error and produces a 4-d action — anything beyond that crosses the line into "is `blue_v4` still good," which is a separate eval question.

### 6. `env_factories.py` wiring

`TeamEnvFactory` learns to resolve `cfg.obs.name` to an `ObsSpec` via `SPEC_BY_NAME[obs_spec_name]` and pass it as `learner_spec` to `QuidditchTeamEnv`:

```python
def _make_thunk(self):
    from envs.quidditch.team_env import QuidditchTeamEnv
    from envs.quidditch.opponents import OpponentControlledEnv, from_spec
    from envs.quidditch.obs_spec import SPEC_BY_NAME
    learner_spec = SPEC_BY_NAME[self.obs_spec_name]
    cfg = self.team_cfg
    learner = self.learner_id
    opp_spec = self.opponent_spec
    reward_stack = self.reward_stack
    def _thunk():
        team = QuidditchTeamEnv(
            cfg=cfg, reward_stack=reward_stack,
            learner_id=learner, learner_spec=learner_spec,
        )
        opp = from_spec(opp_spec)
        return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
    return _thunk
```

`obs_spec_name` already exists on `TeamEnvFactory`; today it's only used by callers reading it. Wiring it into the env construction is the one new behavior.

Video env builder (`build_video_env_fn`) gets the same `learner_id` + `learner_spec` plumbing.

### 7. `team_env.step` — intercept-shaping inputs

`team_env.step` populates the new `StepState` fields each step whenever a learner is configured:

```python
# In team_env.step(), after computing red_pos, blue_pos, dist_b2r:
if self._learner_id is not None:
    red_vel_world = self._world.data.qvel[
        self._red_dofadr : self._red_dofadr + 3
    ].copy()
    future_red   = red_pos + REWARD_LOOKAHEAD_S * red_vel_world
    defender_pos = blue_pos if self._learner_id == self._blue_id else red_pos
    d = float(np.linalg.norm(defender_pos - future_red))
    d_prev = self._dist_def_to_future_red_prev
    self._dist_def_to_future_red_prev = d
else:
    d = 0.0
    d_prev = 0.0
```

`REWARD_LOOKAHEAD_S = 0.5` lives in `envs/quidditch/constants.py` so the env and the YAML can stay in sync visually (and so a future `__post_init__` check on `InterceptShaping` can read it).

`_red_dofadr` / `_blue_dofadr` are cached once on first reset (cost: two `mj_name2id` lookups + two qvel address reads; reused across all subsequent steps). The future-red distance computation itself is two subtractions + one `np.linalg.norm` per step — negligible.

**Why no learner_spec gate.** The cost is small enough to not need gating, and unconditional computation removes a footgun: `InterceptShaping` instantiated in a stack with the wrong learner_spec would silently never fire if its inputs stayed at 0. Decoupling the StepState population from the term's activation rule keeps the term safe to add to any team stack — it gates itself on `dist_red_to_hoop < activation_dist`. The canary (no learner_id) still pays nothing because the `learner_id is None` branch keeps the values at 0 and `InterceptShaping` is not in the canary's `team_v2` stack regardless.

### 8. Experiment YAML — `conf/experiment/blue_v7.yaml`

```yaml
# @package _global_
# Pending: scratch Blue with body-frame ego obs + intercept reward shaping,
# trained against frozen red_v1, randomise_start=true.

run_name: ppo_hoop_blue_7

defaults:
  - override /env: team
  - override /obs: duel_v3_body_ego
  - override /reward: team_v3_intercept
  - override /opponent: frozen
  - override /init: scratch
  - override /curriculum: random_start

env:
  learner_id: blue_0
  team_env_params:
    midpoint_alpha: 0.6              # 60% Red / 40% hoop along the Red↔hoop segment

opponent:
  model_path: models/ppo_hoop_red_1_20260506_103058/best_model

trainer:
  lr: 3e-4
  total_timesteps: 10_000_000
```

Notes:
- `env.team_env_params.midpoint_alpha` overrides the value baked into `conf/env/team.yaml` (currently `0.3`).
- `opponent` is `frozen` with `red_v1` model path; the start point for Red is randomized via `curriculum: random_start` which flips `randomise_red_start=true` on `TeamConfig`.
- `init: scratch` per design discussion — the obs spec is structurally different from `blue_v4`, so warm-start would re-init the body-frame columns anyway; cleanest to start fresh.

### 9. Tests

| File | What it covers |
| --- | --- |
| `tests/envs/quidditch/test_obs_spec.py` | New: `DUEL_V3_BODY_EGO.dim == 25`; the four new constants are distinct from their world-frame counterparts; `SPEC_BY_NAME["DUEL_V3_BODY_EGO"]` resolves; the registry set is exactly `{"SIMPLE_ENV_OBS", "DUEL_V1_BODY", "DUEL_V2_WORLD", "DUEL_V3_BODY_EGO"}`. |
| `tests/envs/quidditch/test_obs_spec.py` (cont.) | New: `world_to_body` helper math — identity rotation passes vec through; 90° yaw rotates a world-x vector into body-(-y) (or +y, depending on sign convention — pinned by the test). |
| `tests/envs/quidditch/rewards/test_reward_stack.py` | New: `InterceptShaping` fires only when `dist_red_to_hoop < activation_dist`; reward is `scale × max(0, closing_rate)`; never penalises Red; zero when not closing. |
| `tests/envs/quidditch/rewards/test_reward_stack.py` (cont.) | New: `default_team_stack()` test gains a `team_v3_intercept` variant — `load_reward_stack("team_v3_intercept")` produces the expected term composition (10 terms, ordered, types correct). |
| `tests/envs/quidditch/test_team_env_v3.py` (new file) | `QuidditchTeamEnv(learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO)` emits a 25-d obs for Blue and a 22-d obs for Red; positions yawed 90° produce body-frame `opp_pos_rel` that differs from world-frame in the expected x↔y swap; `dist_def_to_future_red` is populated and updates step-over-step. |
| `tests/envs/quidditch/test_team_env_v3.py` (cont.) | Smoke: `OpponentControlledEnv(team_v3, learner_id="blue_0")` passes through 25-d obs without re-augmentation; FrameStackWrapper around it produces 75-d. |
| `tests/envs/quidditch/test_team_env_v3.py` (cont.) | `blue_v4` round-trip: load `models/ppo_hoop_blue_4_20260511_202612/best_model` into the new in-env packer path (`QuidditchTeamEnv(learner_id="blue_0", learner_spec=DUEL_V2_WORLD)` wrapped in FrameStackWrapper); assert one `predict()` runs cleanly and returns a (4,) action. Guards against silent obs-math drift when moving augmentation into team_env. |
| `tests/envs/quidditch/test_team_env_canary.py` | Untouched — the canary uses `QuidditchTeamEnv()` with default kwargs (no `learner_id`), so both agents continue to get `DUEL_V1_BODY` and the fingerprint `step 176 / step 684 / total {+1.542, -6.327}` must remain byte-identical. |
| `tests/envs/quidditch/test_augmented_obs.py` | Rewritten to assert the same five properties (25-d shape, `vec_to_hoop`, closing-rate sign, world-frame `opp_vel_rel`, FrameStack doubling) against `QuidditchTeamEnv(learner_id="blue_0", learner_spec=DUEL_V2_WORLD)` directly — no OCE in the path. |
| `tests/scripts/test_train_smoke_wandb_disabled.py` | Add a smoke that composes `blue_v7` (`hydra_compose(experiment="blue_v7", overrides=["trainer.total_timesteps=64"])`) and runs 64 env steps. |

Canary expectations:
- `tests/envs/quidditch/test_scoring_canary.py` (single-agent, `SCORED at step 434 / reward 7.3837`) — byte-identical.
- `tests/envs/quidditch/test_team_env_canary.py` — byte-identical.

If either canary moves a digit, this spec was wrong; investigate before merging.

### 10. Lineage / model registry impact

- `LEGACY_SPECS` in `scripts/migrate_legacy_models.py` — no change (it covers the 7 promoted models, none of which use `DUEL_V3_BODY_EGO`).
- `_check_obs_compat_from_hydra` — no change. `init=scratch` skips the obs-compat check.
- W&B artifact metadata — `blue_v7` will register as a new artifact collection under the run name `ppo_hoop_blue_7`. No conflicts with `blue_5`/`blue_6` (the queued runs from tasks.md never trained).

## Touch list

New files:
- `conf/obs/duel_v3_body_ego.yaml`
- `conf/reward/team_v3_intercept.yaml`
- `conf/experiment/blue_v7.yaml`
- `tests/envs/quidditch/test_team_env_v3.py`

Modified:
- `envs/quidditch/obs_spec.py` — 4 new `ObsBlock` constants, 1 new `ObsSpec`, `SPEC_BY_NAME` entry, `world_to_body` helper.
- `envs/quidditch/team_env.py` — `__init__` adds `learner_id` + `learner_spec` kwargs, `_build_agent_obs` dispatches, new `_pack_agent_obs(agent_id, spec)`, closing-rate cache moves in, `_red_dofadr`/`_blue_dofadr` cached on first reset, future-red distance computed each step and threaded into `StepState`.
- `envs/quidditch/opponents.py` — `OpponentControlledEnv` becomes a thin pass-through; removes `_augment_learner_obs`, `_cache_dofadrs`, `_prev_dist_to_opp`. `FrameStackWrapper` unchanged.
- `envs/quidditch/env_factories.py` — `TeamEnvFactory._make_thunk` (and `build_video_env_fn`) plumbs `learner_id` + `SPEC_BY_NAME[obs_spec_name]` into the team env construction.
- `envs/quidditch/constants.py` — `REWARD_LOOKAHEAD_S = 0.5` constant (mirrors `lookahead_s` in `team_v3_intercept.yaml`).
- `envs/quidditch/rewards/stack.py` — 2 new `StepState` fields.
- `envs/quidditch/rewards/terms.py` — `InterceptShaping` dataclass.
- `tests/envs/quidditch/test_obs_spec.py` — new-constant + helper assertions.
- `tests/envs/quidditch/rewards/test_reward_stack.py` — `InterceptShaping` term tests + `team_v3_intercept` stack composition test.
- `tests/scripts/test_train_smoke_wandb_disabled.py` — `blue_v7` smoke entry.

Unchanged but worth naming (asserting non-regression):
- `tests/envs/quidditch/test_team_env_canary.py` (fingerprint locked)
- `tests/envs/quidditch/test_scoring_canary.py` (fingerprint locked)
- `tests/envs/quidditch/test_augmented_obs.py` (DUEL_V2_WORLD path)
- `scripts/migrate_legacy_models.py`
- All `models/*/.hydra/` (none reference the new constants)

## Open questions for the user

None — all four major axes (obs layout, replace-vs-add reward semantics, intercept formulation, init mode) were resolved during brainstorming. Tuning knobs (`scale=0.05`, `lookahead_s=0.5`, `activation_dist=1.5`, `α=0.6`, `n_stack=3`, `lr=3e-4`, `total_timesteps=10M`) are committed as starting values; sweep later if needed.
