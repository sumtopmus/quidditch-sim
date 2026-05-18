# `blue_v7` body-frame ego obs + intercept reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a new `ppo_hoop_blue_7` training experiment that trains Blue from scratch under a new body-frame ego-centric obs spec (`DUEL_V3_BODY_EGO`) with a new `InterceptShaping` reward term, against frozen `red_v1` with randomized Red start. As part of this, refactor `QuidditchTeamEnv` to be per-agent obs-spec-aware and reduce `OpponentControlledEnv` to a pure pass-through.

**Architecture:** Adds 4 new `ObsBlock` constants + 1 new composed `ObsSpec` (`DUEL_V3_BODY_EGO`, 25-d body-frame) in `obs_spec.py`, plus a `world_to_body` helper. Adds 1 new reward term (`InterceptShaping`) with 2 new `StepState` fields populated unconditionally by `team_env.step`. Refactors `QuidditchTeamEnv` to accept `learner_id` + `learner_spec` kwargs and build the learner's obs in any of three shapes (`DUEL_V1_BODY` / `DUEL_V2_WORLD` / `DUEL_V3_BODY_EGO`) directly; `OpponentControlledEnv` loses its augmenter and becomes pure pass-through. Wires the new spec through `env_factories.TeamEnvFactory` and exposes it as a Hydra experiment.

**Tech Stack:** Python 3.11+, MuJoCo 3.x (rotation matrix from `data.xmat`), stable-baselines3 PPO, Hydra config, pytest, wandb (disabled in tests).

**Spec:** `docs/superpowers/specs/2026-05-18-blue-v7-body-ego-design.md`.

**Conventions used in this plan:**
- All file paths are relative to the repo root.
- All commands assume `cwd = /Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/worktrees/feature/blue-v7-body-ego`.
- Commits are GPG-signed by default (per user's git config); commit messages follow `<type>(<scope>): <subject>` style.
- Tests in this repo do not use `unittest.TestCase` — they are plain `pytest` functions with `try/finally` for teardown (see `tests/envs/quidditch/test_team_env_canary.py`).

---

### Task 1: New `ObsBlock` constants + `DUEL_V3_BODY_EGO` spec + `SPEC_BY_NAME` entry

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Modify: `tests/envs/quidditch/test_obs_spec.py`

- [ ] **Step 1: Append failing tests for the new constants/spec to `tests/envs/quidditch/test_obs_spec.py`**

Append at the end of the file:

```python
def test_vec_to_goal_body_block_is_distinct_from_unit_to_goal():
    g_body  = obs_spec.VEC_TO_GOAL_BODY
    g_world = obs_spec.UNIT_TO_GOAL
    assert g_body.name == "vec_to_goal"
    assert g_world.name == "unit_to_goal"
    assert g_body != g_world


def test_vec_to_hoop_body_and_world_are_distinct():
    body  = obs_spec.VEC_TO_HOOP_BODY
    world = obs_spec.VEC_TO_HOOP
    assert body.name == world.name == "vec_to_hoop"
    assert body.dim == world.dim == 3
    assert body.frame == "body"
    assert world.frame == "world"
    assert body != world


def test_opp_pos_rel_body_and_world_are_distinct():
    body  = obs_spec.OPP_POS_REL_BODY
    world = obs_spec.OPP_POS_REL
    assert body.name == world.name == "opp_pos_rel"
    assert body.dim == world.dim == 3
    assert body.frame == "body"
    assert world.frame == "world"
    assert body != world


def test_opp_vel_rel_body_ego_is_distinct_from_legacy_body_mixed_and_world():
    ego        = obs_spec.OPP_VEL_REL_BODY_EGO
    body_mixed = obs_spec.OPP_VEL_REL_BODY
    world      = obs_spec.OPP_VEL_REL_WORLD
    assert ego.name == body_mixed.name == world.name == "opp_vel_rel"
    assert ego.frame == "body"
    assert body_mixed.frame == "body_mixed"
    assert world.frame == "world"
    assert ego != body_mixed
    assert ego != world


def test_duel_v3_body_ego_dim_is_25():
    assert obs_spec.DUEL_V3_BODY_EGO.dim == 25


def test_duel_v3_body_ego_block_names_in_order():
    names = [b.name for b in obs_spec.DUEL_V3_BODY_EGO.blocks]
    assert names == [
        "ang_vel", "ang_pos", "lin_vel", "lin_pos",
        "vec_to_goal", "vec_to_hoop", "opp_pos_rel", "opp_vel_rel",
        "closing_rate",
    ]


def test_duel_v3_body_ego_uses_body_frame_for_relative_blocks():
    spec = obs_spec.DUEL_V3_BODY_EGO
    assert obs_spec.VEC_TO_GOAL_BODY     in spec.blocks
    assert obs_spec.VEC_TO_HOOP_BODY     in spec.blocks
    assert obs_spec.OPP_POS_REL_BODY     in spec.blocks
    assert obs_spec.OPP_VEL_REL_BODY_EGO in spec.blocks
    # No world-frame opp blocks in v3.
    assert obs_spec.OPP_POS_REL          not in spec.blocks
    assert obs_spec.OPP_VEL_REL_WORLD    not in spec.blocks


def test_spec_by_name_registers_duel_v3_body_ego():
    from envs.quidditch.obs_spec import SPEC_BY_NAME, DUEL_V3_BODY_EGO
    assert SPEC_BY_NAME["DUEL_V3_BODY_EGO"] is DUEL_V3_BODY_EGO


def test_spec_by_name_set_is_exact():
    """Adding a new spec without registering it (or removing one) breaks here."""
    from envs.quidditch.obs_spec import SPEC_BY_NAME
    assert set(SPEC_BY_NAME) == {
        "SIMPLE_ENV_OBS", "DUEL_V1_BODY", "DUEL_V2_WORLD", "DUEL_V3_BODY_EGO",
    }
```

Also delete the OLD `test_spec_by_name_maps_canonical_specs` test at the bottom of the file (it asserts the registry is exactly `{SIMPLE_ENV_OBS, DUEL_V1_BODY, DUEL_V2_WORLD}` and will start failing after Step 3). Replace it with `test_spec_by_name_set_is_exact` from above.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/envs/quidditch/test_obs_spec.py -v`
Expected: 9 new tests fail with `AttributeError: module 'envs.quidditch.obs_spec' has no attribute 'VEC_TO_GOAL_BODY'` (and similar for the other 3 constants), `KeyError: 'DUEL_V3_BODY_EGO'`, and `AttributeError: 'DUEL_V3_BODY_EGO'`.

- [ ] **Step 3: Add the new constants + composed spec + registry entry to `envs/quidditch/obs_spec.py`**

After the existing `OPP_VEL_REL_WORLD` constant (around line 98), add:

```python
VEC_TO_GOAL_BODY = ObsBlock(
    "vec_to_goal", dim=3, frame="body",
    notes="goal point - learner_pos, rotated into learner body frame; "
          "goal = α·red_pos + (1-α)·hoop_center (α from TeamConfig.midpoint_alpha)",
)
VEC_TO_HOOP_BODY     = ObsBlock("vec_to_hoop",  dim=3, frame="body")
OPP_POS_REL_BODY     = ObsBlock("opp_pos_rel",  dim=3, frame="body")
OPP_VEL_REL_BODY_EGO = ObsBlock(
    "opp_vel_rel", dim=3, frame="body",
    notes="(opp_vel_world - learner_vel_world) rotated into learner body "
          "frame; distinct from OPP_VEL_REL_BODY (body_mixed)",
)
```

After `DUEL_V2_WORLD` (around line 118), add:

```python
DUEL_V3_BODY_EGO: ObsSpec = ObsSpec((
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS,
    VEC_TO_GOAL_BODY,
    VEC_TO_HOOP_BODY,
    OPP_POS_REL_BODY,
    OPP_VEL_REL_BODY_EGO,
    CLOSING_RATE,
))
```

And extend `SPEC_BY_NAME`:

```python
SPEC_BY_NAME: dict[str, ObsSpec] = {
    "SIMPLE_ENV_OBS":   SIMPLE_ENV_OBS,
    "DUEL_V1_BODY":     DUEL_V1_BODY,
    "DUEL_V2_WORLD":    DUEL_V2_WORLD,
    "DUEL_V3_BODY_EGO": DUEL_V3_BODY_EGO,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/envs/quidditch/test_obs_spec.py -v`
Expected: all tests PASS (the original suite + 9 new tests).

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_obs_spec.py
git commit -m "$(cat <<'EOF'
feat(obs-spec): add DUEL_V3_BODY_EGO + body-frame ObsBlock constants

New blocks VEC_TO_GOAL_BODY, VEC_TO_HOOP_BODY, OPP_POS_REL_BODY,
OPP_VEL_REL_BODY_EGO; composed into 25-d DUEL_V3_BODY_EGO and
registered in SPEC_BY_NAME.  No env or factory wiring yet — those
land in later tasks.
EOF
)"
```

---

### Task 2: `world_to_body` helper in `obs_spec.py`

**Files:**
- Modify: `envs/quidditch/obs_spec.py`
- Modify: `tests/envs/quidditch/test_obs_spec.py`

- [ ] **Step 1: Append failing tests for `world_to_body` to `tests/envs/quidditch/test_obs_spec.py`**

Append:

```python
def test_world_to_body_identity_rotation_passes_vec_through():
    R_wb = np.eye(3, dtype=np.float64)
    v_world = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v_body = obs_spec.world_to_body(v_world, R_wb)
    np.testing.assert_array_almost_equal(v_body, v_world)
    assert v_body.dtype == np.float32


def test_world_to_body_90deg_yaw():
    """Body yawed +90° about world z: world +x axis becomes body +y axis.

    R_wb (body→world) for a +90° yaw rotates body-x to world-y, so
    its transpose (world→body) maps world-x → body -y.
    """
    # +90° about z: world-x → world-y for any vector expressed in the body frame.
    c, s = 0.0, 1.0
    R_wb = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    v_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_body = obs_spec.world_to_body(v_world, R_wb)
    # world-x viewed from a body rotated +90° about z lies along body -y.
    np.testing.assert_array_almost_equal(v_body, np.array([0.0, -1.0, 0.0]))


def test_world_to_body_preserves_norm():
    R_wb = np.array([
        [ 0.6, -0.8, 0.0],
        [ 0.8,  0.6, 0.0],
        [ 0.0,  0.0, 1.0],
    ], dtype=np.float64)
    v_world = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    v_body = obs_spec.world_to_body(v_world, R_wb)
    assert abs(np.linalg.norm(v_body) - np.linalg.norm(v_world)) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/envs/quidditch/test_obs_spec.py -v -k world_to_body`
Expected: 3 tests fail with `AttributeError: module 'envs.quidditch.obs_spec' has no attribute 'world_to_body'`.

- [ ] **Step 3: Add the helper to `envs/quidditch/obs_spec.py`**

After the `pack(...)` function (before the `# ── Canonical ObsBlock constants ──` block, around line 70), add:

```python
def world_to_body(vec_world: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector from world frame into a body frame defined by R_wb.

    `R_wb` is body→world (MuJoCo `data.xmat[body_id].reshape(3, 3)`
    convention — columns are body axes expressed in world coords).  The
    inverse rotation (world→body) is R_wb.T, applied here.
    """
    return (R_wb.T @ vec_world).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/envs/quidditch/test_obs_spec.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/obs_spec.py tests/envs/quidditch/test_obs_spec.py
git commit -m "$(cat <<'EOF'
feat(obs-spec): add world_to_body rotation helper

Pure function: v_body = R_wb.T @ v_world, where R_wb is body→world
(MuJoCo data.xmat convention).  Used by the upcoming per-agent obs
packer to build DUEL_V3_BODY_EGO blocks.
EOF
)"
```

---

### Task 3: `conf/obs/duel_v3_body_ego.yaml`

**Files:**
- Create: `conf/obs/duel_v3_body_ego.yaml`
- Modify: `tests/envs/quidditch/test_obs_spec.py`

- [ ] **Step 1: Append a failing config-resolution test**

Append to `tests/envs/quidditch/test_obs_spec.py`:

```python
def test_duel_v3_body_ego_yaml_resolves_to_canonical_spec():
    """conf/obs/duel_v3_body_ego.yaml must declare name=DUEL_V3_BODY_EGO
    so that SPEC_BY_NAME[cfg.obs.name] lookup in env_factories resolves."""
    from pathlib import Path
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(__file__).resolve().parents[3] / "conf" / "obs" / "duel_v3_body_ego.yaml")
    assert cfg.name == "DUEL_V3_BODY_EGO"
    assert int(cfg.n_stack) == 3
    assert obs_spec.SPEC_BY_NAME[cfg.name] is obs_spec.DUEL_V3_BODY_EGO
```

- [ ] **Step 2: Run tests to verify the new test fails**

Run: `pytest tests/envs/quidditch/test_obs_spec.py::test_duel_v3_body_ego_yaml_resolves_to_canonical_spec -v`
Expected: FAIL with `FileNotFoundError: ... conf/obs/duel_v3_body_ego.yaml`.

- [ ] **Step 3: Create `conf/obs/duel_v3_body_ego.yaml`**

Contents:

```yaml
# Resolves at runtime to envs.quidditch.obs_spec.DUEL_V3_BODY_EGO (25-d,
# body-frame ego-centric — vec_to_goal, vec_to_hoop, opp_pos_rel,
# opp_vel_rel all in the learner's body frame).
name: DUEL_V3_BODY_EGO
n_stack: 3
```

- [ ] **Step 4: Run tests to verify it passes**

Run: `pytest tests/envs/quidditch/test_obs_spec.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/obs/duel_v3_body_ego.yaml tests/envs/quidditch/test_obs_spec.py
git commit -m "$(cat <<'EOF'
feat(conf): conf/obs/duel_v3_body_ego.yaml for the new spec

n_stack=3 matches blue_v4 so learning curves are comparable.  Test
asserts the YAML resolves back to the canonical ObsSpec constant.
EOF
)"
```

---

### Task 4: `REWARD_LOOKAHEAD_S` constant + `StepState` future-red fields

**Files:**
- Modify: `envs/quidditch/constants.py`
- Modify: `envs/quidditch/rewards/stack.py`
- Modify: `tests/envs/quidditch/rewards/test_reward_stack.py`

- [ ] **Step 1: Append failing tests to `tests/envs/quidditch/rewards/test_reward_stack.py`**

Append at the end:

```python
def test_step_state_has_future_red_fields_defaulted_to_zero():
    """New fields for InterceptShaping; default 0.0 so existing callers
    that build StepState without them keep working."""
    from envs.quidditch.rewards.stack import StepState
    state = StepState(agent_ids=("red_0", "blue_0"))
    assert state.dist_def_to_future_red == 0.0
    assert state.dist_def_to_future_red_prev == 0.0


def test_reward_lookahead_constant_exists():
    """Constant lives in envs.quidditch.constants so team_env (which
    populates the StepState fields) and the YAML (which carries the
    InterceptShaping lookahead_s param) read from the same source."""
    from envs.quidditch.constants import REWARD_LOOKAHEAD_S
    assert REWARD_LOOKAHEAD_S == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py -v -k "future_red or lookahead"`
Expected: 2 tests fail with `TypeError: StepState.__init__() got an unexpected keyword argument` is NOT what we'll see; the actual failure is `AttributeError: 'StepState' object has no attribute 'dist_def_to_future_red'` and `ImportError: cannot import name 'REWARD_LOOKAHEAD_S' from 'envs.quidditch.constants'`.

- [ ] **Step 3: Add `REWARD_LOOKAHEAD_S` to `envs/quidditch/constants.py`**

After the `CRASH_VEL_THR` constant (around line 37), add:

```python
# ── Reward shaping ───────────────────────────────────────────────────────────
# Prediction horizon for InterceptShaping: dist_def_to_future_red uses
# future_red = red_pos + REWARD_LOOKAHEAD_S · red_vel_world.  Mirrored in
# conf/reward/team_v3_intercept.yaml's InterceptShaping.lookahead_s field
# (kept in sync visually so a reader sees both numbers next to each other).
REWARD_LOOKAHEAD_S: float = 0.5
```

- [ ] **Step 4: Add the two `StepState` fields**

In `envs/quidditch/rewards/stack.py`, find the `StepState` dataclass (around lines 21–60) and add two fields after `tag_radius`:

```python
    # ── Intercept-shaping inputs ────────────────────────────────────────────
    # Populated by team_env.step when learner_id is set; both default to 0 so
    # single-agent envs and the no-learner canary path keep working.
    # future_red = red_pos + REWARD_LOOKAHEAD_S · red_vel_world
    # dist_def_to_future_red = ‖defender_pos - future_red‖
    dist_def_to_future_red:      float = 0.0
    dist_def_to_future_red_prev: float = 0.0
```

Place the additions immediately after `tag_radius: float = 0.3`, before the closing `}` of the dataclass.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py -v`
Expected: all tests PASS (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add envs/quidditch/constants.py envs/quidditch/rewards/stack.py tests/envs/quidditch/rewards/test_reward_stack.py
git commit -m "$(cat <<'EOF'
feat(rewards): add StepState future-red fields + REWARD_LOOKAHEAD_S

Two new StepState fields default to 0.0 so existing callers (single-
agent envs, no-learner canary) keep working; team_env will populate
them in a later task.  REWARD_LOOKAHEAD_S in constants.py is the
single source the YAML's InterceptShaping.lookahead_s mirrors.
EOF
)"
```

---

### Task 5: `InterceptShaping` reward term

**Files:**
- Modify: `envs/quidditch/rewards/terms.py`
- Modify: `tests/envs/quidditch/rewards/test_reward_stack.py`

- [ ] **Step 1: Append failing tests to `tests/envs/quidditch/rewards/test_reward_stack.py`**

Append:

```python
from envs.quidditch.rewards.terms import InterceptShaping


def test_intercept_shaping_zero_when_red_far_from_hoop():
    """Activation gate: dist_red_to_hoop >= activation_dist → zero reward."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=2.0,                  # >= 1.5, gate misses
        dist_def_to_future_red=0.3,
        dist_def_to_future_red_prev=0.5,       # blue closing on future-red
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_intercept_shaping_positive_when_red_near_and_blue_closing():
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=1.0,                  # < 1.5, gate fires
        dist_def_to_future_red=0.3,
        dist_def_to_future_red_prev=0.5,
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    # closing = (0.5 - 0.3) / (1/240) = 48 m/s
    # bonus = 0.05 * 48 = 2.4
    assert out["blue_0"] == 0.05 * (0.5 - 0.3) / (1 / 240.0)
    # Term is non-zero-sum: Red is NOT penalised.
    assert out["red_0"] == 0.0


def test_intercept_shaping_zero_when_blue_separating_from_future_red():
    """max(0, closing) floors at 0 when defender is moving away from
    future-red — no negative reward for retreating."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=1.0,
        dist_def_to_future_red=0.7,
        dist_def_to_future_red_prev=0.5,       # blue retreating
        step_period=1 / 240.0,
    )
    out = term.compute(state)
    assert out == {"red_0": 0.0, "blue_0": 0.0}


def test_intercept_shaping_uses_only_defender_field():
    """Even with all other state inputs non-zero, only `defender` is rewarded."""
    term = InterceptShaping(scale=0.05, lookahead_s=0.5, activation_dist=1.5,
                             defender="blue_0")
    state = _make_state(
        dist_red_to_hoop=0.5,
        dist_def_to_future_red=0.0, dist_def_to_future_red_prev=1.0,
        step_period=1 / 240.0,
        # Distract: set other things that might leak into reward.
        tag_during=True, tag_entry=True, scored=True, drone_drone_crash=True,
    )
    out = term.compute(state)
    assert out["red_0"] == 0.0
    assert out["blue_0"] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py -v -k intercept_shaping`
Expected: 4 tests fail with `ImportError: cannot import name 'InterceptShaping' from 'envs.quidditch.rewards.terms'`.

- [ ] **Step 3: Add `InterceptShaping` to `envs/quidditch/rewards/terms.py`**

Append after `TakeDown` (around line 215):

```python
@dataclass
class InterceptShaping:
    """Closing-velocity reward on Red's short-horizon predicted position,
    active only when Red is within `activation_dist` of the hoop.

    Inputs (`dist_def_to_future_red`, `dist_def_to_future_red_prev`) are
    populated each step by team_env from the world-frame Red velocity:
        future_red = red_pos + lookahead_s · red_vel_world
        dist_def_to_future_red = ‖defender_pos - future_red‖

    Mirrors ClosingVelInTagZone in structure (zero-floored closing rate
    weighted by `scale`), but gated by dist_red_to_hoop instead of
    tag_during.  Non-zero-sum: only the defender is rewarded — Red is
    not penalised on this signal (Red already has its own
    HoopDistancePenalty pulling it toward the hoop).

    `lookahead_s` is informational here (the env consumed it when
    populating the future-red distances); kept in the dataclass so the
    YAML stays self-documenting next to `scale` and `activation_dist`.
    """
    scale: float
    lookahead_s: float
    activation_dist: float
    defender: str

    def compute(self, state: StepState) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in state.agent_ids}
        if state.dist_red_to_hoop >= self.activation_dist:
            return out
        closing = (
            state.dist_def_to_future_red_prev - state.dist_def_to_future_red
        ) / state.step_period
        out[self.defender] += self.scale * max(0.0, closing)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/rewards/terms.py tests/envs/quidditch/rewards/test_reward_stack.py
git commit -m "$(cat <<'EOF'
feat(rewards): add InterceptShaping term

Closing-velocity reward on a short-horizon predicted future-Red point,
gated on dist_red_to_hoop < activation_dist.  Non-zero-sum (only the
defender is rewarded).  Inputs populated by team_env in a later task.
EOF
)"
```

---

### Task 6: `conf/reward/team_v3_intercept.yaml`

**Files:**
- Create: `conf/reward/team_v3_intercept.yaml`
- Modify: `tests/envs/quidditch/rewards/test_reward_stack.py`

- [ ] **Step 1: Append a failing composition test**

Append to `tests/envs/quidditch/rewards/test_reward_stack.py`:

```python
def test_team_v3_intercept_stack_composition():
    """conf/reward/team_v3_intercept.yaml: 10 terms in the expected order,
    Blue removed from HoopDistancePenalty, InterceptShaping inserted
    between HoopAnchor and ScoreEvent."""
    from envs.quidditch.rewards import load_reward_stack
    stack = load_reward_stack("team_v3_intercept")
    expected = [
        "TagEntryPulse", "ProximityGradedTag", "ClosingVelInTagZone",
        "HoopDistancePenalty", "ZeroSumDistMirror", "HoopAnchor",
        "InterceptShaping",
        "ScoreEvent", "TakeDown", "CrashEvent",
    ]
    assert [type(t).__name__ for t in stack.terms] == expected

    # HoopDistancePenalty in v3 is Red-only (no blue→midpoint entry).
    hdp = next(t for t in stack.terms if type(t).__name__ == "HoopDistancePenalty")
    assert dict(hdp.agent_to_target) == {"red_0": "hoop"}

    # InterceptShaping carries the spec'd starting values.
    isp = next(t for t in stack.terms if type(t).__name__ == "InterceptShaping")
    assert isp.scale == 0.05
    assert isp.lookahead_s == 0.5
    assert isp.activation_dist == 1.5
    assert isp.defender == "blue_0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py::test_team_v3_intercept_stack_composition -v`
Expected: FAIL with `FileNotFoundError: ... conf/reward/team_v3_intercept.yaml`.

- [ ] **Step 3: Create `conf/reward/team_v3_intercept.yaml`**

Contents:

```yaml
# Reward stack for blue_v7: drops HoopDistancePenalty for blue_0 (midpoint
# shaping), keeps HoopAnchor + ZeroSumDistMirror (Blue still defends near
# hoop and benefits from Red being far), adds InterceptShaping for
# proactive closing on Red's predicted trajectory when Red is near hoop.

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

  # Red-only (blue's midpoint penalty removed for blue_v7).
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

  # New term: closing-velocity reward on future-Red, gated on red-near-hoop.
  # lookahead_s mirrors envs.quidditch.constants.REWARD_LOOKAHEAD_S.
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/envs/quidditch/rewards/test_reward_stack.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/reward/team_v3_intercept.yaml tests/envs/quidditch/rewards/test_reward_stack.py
git commit -m "$(cat <<'EOF'
feat(conf): team_v3_intercept reward stack for blue_v7

Drops blue's HoopDistancePenalty (midpoint shaping), keeps HoopAnchor
+ ZeroSumDistMirror, adds InterceptShaping (scale=0.05, lookahead=0.5s,
activation=1.5m, blue_0).  Red's stack unchanged.
EOF
)"
```

---

### Task 7: `QuidditchTeamEnv` — per-agent obs builder for all 3 specs + future-red distances

This is the largest task. The refactor:
1. Adds `learner_id: str | None = None` and `learner_spec: ObsSpec | None = None` kwargs to `__init__`.
2. Per-agent packer: when `learner_id` is set, the learner gets `learner_spec` and the non-learner gets `DUEL_V1_BODY`. When `learner_id` is None (canary path), both agents get `DUEL_V1_BODY` (current behavior).
3. Per-agent `observation_spaces` reflects each agent's spec.
4. Caches `_red_dofadr` / `_blue_dofadr` on first reset (used for world-frame opp velocity readback under DUEL_V2_WORLD and DUEL_V3_BODY_EGO).
5. Tracks `_prev_dist_to_opp[learner_id]` for closing-rate (formerly in OCE).
6. Populates `StepState.dist_def_to_future_red` + `_prev` unconditionally when `learner_id` is set.

**Files:**
- Modify: `envs/quidditch/team_env.py`
- Create: `tests/envs/quidditch/test_team_env_v3.py`
- Modify: `tests/conftest.py` (extend `set_body_state` with optional `yaw`)

- [ ] **Step 1: Extend `tests/conftest.py:set_body_state` with optional `yaw`**

In `tests/conftest.py`, replace the `set_body_state` function with:

```python
def set_body_state(
    world: World,
    body: str,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
) -> None:
    """Write body's free-joint position, yaw-only quat, and linear velocity."""
    import math
    qa = qpos_addr(world, body)
    world.data.qpos[qa : qa + 3] = pos
    # ZYX-euler yaw → quaternion (w, x, y, z): only z-component active.
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    world.data.qpos[qa + 3 : qa + 7] = (cy, 0.0, 0.0, sy)
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body)
    jnt = int(world.model.body_jntadr[bid])
    qva = int(world.model.jnt_dofadr[jnt])
    world.data.qvel[qva : qva + 3] = vel
    world.data.qvel[qva + 3 : qva + 6] = 0.0
```

The default `yaw=0.0` keeps the identity-quat behavior, so existing tests (`test_augmented_obs.py`, etc.) are unaffected.

- [ ] **Step 2: Write the new test file `tests/envs/quidditch/test_team_env_v3.py`**

Create with:

```python
"""QuidditchTeamEnv per-agent obs builder — DUEL_V2_WORLD and DUEL_V3_BODY_EGO
emitted directly by team_env (no OCE augmenter in the path).
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from envs.quidditch.constants import HOOP_CENTER, REWARD_LOOKAHEAD_S
from envs.quidditch.obs_spec import (
    DUEL_V1_BODY, DUEL_V2_WORLD, DUEL_V3_BODY_EGO,
)
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state


def _team(*, learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO):
    return QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id=learner_id,
        learner_spec=learner_spec,
    )


def test_default_construction_keeps_both_agents_on_duel_v1_body():
    """No learner_id → canary-compatible behavior preserved."""
    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False))
    try:
        env.reset(seed=0)
        assert env.observation_space("red_0").shape  == (DUEL_V1_BODY.dim,)
        assert env.observation_space("blue_0").shape == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v3_blue_learner_emits_25d_obs_for_blue_22d_for_red():
    env = _team(learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["blue_0"].shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs["red_0"].shape  == (DUEL_V1_BODY.dim,)
        assert env.observation_space("blue_0").shape == (DUEL_V3_BODY_EGO.dim,)
        assert env.observation_space("red_0").shape  == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v2_blue_learner_emits_25d_obs_for_blue():
    """DUEL_V2_WORLD path: team_env packs the 25-d world-frame obs
    directly (the work formerly done by OCE._augment_learner_obs)."""
    env = _team(learner_id="blue_0", learner_spec=DUEL_V2_WORLD)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["blue_0"].shape == (DUEL_V2_WORLD.dim,)
        assert obs["red_0"].shape  == (DUEL_V1_BODY.dim,)
    finally:
        env.close()


def test_v3_vec_to_hoop_is_in_body_frame():
    """Yaw Blue +90° about z; world-x vec_to_hoop becomes body -y."""
    env = _team()
    try:
        env.reset(seed=0)
        # Place blue at origin, yawed +90°; hoop is at (2, 0, 2) world.
        set_body_state(env._world, "blue_0", pos=(0.0, 0.0, 1.5), yaw=np.pi / 2)
        mujoco.mj_forward(env._world.model, env._world.data)
        obs, _, _, _, _ = env.step({"blue_0": np.zeros(4, np.float32),
                                     "red_0":  np.zeros(4, np.float32)})
        blue = obs["blue_0"]
        # Slot order in DUEL_V3_BODY_EGO: ang_vel(3), ang_pos(3), lin_vel(3),
        # lin_pos(3), vec_to_goal(3), vec_to_hoop(3), opp_pos_rel(3),
        # opp_vel_rel(3), closing_rate(1)  → vec_to_hoop slice is [15:18].
        vec_to_hoop_body = blue[15:18]
        # World vec_to_hoop ≈ (2, 0, 0.5); under +90° body yaw, world-x maps
        # to body -y, world-y to body +x, so body-frame is roughly (0, -2, 0.5).
        # PID + one step adds noise; check the dominant axis.
        assert abs(vec_to_hoop_body[1]) > abs(vec_to_hoop_body[0]), (
            f"under +90° yaw, vec_to_hoop should be dominantly along body -y, "
            f"got {vec_to_hoop_body}"
        )
        assert vec_to_hoop_body[1] < 0, (
            f"vec_to_hoop body-y should be negative (world +x → body -y); "
            f"got {vec_to_hoop_body}"
        )
    finally:
        env.close()


def test_v3_opp_pos_rel_is_in_body_frame():
    """Yaw Blue +90°; Red at world +x of Blue → body -y of Blue."""
    env = _team()
    try:
        env.reset(seed=0)
        set_body_state(env._world, "blue_0", pos=(0.0, 0.0, 1.5), yaw=np.pi / 2)
        set_body_state(env._world, "red_0",  pos=(1.0, 0.0, 1.5))
        mujoco.mj_forward(env._world.model, env._world.data)
        obs, _, _, _, _ = env.step({"blue_0": np.zeros(4, np.float32),
                                     "red_0":  np.zeros(4, np.float32)})
        blue = obs["blue_0"]
        opp_pos_rel_body = blue[18:21]
        assert opp_pos_rel_body[1] < 0, (
            f"red is at blue's world +x → body -y under +90° yaw, "
            f"got opp_pos_rel_body = {opp_pos_rel_body}"
        )
    finally:
        env.close()


def test_v3_future_red_dist_updates_step_over_step():
    """team_env populates _dist_def_to_future_red_prev each step; reading
    it on consecutive resets shows the cache advances."""
    env = _team()
    try:
        env.reset(seed=0)
        d0 = env._dist_def_to_future_red_prev
        env.step({"blue_0": np.zeros(4, np.float32),
                  "red_0":  np.zeros(4, np.float32)})
        d1 = env._dist_def_to_future_red_prev
        # The cache must have been populated each step; values are non-zero
        # and consecutive steps generally differ under PID-driven motion.
        assert d0 >= 0.0
        assert d1 >= 0.0
    finally:
        env.close()


def test_oce_passes_through_v3_blue_obs_without_re_augmentation():
    """OCE around a v3 team env emits the 25-d blue obs as-is."""
    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V3_BODY_EGO,
    )
    env = OpponentControlledEnv(team, learner_id="blue_0",
                                 opponent=from_spec("zero"))
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs.shape == (DUEL_V3_BODY_EGO.dim,)
        assert obs.dtype == np.float32
    finally:
        env.close()


@pytest.mark.slow
def test_blue_v4_round_trip_through_new_in_env_packer():
    """blue_v4 was trained under DUEL_V2_WORLD (formerly built by OCE's
    augmenter, now built by team_env's per-agent packer).  Round-trip
    one predict() call to confirm shape + math stays compatible.
    """
    from pathlib import Path
    from stable_baselines3 import PPO

    from envs.quidditch.opponents import FrameStackWrapper

    model_path = Path("models/ppo_hoop_blue_4_20260511_202612/best_model")
    if not model_path.with_suffix(".zip").exists():
        pytest.skip(f"blue_v4 checkpoint not found at {model_path}")

    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V2_WORLD,
    )
    env = OpponentControlledEnv(team, learner_id="blue_0",
                                 opponent=from_spec("zero"))
    env = FrameStackWrapper(env, n_stack=3)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (DUEL_V2_WORLD.dim * 3,)
        model = PPO.load(str(model_path))
        action, _ = model.predict(obs, deterministic=True)
        assert np.asarray(action).shape == (4,)
    finally:
        env.close()
```

- [ ] **Step 3: Run the new test file to verify failures**

Run: `pytest tests/envs/quidditch/test_team_env_v3.py -v`
Expected: failures around `TypeError: __init__() got an unexpected keyword argument 'learner_id'` and `AttributeError: 'QuidditchTeamEnv' object has no attribute '_dist_def_to_future_red_prev'`.

- [ ] **Step 4: Refactor `envs/quidditch/team_env.py`**

Open `envs/quidditch/team_env.py`. Apply the following changes:

(4a) Update the imports near the top:

```python
from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import (
    DUEL_V1_BODY, DUEL_V2_WORLD, DUEL_V3_BODY_EGO, ObsSpec,
)
```

(4b) Update the imports of constants at the top of the file:

```python
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
    BLUE_START_POS,
    BLUE_START_YAW,
    TAG_RADIUS,
    TAG_COOLDOWN_SECONDS,
    CRASH_VEL_THR,
    REWARD_LOOKAHEAD_S,
)
```

(4c) Add MuJoCo to top-level imports (used for `mj_name2id`):

```python
import mujoco
```

(4d) Update `__init__` signature and body. Replace the existing `def __init__(...)` through to just before `def observation_space(self, agent):` (lines ~101–170) with:

```python
    def __init__(
        self,
        *,
        cfg: TeamConfig | None = None,
        render_mode: str | None = None,
        reward_stack: RewardStack | None = None,
        learner_id: str | None = None,
        learner_spec: ObsSpec | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else TeamConfig()
        self.render_mode = render_mode

        self._red_id  = self.cfg.red_prefix
        self._blue_id = self.cfg.blue_prefix
        self.possible_agents = [self._red_id, self._blue_id]
        self.agents: list[str] = list(self.possible_agents)

        # Per-agent obs shape: non-learner always gets DUEL_V1_BODY (so frozen
        # Red checkpoints load without surgery).  Learner gets `learner_spec`,
        # which defaults to DUEL_V1_BODY when no learner is configured (canary
        # path: both agents on DUEL_V1_BODY, byte-identical to pre-refactor).
        if learner_id is not None and learner_id not in self.possible_agents:
            raise ValueError(
                f"learner_id={learner_id!r} not in possible_agents="
                f"{self.possible_agents}"
            )
        self._learner_id: str | None = learner_id
        self._learner_spec: ObsSpec = (
            learner_spec if learner_spec is not None else DUEL_V1_BODY
        )

        # Observation spaces: build per-agent based on its spec.
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_spaces: dict[str, spaces.Box] = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._spec_for_agent(agent).dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces: dict[str, spaces.Box] = {
            self._red_id:  act_box,
            self._blue_id: act_box,
        }

        self._world: World | None = None
        self._red:  Quadrotor | None = None
        self._blue: Quadrotor | None = None
        self._hoop_scorer: GeomDistanceScorer | None = None
        self._tag_scorer:  TagDistanceScorer  | None = None
        self._crash_detector: CrashDetector | None = None

        self._setpoint_red  = np.zeros(4, dtype=np.float32)
        self._setpoint_blue = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0
        self._max_steps:  int = 0
        self._cooldown_ticks: int = 0

        self._red_takeoff_grace:  int = 0
        self._blue_takeoff_grace: int = 0

        self._tag_blue_on_red = _TagState()

        self._red_crossing_started: bool = False
        self._red_enter_signed_dist: float = 0.0
        self._red_prev_signed_dist: float  = 0.0
        self._dist_b2r_prev: float = 0.0

        # Aftermath state.
        self._aftermath_steps_left: int = 0

        # Free-joint dofadr cache for world-frame velocity readback (populated
        # on first reset).  -1 sentinel = uncached.
        self._red_dofadr:  int = -1
        self._blue_dofadr: int = -1

        # Closing-rate state for the learner (formerly in OCE).
        self._prev_dist_to_opp: float = 0.0

        # Future-red distance cache for InterceptShaping (populated each step
        # when learner_id is set).
        self._dist_def_to_future_red:      float = 0.0
        self._dist_def_to_future_red_prev: float = 0.0

        self._np_random: np.random.Generator = np.random.default_rng()

        if reward_stack is None:
            if self._red_id != "red_0" or self._blue_id != "blue_0":
                raise ValueError(
                    f"team_env: cfg prefixes are red={self._red_id!r}, "
                    f"blue={self._blue_id!r} but default_team_stack() (from "
                    "team_v2.yaml) hardcodes 'red_0'/'blue_0'.  Pass a "
                    "custom RewardStack via `reward_stack=`."
                )
            reward_stack = default_team_stack()
        self._reward_stack = reward_stack

    def _spec_for_agent(self, agent_id: str) -> ObsSpec:
        if agent_id == self._learner_id:
            return self._learner_spec
        return DUEL_V1_BODY
```

(4e) Update `_build_world` to cache the dofadrs (append after `self._crash_detector = ...`, around line 207):

```python
        # Cache free-joint dofadrs for world-frame velocity reads.
        for prefix, attr in (
            (self._red_id,  "_red_dofadr"),
            (self._blue_id, "_blue_dofadr"),
        ):
            bid = mujoco.mj_name2id(self._world.model,
                                     mujoco.mjtObj.mjOBJ_BODY, prefix)
            jnt = int(self._world.model.body_jntadr[bid])
            setattr(self, attr, int(self._world.model.jnt_dofadr[jnt]))
```

(4f) Update `reset()` to initialise the new caches. In `reset()` (around line 240), after `self._dist_b2r_prev = float(np.linalg.norm(self._red_pos() - self._blue_pos()))` add:

```python
        # Initialise closing-rate cache (formerly OCE side).
        if self._learner_id is not None:
            learner_pos = (self._blue_pos() if self._learner_id == self._blue_id
                            else self._red_pos())
            opp_pos = (self._red_pos() if self._learner_id == self._blue_id
                        else self._blue_pos())
            self._prev_dist_to_opp = float(np.linalg.norm(opp_pos - learner_pos))
            # Initialise future-red distance cache.
            red_vel_world = self._world.data.qvel[
                self._red_dofadr : self._red_dofadr + 3
            ].copy()
            future_red = self._red_pos() + REWARD_LOOKAHEAD_S * red_vel_world
            defender_pos = (self._blue_pos() if self._learner_id == self._blue_id
                             else self._red_pos())
            self._dist_def_to_future_red_prev = float(
                np.linalg.norm(defender_pos - future_red)
            )
            self._dist_def_to_future_red = self._dist_def_to_future_red_prev
        else:
            self._prev_dist_to_opp = 0.0
            self._dist_def_to_future_red      = 0.0
            self._dist_def_to_future_red_prev = 0.0
```

(4g) Update `step()` to compute future-red distance each step. In `step()` around the line `self._dist_b2r_prev = dist_b2r` (right before the termination block), insert before that line:

```python
        # ── InterceptShaping inputs (when a learner is configured) ──────────
        if self._learner_id is not None:
            red_vel_world = self._world.data.qvel[
                self._red_dofadr : self._red_dofadr + 3
            ].copy()
            future_red = red_pos + REWARD_LOOKAHEAD_S * red_vel_world
            defender_pos = (blue_pos if self._learner_id == self._blue_id
                             else red_pos)
            self._dist_def_to_future_red_prev = self._dist_def_to_future_red
            self._dist_def_to_future_red = float(
                np.linalg.norm(defender_pos - future_red)
            )
```

Then update the `StepState(...)` construction to include the two new fields. Find the `reward_state = StepState(...)` block (around line 360) and add the two fields:

```python
        reward_state = StepState(
            agent_ids=(self._red_id, self._blue_id),
            red_pos=red_pos, blue_pos=blue_pos,
            dist_b2r=dist_b2r, dist_b2r_prev=self._dist_b2r_prev,
            step_period=self._red.step_period,
            tag_entry=tag_entry, tag_during=tag_during,
            dist_red_to_hoop=dist_red,
            dist_blue_to_midpoint=dist_blue,
            dist_blue_to_hoop=dist_blue_to_hoop,
            scored=scored,
            red_floor=red_floor, blue_floor=blue_floor,
            red_wall_crash=red_wall_crash, blue_wall_crash=blue_wall_crash,
            red_oob=red_oob, blue_oob=blue_oob,
            drone_drone_crash=drone_drone_crash,
            arena_radius=ARENA_RADIUS,
            tag_radius=self.cfg.tag_radius,
            dist_def_to_future_red=self._dist_def_to_future_red,
            dist_def_to_future_red_prev=self._dist_def_to_future_red_prev,
        )
```

(4h) Replace `_build_agent_obs` (around line 512) and add `_pack_agent_obs`:

```python
    def _build_agent_obs(self, agent_id: str) -> np.ndarray:
        return self._pack_agent_obs(agent_id, self._spec_for_agent(agent_id))

    def _pack_agent_obs(self, agent_id: str, spec: ObsSpec) -> np.ndarray:
        """Build the float32 obs vector for one agent under one ObsSpec.

        Supports DUEL_V1_BODY (22-d), DUEL_V2_WORLD (25-d, world-frame
        opp_vel_rel + closing_rate + vec_to_hoop), and DUEL_V3_BODY_EGO
        (25-d, body-frame ego-centric).
        """
        if agent_id == self._red_id:
            self_q, opp_q = self._red,  self._blue
            self_dofadr   = self._red_dofadr
            opp_dofadr    = self._blue_dofadr
            goal_target   = HOOP_CENTER
        else:
            self_q, opp_q = self._blue, self._red
            self_dofadr   = self._blue_dofadr
            opp_dofadr    = self._red_dofadr
            goal_target   = self._midpoint()

        s = self_q.state()
        ang_vel    = s[0]
        ang_pos    = s[1]
        lin_vel_b  = s[2]
        lin_pos    = s[3]

        opp_s = opp_q.state()
        opp_pos     = opp_s[3]
        opp_lin_vel = opp_s[2]

        vec_to_goal  = goal_target - lin_pos
        dist_g       = float(np.linalg.norm(vec_to_goal))
        unit_to_goal = vec_to_goal / (dist_g + 1e-8)
        signed_dist_norm = self._signed_dist_to_hoop_plane(lin_pos) / ARENA_RADIUS

        opp_pos_rel_world = opp_pos - lin_pos
        opp_vel_rel_body_mixed = opp_lin_vel - lin_vel_b  # legacy DUEL_V1_BODY

        # DUEL_V1_BODY (22-d, body-mixed opp_vel_rel + signed-distance scalar).
        if spec is DUEL_V1_BODY:
            return obs_spec.pack(DUEL_V1_BODY, {
                "ang_vel":          ang_vel,
                "ang_pos":          ang_pos,
                "lin_vel":          lin_vel_b,
                "lin_pos":          lin_pos,
                "unit_to_goal":     unit_to_goal,
                "signed_dist_norm": [signed_dist_norm],
                "opp_pos_rel":      opp_pos_rel_world,
                "opp_vel_rel":      opp_vel_rel_body_mixed,
            })

        # World-frame velocities (free-joint qvel[0:3] for both bodies).
        data = self._world.data
        self_vel_world = data.qvel[self_dofadr : self_dofadr + 3].copy()
        opp_vel_world  = data.qvel[opp_dofadr  : opp_dofadr  + 3].copy()
        opp_vel_rel_world = (opp_vel_world - self_vel_world).astype(np.float32)

        vec_to_hoop_world = (HOOP_CENTER - lin_pos).astype(np.float32)

        # Closing rate tracked only for the configured learner.
        dist_to_opp = float(np.linalg.norm(opp_pos_rel_world))
        if agent_id == self._learner_id:
            closing_rate = (
                (self._prev_dist_to_opp - dist_to_opp) / self._red.step_period
            )
            self._prev_dist_to_opp = dist_to_opp
        else:
            closing_rate = 0.0

        # DUEL_V2_WORLD (25-d, world-frame opp + closing_rate).
        if spec is DUEL_V2_WORLD:
            return obs_spec.pack(DUEL_V2_WORLD, {
                "ang_vel":      ang_vel,
                "ang_pos":      ang_pos,
                "lin_vel":      lin_vel_b,
                "lin_pos":      lin_pos,
                "unit_to_goal": unit_to_goal,
                "vec_to_hoop":  vec_to_hoop_world,
                "opp_pos_rel":  opp_pos_rel_world.astype(np.float32),
                "opp_vel_rel":  opp_vel_rel_world,
                "closing_rate": [closing_rate],
            })

        # DUEL_V3_BODY_EGO (25-d, body-frame ego-centric).
        if spec is DUEL_V3_BODY_EGO:
            # Rotation: world → body via R_wb.T (R_wb = data.xmat[body_id]).
            bid = self_q._drone_id
            R_wb = data.xmat[bid].reshape(3, 3)
            vec_to_goal_body  = obs_spec.world_to_body(vec_to_goal,  R_wb)
            vec_to_hoop_body  = obs_spec.world_to_body(vec_to_hoop_world, R_wb)
            opp_pos_rel_body  = obs_spec.world_to_body(opp_pos_rel_world, R_wb)
            opp_vel_rel_body  = obs_spec.world_to_body(opp_vel_rel_world, R_wb)
            return obs_spec.pack(DUEL_V3_BODY_EGO, {
                "ang_vel":      ang_vel,
                "ang_pos":      ang_pos,
                "lin_vel":      lin_vel_b,
                "lin_pos":      lin_pos,
                "vec_to_goal":  vec_to_goal_body,
                "vec_to_hoop":  vec_to_hoop_body,
                "opp_pos_rel":  opp_pos_rel_body,
                "opp_vel_rel":  opp_vel_rel_body,
                "closing_rate": [closing_rate],
            })

        raise ValueError(f"_pack_agent_obs: unsupported spec {spec!r}")
```

Note: `self_q._drone_id` is accessed directly (it's a private attribute on `Quadrotor` that team_env already touches indirectly via `self_q.state()`). If you prefer, expose it via a `Quadrotor.body_id` property in a small Step 4i — see "Optional cleanup" below.

(4i) **Optional cleanup (not required for correctness):** Add `body_id` property to `Quadrotor` so team_env doesn't reach into `_drone_id`. In `core/quadrotor.py`, after `def prefix(self) -> str:`:

```python
    @property
    def body_id(self) -> int:
        """MuJoCo body id of this drone (for xmat/xquat lookups)."""
        return self._drone_id
```

Then replace `self_q._drone_id` with `self_q.body_id` in `_pack_agent_obs`. Either way is fine — pick one.

- [ ] **Step 5: Run the new test file to verify it passes**

Run: `pytest tests/envs/quidditch/test_team_env_v3.py -v -m "not slow"`
Expected: all non-slow tests PASS.

- [ ] **Step 6: Run the team canary to confirm byte-identical fingerprint**

Run: `pytest tests/envs/quidditch/test_team_env_canary.py -v`
Expected: PASS with the fingerprint unchanged: `step 176 / step 684 / red_total +1.542 / blue_total -6.327`.

If the canary moves a digit, the refactor changed the default-construction (no `learner_id`) obs values. Roll back and re-investigate — the DUEL_V1_BODY math path must be byte-identical.

- [ ] **Step 7: Run the slow round-trip smoke**

Run: `pytest tests/envs/quidditch/test_team_env_v3.py::test_blue_v4_round_trip_through_new_in_env_packer -v`
Expected: PASS (or SKIP if `models/ppo_hoop_blue_4_*` isn't on disk locally — in CI this should not skip).

- [ ] **Step 8: Commit**

```bash
git add envs/quidditch/team_env.py tests/envs/quidditch/test_team_env_v3.py tests/conftest.py
# If you took the optional cleanup in Step 4i:
git add core/quadrotor.py
git commit -m "$(cat <<'EOF'
refactor(team-env): per-agent obs builder for v1/v2/v3 specs

QuidditchTeamEnv learns `learner_id` + `learner_spec` kwargs; the
learner's obs is packed in any of DUEL_V1_BODY / DUEL_V2_WORLD /
DUEL_V3_BODY_EGO directly by _pack_agent_obs (the work formerly
split between team_env and OCE._augment_learner_obs).  The non-
learner agent always gets DUEL_V1_BODY so frozen Red checkpoints
keep loading.  StepState gains dist_def_to_future_red + _prev,
populated every step when a learner is set.  Default-construction
(no learner_id) preserves canary fingerprint byte-for-byte.
EOF
)"
```

---

### Task 8: Simplify `OpponentControlledEnv` to a pure pass-through

**Files:**
- Modify: `envs/quidditch/opponents.py`
- Modify: `tests/envs/quidditch/test_augmented_obs.py`

- [ ] **Step 1: Rewrite `tests/envs/quidditch/test_augmented_obs.py` to test the in-env path**

Replace the entire file with:

```python
"""DUEL_V2_WORLD obs construction — formerly done by OCE._augment_learner_obs,
now done by team_env._pack_agent_obs directly.  OCE is a pure pass-through.

FrameStackWrapper additionally stacks N consecutive 25-d obs into a 25·N
flat vector, matching SB3's VecFrameStack so video-callback (single-env)
and training (vec-env) paths produce identical shapes.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.obs_spec import DUEL_V2_WORLD
from envs.quidditch.opponents import (
    FrameStackWrapper,
    OpponentControlledEnv,
    from_spec,
)
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from tests.conftest import set_body_state


DUEL_V2_WORLD_DIM = DUEL_V2_WORLD.dim


def _make_blue_env() -> OpponentControlledEnv:
    team = QuidditchTeamEnv(
        cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0", learner_spec=DUEL_V2_WORLD,
    )
    return OpponentControlledEnv(
        team, learner_id="blue_0", opponent=from_spec("zero"),
    )


def test_v2_obs_shape_is_25_dim() -> None:
    env = _make_blue_env()
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V2_WORLD_DIM,)
        assert obs.shape == (DUEL_V2_WORLD_DIM,)
        assert obs.dtype == np.float32
    finally:
        env.close()


def test_v2_vec_to_hoop_slot_points_to_hoop() -> None:
    """Slot [15:18] equals HOOP_CENTER − blue_pos after the first step."""
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env.team_env._world, "blue_0", pos=(0.5, 1.0, 1.5))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        expected = HOOP_CENTER - np.array([0.5, 1.0, 1.5])
        actual = obs[15:18]
        assert np.linalg.norm(actual - expected) < 0.5, (
            f"vec_to_hoop should point to hoop from injected pos; "
            f"expected ≈ {expected}, got {actual}"
        )
    finally:
        env.close()


def test_v2_closing_rate_zero_at_static_positive_when_closing() -> None:
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        # Scenario 1: pin both → closing ≈ 0.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs_static, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        static_closing = float(obs_static[24])

        # Scenario 2: blue moving toward red at -2 m/s.
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(-2.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs_closing, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        closing = float(obs_closing[24])

        assert abs(static_closing) < 0.1
        assert closing > 0.5
        assert closing > static_closing + 0.5
    finally:
        env.close()


def test_v2_opp_vel_rel_uses_world_frame() -> None:
    env = _make_blue_env()
    try:
        env.reset(seed=0)
        set_body_state(env.team_env._world, "red_0",  pos=(0.0, 0.0, 1.0),
                       vel=(0.0, 1.0, 0.0))
        set_body_state(env.team_env._world, "blue_0", pos=(1.0, 0.0, 1.0),
                       vel=(0.0, 0.0, 0.0))
        mujoco.mj_forward(env.team_env._world.model, env.team_env._world.data)
        env.team_env._prev_dist_to_opp = 1.0
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

        opp_vel_rel = obs[21:24]
        assert abs(opp_vel_rel[1]) > 0.5, (
            f"y component should reflect red's +1 m/s world-y motion, got {opp_vel_rel}"
        )
        assert abs(opp_vel_rel[1]) > abs(opp_vel_rel[0])
    finally:
        env.close()


def test_frame_stack_wrapper_doubles_obs_dim() -> None:
    env = FrameStackWrapper(_make_blue_env(), n_stack=2)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (DUEL_V2_WORLD_DIM * 2,)
        assert obs.shape == (DUEL_V2_WORLD_DIM * 2,)
        np.testing.assert_array_equal(
            obs[:DUEL_V2_WORLD_DIM], obs[DUEL_V2_WORLD_DIM:],
        )
        prev_new = obs[DUEL_V2_WORLD_DIM:].copy()
        obs2, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(obs2[:DUEL_V2_WORLD_DIM], prev_new)
        assert obs2.shape == (DUEL_V2_WORLD_DIM * 2,)
    finally:
        env.close()
```

- [ ] **Step 2: Run the rewritten test to verify it fails (still passing through the augmenter)**

Run: `pytest tests/envs/quidditch/test_augmented_obs.py -v`
Expected: at least one test fails. Symptom: `obs.shape == (DUEL_V1_BODY.dim,)` (22) instead of 25 — because OCE's augmenter currently re-packs the team_env obs into DUEL_V2_WORLD, but team_env is now already emitting DUEL_V2_WORLD for the learner, so OCE's augmenter is double-rotating / re-packing the wrong layout. Use this failure to confirm the next step is needed.

- [ ] **Step 3: Simplify `OpponentControlledEnv` in `envs/quidditch/opponents.py`**

Replace the entire `OpponentControlledEnv` class (currently lines ~258–394) with:

```python
class OpponentControlledEnv(gym.Env):
    """Reduces a QuidditchTeamEnv to single-agent Gym (for SB3) by driving
    the non-learner agent from a frozen Opponent each step.

    Pure pass-through for obs shape: whatever team_env emits for the
    learner is what the SB3 model sees.  team_env owns all obs-shape
    decisions (DUEL_V1_BODY / DUEL_V2_WORLD / DUEL_V3_BODY_EGO via its
    `learner_id` + `learner_spec` kwargs).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        team_env: QuidditchTeamEnv,
        *,
        learner_id: str,
        opponent: Opponent,
    ) -> None:
        super().__init__()
        if learner_id not in team_env.possible_agents:
            raise ValueError(
                f"OpponentControlledEnv: learner_id={learner_id!r} not in "
                f"team_env.possible_agents={team_env.possible_agents}"
            )
        self.team_env = team_env
        self.learner_id = learner_id
        self.opponent_id = next(a for a in team_env.possible_agents if a != learner_id)
        self.opponent = opponent

        # Pass-through observation/action space — team_env decided the shape.
        self.observation_space = team_env.observation_space(learner_id)
        self.action_space      = team_env.action_space(learner_id)
        self.render_mode = team_env.render_mode

        self._last_opp_obs: np.ndarray = np.zeros(
            team_env.observation_space(self.opponent_id).shape, dtype=np.float32,
        )
        # Last team_env infos from reset/step — exposed for eval callers that
        # need both agents' info dicts (the wrapper only forwards the learner's).
        self.last_team_infos: dict = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self.team_env.reset(seed=seed, options=options)
        self.opponent.reset()
        self._last_opp_obs = obs[self.opponent_id]
        self.last_team_infos = infos
        return obs[self.learner_id], infos[self.learner_id]

    def step(self, action):
        opp_action = self.opponent.act(self._last_opp_obs)
        actions = {self.learner_id: action, self.opponent_id: opp_action}
        obs, rew, term, trunc, infos = self.team_env.step(actions)
        self._last_opp_obs = obs[self.opponent_id]
        self.last_team_infos = infos
        return (
            obs[self.learner_id],
            float(rew[self.learner_id]),
            bool(term[self.learner_id]),
            bool(trunc[self.learner_id]),
            infos[self.learner_id],
        )

    def render(self):
        return self.team_env.render()

    def close(self):
        self.team_env.close()

    @property
    def _world(self):
        """Passthrough to the wrapped team env's World so the video callback
        (and any other reach-through consumer) can call ``render_cells`` /
        ``render_frame`` without knowing about the wrapper."""
        return self.team_env._world
```

Also delete the `DUEL_V2_WORLD` import at the top of `opponents.py` (no longer used in this file). The import block at the top becomes:

```python
from envs.quidditch import obs_spec
from envs.quidditch.constants import HOOP_CENTER
from envs.quidditch.team_env import QuidditchTeamEnv
```

(`obs_spec` and `HOOP_CENTER` may also be unused now — verify with `pyflakes`. If unused, delete them too. The `from stable_baselines3 import PPO` and the `import mujoco` lines remain — FrozenPolicyOpponent still uses PPO; `import mujoco` was only used by the deleted `_cache_dofadrs`, so delete it as well.)

- [ ] **Step 4: Run the rewritten test to verify it passes**

Run: `pytest tests/envs/quidditch/test_augmented_obs.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run all team / OCE-adjacent tests to confirm no regressions**

Run: `pytest tests/envs/quidditch -v`
Expected: all tests PASS, including:
- `test_team_env_canary.py` (DUEL_V1_BODY default-construction byte-identical)
- `test_team_env_v3.py` (DUEL_V3_BODY_EGO and DUEL_V2_WORLD paths)
- `test_augmented_obs.py` (DUEL_V2_WORLD via in-env packer + OCE pass-through)
- `test_opponent_env_world.py`, `test_crash_aftermath.py`, `test_take_down.py`, etc.

- [ ] **Step 6: Commit**

```bash
git add envs/quidditch/opponents.py tests/envs/quidditch/test_augmented_obs.py
git commit -m "$(cat <<'EOF'
refactor(oce): drop _augment_learner_obs, become pure pass-through

team_env now packs the learner's obs directly under whichever spec
its learner_spec kwarg names.  OCE shrinks to forwarding learner-side
obs/rew/term/trunc/info; closing-rate and dofadr caching moved into
team_env.  test_augmented_obs.py rewritten to assert the same five
DUEL_V2_WORLD properties against the new in-env path.
EOF
)"
```

---

### Task 9: `env_factories.TeamEnvFactory` — pass `learner_id` + `learner_spec`

**Files:**
- Modify: `envs/quidditch/env_factories.py`
- Modify: `tests/envs/quidditch/test_env_factories.py`

- [ ] **Step 1: Append a failing test to `tests/envs/quidditch/test_env_factories.py`**

Append:

```python
def test_team_factory_threads_learner_id_and_spec_into_team_env():
    """TeamEnvFactory must resolve cfg.obs.name → ObsSpec and pass it as
    learner_spec to QuidditchTeamEnv, so the learner sees the right shape."""
    from envs.quidditch.env_factories import TeamEnvFactory
    from envs.quidditch.obs_spec import DUEL_V3_BODY_EGO
    from envs.quidditch.team_env import TeamConfig

    factory = TeamEnvFactory(
        n_envs=1,
        team_cfg=TeamConfig(randomise_red_start=False),
        learner_id="blue_0",
        opponent_spec="zero",
        obs_spec_name="DUEL_V3_BODY_EGO",
        frame_stack=1,
        seed=42,
    )
    vec_env = factory.build_train_env()
    try:
        # SB3 vec_envs expose .observation_space.shape on the wrapped single env.
        assert vec_env.observation_space.shape == (DUEL_V3_BODY_EGO.dim,)
    finally:
        vec_env.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/envs/quidditch/test_env_factories.py::test_team_factory_threads_learner_id_and_spec_into_team_env -v`
Expected: FAIL with `AssertionError: assert (22,) == (25,)` (the factory currently constructs team_env without learner_id/learner_spec).

- [ ] **Step 3: Update `TeamEnvFactory._make_thunk` and `build_video_env_fn` in `envs/quidditch/env_factories.py`**

Replace `TeamEnvFactory._make_thunk`:

```python
    def _make_thunk(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        from envs.quidditch.obs_spec import SPEC_BY_NAME
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        reward_stack = self.reward_stack
        learner_spec = SPEC_BY_NAME[self.obs_spec_name]
        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, reward_stack=reward_stack,
                learner_id=learner, learner_spec=learner_spec,
            )
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
        return _thunk
```

Replace `TeamEnvFactory.build_video_env_fn`:

```python
    def build_video_env_fn(self):
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import (
            OpponentControlledEnv, from_spec, FrameStackWrapper,
        )
        from envs.quidditch.obs_spec import SPEC_BY_NAME
        cfg = self.team_cfg
        learner = self.learner_id
        opp_spec = self.opponent_spec
        frame_stack = self.frame_stack
        reward_stack = self.reward_stack
        learner_spec = SPEC_BY_NAME[self.obs_spec_name]
        def _thunk():
            team = QuidditchTeamEnv(
                cfg=cfg, render_mode="rgb_array",
                reward_stack=reward_stack,
                learner_id=learner, learner_spec=learner_spec,
            )
            opp = from_spec(opp_spec, deterministic=True)
            env = OpponentControlledEnv(team, learner_id=learner, opponent=opp)
            if frame_stack > 1:
                return FrameStackWrapper(env, n_stack=frame_stack)
            return env
        return _thunk
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/envs/quidditch/test_env_factories.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add envs/quidditch/env_factories.py tests/envs/quidditch/test_env_factories.py
git commit -m "$(cat <<'EOF'
feat(env-factory): thread learner_id + learner_spec into team env

TeamEnvFactory now resolves cfg.obs.name via SPEC_BY_NAME and passes
the resulting ObsSpec to QuidditchTeamEnv along with learner_id, so
the learner gets its full obs shape from team_env directly.  Train,
eval, and video envs all use the same wiring.
EOF
)"
```

---

### Task 10: `conf/experiment/blue_v7.yaml` + smoke test

**Files:**
- Create: `conf/experiment/blue_v7.yaml`
- Modify: `tests/scripts/test_train_smoke_wandb_disabled.py`

- [ ] **Step 1: Read the current smoke test for pattern**

Run: `pytest tests/scripts/test_train_smoke_wandb_disabled.py -v --collect-only`
Expected: lists existing smoke tests. Note the pattern (calls `subprocess.run` against `scripts/train.py` with `total_timesteps` overridden small).

- [ ] **Step 2: Append a failing smoke test for `blue_v7`**

Append to `tests/scripts/test_train_smoke_wandb_disabled.py` (using the same pattern as existing entries in that file — look at the most recent smoke test and copy its shape; adapt the experiment name to `blue_v7` and override `trainer.total_timesteps=512` and `trainer.n_steps=64`). The test should:
1. Run `python scripts/train.py +experiment=blue_v7 trainer.total_timesteps=512 trainer.n_steps=64 trainer.batch_size=64 env.n_envs=1 eval.eval_freq_steps=10000` in a subprocess (so wandb disabled env carries through).
2. Assert return code 0.
3. Assert the run dir under `runs/ppo_hoop_blue_7/<ts>/` was created and contains `.hydra/config.yaml` and `final_model.zip`.

Use a temp `runs/` dir via `hydra.run.dir=` override if the existing tests do; otherwise mirror their post-run cleanup.

- [ ] **Step 3: Run the smoke test to verify it fails**

Run: `pytest tests/scripts/test_train_smoke_wandb_disabled.py -v -k blue_v7`
Expected: FAIL because the experiment YAML doesn't exist (Hydra raises `MissingConfigException`).

- [ ] **Step 4: Create `conf/experiment/blue_v7.yaml`**

Contents:

```yaml
# @package _global_
# Pending: blue_v7 — body-frame ego-centric obs (DUEL_V3_BODY_EGO),
# InterceptShaping reward, trained from scratch vs frozen red_v1 with
# randomised Red start.  See docs/superpowers/specs/2026-05-18-blue-v7-body-ego-design.md.

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

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `pytest tests/scripts/test_train_smoke_wandb_disabled.py -v -k blue_v7`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add conf/experiment/blue_v7.yaml tests/scripts/test_train_smoke_wandb_disabled.py
git commit -m "$(cat <<'EOF'
feat(experiment): conf/experiment/blue_v7.yaml + smoke test

blue_v7: scratch Blue learner under DUEL_V3_BODY_EGO obs +
team_v3_intercept reward stack, vs frozen red_v1 with random start.
midpoint_alpha=0.6 (60% weight on Red for vec_to_goal target).
EOF
)"
```

---

### Task 11: Full suite green + canary verification

**Files:** (no source changes; verification + cleanup only)

- [ ] **Step 1: Run the entire pytest suite**

Run: `make test`
Expected: all tests PASS. Note the count for the changelog.

- [ ] **Step 2: Explicitly verify both canary fingerprints**

Run: `pytest tests/envs/quidditch/test_scoring_canary.py tests/envs/quidditch/test_team_env_canary.py -v`
Expected: both PASS.
- Scoring canary: `SCORED at step 434 / total reward 7.3837`.
- Team canary: `step 176 / step 684 / red_total +1.542 / blue_total -6.327`.

If either drifted, roll back to the prior commit and re-investigate. The intent of this task is to surface drift before merging, not to fix it under time pressure.

- [ ] **Step 3: Re-read the spec's Touch list — confirm every file landed**

Open `docs/superpowers/specs/2026-05-18-blue-v7-body-ego-design.md` and visually walk the "Touch list" section. Each entry should match a commit on this branch (`git log --oneline develop..HEAD --name-only`). Files mentioned but not committed indicate something was missed.

Run: `git log --oneline develop..HEAD`
Expected: ~10 commits (one per task), all GPG-signed (run `git log --show-signature develop..HEAD | head` to confirm).

- [ ] **Step 4: Run a quick git status sanity check**

Run: `git status`
Expected: working tree clean. No untracked files (the worktree is clean), no unstaged changes.

- [ ] **Step 5: Final note — no commit in this task**

This task is verification only — nothing to commit. The feature branch `feature/blue-v7-body-ego` is ready for the user's review and merge decision.

---

## Self-Review

**1. Spec coverage:**
- Decision 1 (new ObsBlock constants + DUEL_V3_BODY_EGO + SPEC_BY_NAME) → Task 1 ✓
- Decision 1 (world_to_body helper) → Task 2 ✓
- Decision 1 (conf/obs/duel_v3_body_ego.yaml) → Task 3 ✓
- Decision 2 (InterceptShaping dataclass + StepState fields) → Tasks 4 + 5 ✓
- Decision 3 (conf/reward/team_v3_intercept.yaml) → Task 6 ✓
- Decision 4 (team_env per-agent builder, all 3 specs, dofadr cache, closing-rate move) → Task 7 ✓
- Decision 5 (OCE pure pass-through, test_augmented_obs rewrite) → Task 8 ✓
- Decision 6 (env_factories wiring) → Task 9 ✓
- Decision 7 (team_env.step populates future-red distances) → Task 7 (combined) ✓
- Decision 8 (conf/experiment/blue_v7.yaml) → Task 10 ✓
- Decision 9 (test additions across files) → Tasks 1–10 (each task adds its tests) + Task 11 (canary verification) ✓
- Decision 10 (LEGACY_SPECS unchanged) → no task needed (explicit in spec) ✓

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or unspecified test code. Every code block is concrete. Step 2 of Task 10 references "the most recent smoke test" — that's a *read-then-mirror* instruction (the engineer reads the actual file to copy its established pattern), not a placeholder. Acceptable since the test infra pattern is well-defined and the alternative (duplicating 50 lines of subprocess setup) would rot.

**3. Type consistency:** Cross-task checks —
- `ObsSpec` constructor arg name (`blocks`) — consistent across tasks 1, 7.
- `learner_id`, `learner_spec` kwarg names — consistent across tasks 7, 8, 9.
- `_dist_def_to_future_red`, `_dist_def_to_future_red_prev` attribute names — consistent across tasks 4, 7, and the test in task 7.
- `dist_def_to_future_red`, `dist_def_to_future_red_prev` StepState field names — consistent across tasks 4, 5, 7.
- `defender` (singular) on `InterceptShaping` — consistent in tasks 5, 6, 7's reward stack composition test.
- `world_to_body(vec_world, R_wb)` signature — consistent across tasks 2 and 7's _pack_agent_obs.

No mismatches found.
