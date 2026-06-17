"""Unit tests for envs.quidditch.obs_spec dataclasses."""
import pytest

from envs.quidditch import obs_spec


def test_obsblock_equality_full_tuple():
    a = obs_spec.ObsBlock("opp_vel_rel", dim=3, frame="world")
    b = obs_spec.ObsBlock("opp_vel_rel", dim=3, frame="world")
    c = obs_spec.ObsBlock("opp_vel_rel", dim=3, frame="body_mixed")
    d = obs_spec.ObsBlock("opp_vel_rel", dim=2, frame="world")
    assert a == b
    assert a != c              # frame differs
    assert a != d              # dim differs


def test_obsblock_notes_distinguishes_at_equality_level():
    # Notes participate in dataclass equality (frozen=True default eq=True).
    # check_obs_compat treats notes as informational; ObsBlock equality is
    # strict so the spec module doesn't lose information.
    a = obs_spec.ObsBlock("x", dim=1, notes="first")
    b = obs_spec.ObsBlock("x", dim=1, notes="second")
    assert a != b


def test_obsblock_is_frozen():
    b = obs_spec.ObsBlock("x", dim=1)
    with pytest.raises(Exception):  # FrozenInstanceError
        b.dim = 2  # type: ignore[misc]


def test_obsspec_dim_sums_blocks():
    spec = obs_spec.ObsSpec((
        obs_spec.ObsBlock("a", dim=3),
        obs_spec.ObsBlock("b", dim=1),
        obs_spec.ObsBlock("c", dim=4),
    ))
    assert spec.dim == 8


def test_obsspec_offsets_returns_correct_slices():
    A = obs_spec.ObsBlock("a", dim=3)
    B = obs_spec.ObsBlock("b", dim=1)
    C = obs_spec.ObsBlock("c", dim=4)
    spec = obs_spec.ObsSpec((A, B, C))
    offs = spec.offsets()
    assert offs == [(A, slice(0, 3)), (B, slice(3, 4)), (C, slice(4, 8))]


def test_obsspec_equality_is_structural():
    A1 = obs_spec.ObsBlock("a", dim=1)
    A2 = obs_spec.ObsBlock("a", dim=1)
    assert obs_spec.ObsSpec((A1,)) == obs_spec.ObsSpec((A2,))


import numpy as np


def test_pack_concatenates_in_spec_order():
    A = obs_spec.ObsBlock("a", dim=3)
    B = obs_spec.ObsBlock("b", dim=1)
    spec = obs_spec.ObsSpec((A, B))
    arr = obs_spec.pack(spec, {"a": np.array([1, 2, 3], dtype=np.float32),
                                "b": np.array([4],       dtype=np.float32)})
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, np.array([1, 2, 3, 4], dtype=np.float32))


def test_pack_accepts_python_list_for_scalar_block():
    spec = obs_spec.ObsSpec((obs_spec.ObsBlock("s", dim=1),))
    arr = obs_spec.pack(spec, {"s": [0.5]})
    np.testing.assert_array_equal(arr, np.array([0.5], dtype=np.float32))


def test_pack_raises_on_missing_block():
    spec = obs_spec.ObsSpec((obs_spec.ObsBlock("a", dim=1),))
    with pytest.raises(KeyError):
        obs_spec.pack(spec, {})


def test_pack_raises_on_dim_mismatch():
    spec = obs_spec.ObsSpec((obs_spec.ObsBlock("a", dim=3),))
    with pytest.raises(ValueError):
        obs_spec.pack(spec, {"a": np.array([1, 2], dtype=np.float32)})


# Load composed specs from conf/obs/*.yaml at module level so test bodies
# stay free of repeated load_obs_yaml() calls.
SIMPLE_ENV_OBS = obs_spec.load_obs_yaml("simple")
DUEL_V1_BODY = obs_spec.load_obs_yaml("duel_v1_body")
DUEL_V2_WORLD = obs_spec.load_obs_yaml("duel_v2_world")
DUEL_V3_BODY_EGO = obs_spec.load_obs_yaml("duel_v3_body_ego")


def test_simple_env_obs_dim_is_16():
    assert SIMPLE_ENV_OBS.dim == 16


def test_team_env_obs_dim_is_22():
    assert DUEL_V1_BODY.dim == 22


def test_augmented_obs_dim_is_25():
    assert DUEL_V2_WORLD.dim == 25


def test_simple_env_obs_is_prefix_of_team_env_obs():
    # Cross-env obs compatibility (16->22) depends on this prefix relationship.
    n = len(SIMPLE_ENV_OBS.blocks)
    assert DUEL_V1_BODY.blocks[:n] == SIMPLE_ENV_OBS.blocks


def test_opp_vel_rel_body_and_world_are_distinct():
    body = obs_spec.OPP_VEL_REL_BODY
    world = obs_spec.OPP_VEL_REL_WORLD
    assert body.name == "opp_vel_rel_body_mixed"
    assert world.name == "opp_vel_rel_world"
    assert body.dim == world.dim == 3
    assert body.frame != world.frame
    assert body != world


def test_team_env_obs_uses_body_mixed_opp_vel_rel():
    # The 22-d legacy team obs is body_mixed; the augmented 25-d uses world.
    assert obs_spec.OPP_VEL_REL_BODY in DUEL_V1_BODY.blocks
    assert obs_spec.OPP_VEL_REL_BODY not in DUEL_V2_WORLD.blocks
    assert obs_spec.OPP_VEL_REL_WORLD in DUEL_V2_WORLD.blocks


def test_vec_to_goal_body_block_is_distinct_from_unit_to_goal():
    g_body  = obs_spec.VEC_TO_GOAL_BODY
    g_world = obs_spec.UNIT_TO_GOAL
    assert g_body.name == "vec_to_goal"
    assert g_world.name == "unit_to_goal"
    assert g_body != g_world


def test_vec_to_hoop_body_and_world_are_distinct():
    body  = obs_spec.VEC_TO_HOOP_BODY
    world = obs_spec.VEC_TO_HOOP
    assert body.name == "vec_to_hoop_body"
    assert world.name == "vec_to_hoop_world"
    assert body.dim == world.dim == 3
    assert body.frame == "body"
    assert world.frame == "world"
    assert body != world


def test_opp_pos_rel_body_and_world_are_distinct():
    body  = obs_spec.OPP_POS_REL_BODY
    world = obs_spec.OPP_POS_REL
    assert body.name == "opp_pos_rel_body"
    assert world.name == "opp_pos_rel_world"
    assert body.dim == world.dim == 3
    assert body.frame == "body"
    assert world.frame == "world"
    assert body != world


def test_opp_vel_rel_body_ego_is_distinct_from_legacy_body_mixed_and_world():
    ego        = obs_spec.OPP_VEL_REL_BODY_EGO
    body_mixed = obs_spec.OPP_VEL_REL_BODY
    world      = obs_spec.OPP_VEL_REL_WORLD
    assert ego.name == "opp_vel_rel_body_ego"
    assert body_mixed.name == "opp_vel_rel_body_mixed"
    assert world.name == "opp_vel_rel_world"
    assert ego.frame == "body"
    assert body_mixed.frame == "body_mixed"
    assert world.frame == "world"
    assert ego != body_mixed
    assert ego != world


def test_duel_v3_body_ego_dim_is_25():
    assert DUEL_V3_BODY_EGO.dim == 25


def test_duel_v3_body_ego_block_names_in_order():
    names = [b.name for b in DUEL_V3_BODY_EGO.blocks]
    assert names == [
        "ang_vel", "ang_pos", "lin_vel", "lin_pos",
        "vec_to_goal", "vec_to_hoop_body", "opp_pos_rel_body", "opp_vel_rel_body_ego",
        "closing_rate",
    ]


def test_duel_v3_body_ego_uses_body_frame_for_relative_blocks():
    spec = DUEL_V3_BODY_EGO
    assert obs_spec.VEC_TO_GOAL_BODY     in spec.blocks
    assert obs_spec.VEC_TO_HOOP_BODY     in spec.blocks
    assert obs_spec.OPP_POS_REL_BODY     in spec.blocks
    assert obs_spec.OPP_VEL_REL_BODY_EGO in spec.blocks
    # No world-frame opp blocks in v3.
    assert obs_spec.OPP_POS_REL          not in spec.blocks
    assert obs_spec.OPP_VEL_REL_WORLD    not in spec.blocks


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


def test_duel_v3_body_ego_yaml_resolves_to_canonical_spec():
    """conf/obs/duel_v3_body_ego.yaml must declare name=DUEL_V3_BODY_EGO and
    its blocks list must rebuild the canonical 25-d body-frame spec."""
    from pathlib import Path
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(__file__).resolve().parents[3] / "conf" / "obs" / "duel_v3_body_ego.yaml")
    assert cfg.name == "DUEL_V3_BODY_EGO"
    assert int(cfg.n_stack) == 3
    assert obs_spec.load_obs_yaml("duel_v3_body_ego") == DUEL_V3_BODY_EGO
