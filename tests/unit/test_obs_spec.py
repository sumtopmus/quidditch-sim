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


def test_simple_env_obs_dim_is_16():
    assert obs_spec.SIMPLE_ENV_OBS.dim == 16


def test_team_env_obs_dim_is_22():
    assert obs_spec.TEAM_ENV_OBS.dim == 22


def test_augmented_obs_dim_is_25():
    assert obs_spec.AUGMENTED_OBS.dim == 25


def test_simple_env_obs_is_prefix_of_team_env_obs():
    # warm_start_ppo (16->22) depends on this prefix relationship.
    n = len(obs_spec.SIMPLE_ENV_OBS.blocks)
    assert obs_spec.TEAM_ENV_OBS.blocks[:n] == obs_spec.SIMPLE_ENV_OBS.blocks


def test_opp_vel_rel_body_and_world_are_distinct():
    body = obs_spec.OPP_VEL_REL_BODY
    world = obs_spec.OPP_VEL_REL_WORLD
    assert body.name == world.name == "opp_vel_rel"
    assert body.dim == world.dim == 3
    assert body.frame != world.frame
    assert body != world


def test_team_env_obs_uses_body_mixed_opp_vel_rel():
    # The 22-d legacy team obs is body_mixed; the augmented 25-d uses world.
    assert obs_spec.OPP_VEL_REL_BODY in obs_spec.TEAM_ENV_OBS.blocks
    assert obs_spec.OPP_VEL_REL_BODY not in obs_spec.AUGMENTED_OBS.blocks
    assert obs_spec.OPP_VEL_REL_WORLD in obs_spec.AUGMENTED_OBS.blocks
