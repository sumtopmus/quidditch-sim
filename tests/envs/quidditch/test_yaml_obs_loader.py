"""Tests for the YAML-driven obs loader (BLOCK_BY_NAME, build_spec_from_block_names, load_obs_yaml)."""
from pathlib import Path

import pytest

from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import (
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM,
    ObsSpec,
)


def test_block_by_name_contains_every_module_level_obs_block():
    """Every ObsBlock attribute on the module must appear in BLOCK_BY_NAME."""
    expected = {
        name: obj
        for name, obj in vars(obs_spec).items()
        if isinstance(obj, obs_spec.ObsBlock)
    }
    assert obs_spec.BLOCK_BY_NAME == expected


def test_block_by_name_keys_are_python_identifiers():
    """Keys are uppercase Python identifiers (ANG_VEL, OPP_VEL_REL_BODY_EGO, ...)."""
    for name in obs_spec.BLOCK_BY_NAME:
        assert name.isidentifier(), f"{name!r} is not a valid Python identifier"
        assert name.isupper() or "_" in name, f"{name!r} should be UPPER_SNAKE_CASE"


def test_build_spec_from_block_names_constructs_ordered_spec():
    spec = obs_spec.build_spec_from_block_names(
        ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS", "UNIT_TO_GOAL", "SIGNED_DIST_NORM"]
    )
    assert isinstance(spec, ObsSpec)
    assert spec.blocks == (ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL, SIGNED_DIST_NORM)
    assert spec.dim == 16  # 3+3+3+3+3+1


def test_build_spec_from_block_names_raises_on_unknown_block():
    with pytest.raises(KeyError) as excinfo:
        obs_spec.build_spec_from_block_names(["ANG_VEL", "DOES_NOT_EXIST"])
    msg = str(excinfo.value)
    assert "DOES_NOT_EXIST" in msg
    assert "ANG_VEL" in msg  # error lists known blocks


def test_load_obs_yaml_simple(tmp_path: Path, monkeypatch):
    """load_obs_yaml reads conf/obs/<stem>.yaml and builds an ObsSpec.

    Uses the repo's real conf/obs/simple.yaml — at this task the YAML still
    holds the legacy `name`-only schema, so the test only asserts that the
    helper raises a clear error when `blocks:` is absent.
    """
    with pytest.raises(KeyError, match="blocks"):
        obs_spec.load_obs_yaml("simple")


def test_opp_vel_rel_variants_have_unique_names():
    """The 3 opp_vel_rel variants must carry distinct .name fields after rename."""
    from envs.quidditch.obs_spec import (
        OPP_VEL_REL_BODY, OPP_VEL_REL_WORLD, OPP_VEL_REL_BODY_EGO,
    )
    names = {OPP_VEL_REL_BODY.name, OPP_VEL_REL_WORLD.name, OPP_VEL_REL_BODY_EGO.name}
    assert names == {"opp_vel_rel_body_mixed", "opp_vel_rel_world", "opp_vel_rel_body_ego"}


def test_vec_to_hoop_variants_have_unique_names():
    from envs.quidditch.obs_spec import VEC_TO_HOOP, VEC_TO_HOOP_BODY
    assert VEC_TO_HOOP.name == "vec_to_hoop_world"
    assert VEC_TO_HOOP_BODY.name == "vec_to_hoop_body"


def test_opp_pos_rel_variants_have_unique_names():
    from envs.quidditch.obs_spec import OPP_POS_REL, OPP_POS_REL_BODY
    assert OPP_POS_REL.name == "opp_pos_rel_world"
    assert OPP_POS_REL_BODY.name == "opp_pos_rel_body"


def test_read_obs_spec_translates_legacy_opp_vel_rel_body_mixed(tmp_path):
    """A run_info.toml with the pre-rename name field parses to the new name."""
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 3\n'
        'n_stack = 1\n'
        'slots = [\n'
        '  {name = "opp_vel_rel", dim = 3, frame = "body_mixed",'
        ' notes = "legacy: each velocity in its own body frame"},\n'
        ']\n'
    )
    spec, n_stack = read_obs_spec(info)
    assert spec.blocks[0].name == "opp_vel_rel_body_mixed"
    assert n_stack == 1


def test_read_obs_spec_translates_legacy_vec_to_hoop_world(tmp_path):
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 3\n'
        'n_stack = 3\n'
        'slots = [\n'
        '  {name = "vec_to_hoop", dim = 3, frame = "world",'
        ' notes = "HOOP_CENTER - learner_pos, not normalized"},\n'
        ']\n'
    )
    spec, _ = read_obs_spec(info)
    assert spec.blocks[0].name == "vec_to_hoop_world"


def test_read_obs_spec_passthrough_for_non_renamed_names(tmp_path):
    """Single-variant blocks (ang_vel, lin_pos, ...) keep their existing names."""
    from scripts._train_common import read_obs_spec
    info = tmp_path / "run_info.toml"
    info.write_text(
        '[obs]\n'
        'dim = 6\n'
        'n_stack = 1\n'
        'slots = [\n'
        '  {name = "ang_vel", dim = 3, frame = "body"},\n'
        '  {name = "lin_pos", dim = 3, frame = "world"},\n'
        ']\n'
    )
    spec, _ = read_obs_spec(info)
    assert [b.name for b in spec.blocks] == ["ang_vel", "lin_pos"]
