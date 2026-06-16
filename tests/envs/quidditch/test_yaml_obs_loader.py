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


def test_load_obs_yaml_simple():
    """load_obs_yaml reads conf/obs/simple.yaml and builds an ObsSpec."""
    spec = obs_spec.load_obs_yaml("simple")
    assert isinstance(spec, ObsSpec)
    assert spec.dim == 16


def test_load_obs_yaml_missing_blocks_field_raises(tmp_path: Path, monkeypatch):
    """If a YAML lacks blocks:, load_obs_yaml raises KeyError mentioning blocks."""
    repo_root = Path(obs_spec.__file__).resolve().parents[2]
    target = repo_root / "conf" / "obs" / "_test_no_blocks.yaml"
    target.write_text("name: TEST\nn_stack: 1\n")
    try:
        with pytest.raises(KeyError, match="blocks"):
            obs_spec.load_obs_yaml("_test_no_blocks")
    finally:
        target.unlink()


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


def test_obs_config_schema_carries_blocks():
    from config_schema import ObsConfig
    cfg = ObsConfig(name="X", n_stack=2, blocks=["ANG_VEL", "ANG_POS"])
    assert cfg.blocks == ["ANG_VEL", "ANG_POS"]


def test_obs_config_default_blocks_is_empty_list():
    """Default empty list keeps current YAMLs (no blocks: field) loading."""
    from config_schema import ObsConfig
    cfg = ObsConfig()
    assert cfg.blocks == []


def test_composed_constants_are_gone():
    """The composed-spec Python constants are deleted; only ObsBlock and helpers remain."""
    from envs.quidditch import obs_spec
    for removed in ("SIMPLE_ENV_OBS", "DUEL_V1_BODY", "DUEL_V2_WORLD",
                    "DUEL_V3_BODY_EGO", "SPEC_BY_NAME"):
        assert not hasattr(obs_spec, removed), f"{removed} should be deleted"
