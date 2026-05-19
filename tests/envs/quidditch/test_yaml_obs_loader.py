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
