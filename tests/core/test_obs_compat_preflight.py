"""Behavior contract for core.obs_compat.preflight.

preflight(parent_uri, child_obs_name, child_n_stack) -> PreflightReport
- Resolves parent_uri to its obs spec WITHOUT loading the model
- Returns a structured report (no print, no sys.exit)
- compatible=False ⇒ check_obs_compat would strict-raise
- surgery_required=True ⇒ would need init.mode=warm_start
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _make_parent(tmp_path: Path, *, obs_name: str, n_stack: int) -> Path:
    d = tmp_path / "parent"
    (d / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "parent",
        "obs": {"name": obs_name, "n_stack": n_stack},
    }), d / ".hydra" / "config.yaml")
    return d


def test_preflight_matched_specs_returns_compatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    assert report.compatible is True
    assert report.surgery_required is False
    assert report.parent_spec_name == "DUEL_V2_WORLD"
    assert report.child_spec_name == "DUEL_V2_WORLD"


def test_preflight_n_stack_mismatch_is_incompatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=1)
    assert report.compatible is False
    assert report.surgery_required is True
    assert report.parent_n_stack == 3
    assert report.child_n_stack == 1


def test_preflight_spec_change_is_surgery_required(tmp_path: Path) -> None:
    """Different obs specs (e.g. V1_BODY → V2_WORLD) need warm_start."""
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V1_BODY", n_stack=1)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    assert report.compatible is False
    assert report.surgery_required is True
    assert any(d.status in ("frame_changed", "removed", "added") for d in report.diff)


def test_preflight_handles_v3_body_ego(tmp_path: Path) -> None:
    """DUEL_V3_BODY_EGO is a recognized spec (added 2026-05-18)."""
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V3_BODY_EGO", n_stack=1)
    report = preflight(str(parent), child_obs_name="DUEL_V3_BODY_EGO", child_n_stack=1)
    assert report.compatible is True


def test_preflight_unknown_parent_spec_raises(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="NO_SUCH_SPEC", n_stack=1)
    with pytest.raises(KeyError, match="NO_SUCH_SPEC"):
        preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)


def test_preflight_missing_parent_config_raises(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    with pytest.raises(FileNotFoundError):
        preflight(str(tmp_path / "does-not-exist"),
                  child_obs_name="DUEL_V2_WORLD", child_n_stack=3)


def test_preflight_diff_columns_aligned_for_compatible(tmp_path: Path) -> None:
    from core.obs_compat import preflight
    parent = _make_parent(tmp_path, obs_name="DUEL_V2_WORLD", n_stack=3)
    report = preflight(str(parent), child_obs_name="DUEL_V2_WORLD", child_n_stack=3)
    assert all(d.status == "matched" for d in report.diff)
