"""check_obs_compat — load-time obs-spec compatibility check."""
from pathlib import Path

import pytest

from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import (
    ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL,
    SIGNED_DIST_NORM, VEC_TO_HOOP, OPP_POS_REL,
    OPP_VEL_REL_BODY, OPP_VEL_REL_WORLD, CLOSING_RATE,
    SIMPLE_ENV_OBS, DUEL_V1_BODY, DUEL_V2_WORLD,
    ObsBlock, ObsSpec,
)
from scripts._train_common import check_obs_compat, format_obs_block


def _write_info(tmp_path: Path, spec: obs_spec.ObsSpec, n_stack: int) -> Path:
    p = tmp_path / "run_info.toml"
    p.write_text("[run]\nname='x'\n" + format_obs_block(spec, n_stack))
    return p


def test_identical_specs_pass(tmp_path: Path):
    info = _write_info(tmp_path, DUEL_V1_BODY, n_stack=1)
    parent = check_obs_compat(info, current=DUEL_V1_BODY, current_n_stack=1, surgery=False)
    assert parent == (DUEL_V1_BODY, 1)


def test_added_block_refuses(tmp_path: Path, capsys):
    info = _write_info(tmp_path, SIMPLE_ENV_OBS, n_stack=1)
    with pytest.raises(SystemExit):
        check_obs_compat(info, current=DUEL_V1_BODY, current_n_stack=1, surgery=False)
    out = capsys.readouterr().out
    assert "❌" in out  # ❌ for added/removed
    assert "opp_pos_rel" in out


def test_removed_block_refuses(tmp_path: Path, capsys):
    info = _write_info(tmp_path, DUEL_V1_BODY, n_stack=1)
    with pytest.raises(SystemExit):
        check_obs_compat(info, current=SIMPLE_ENV_OBS, current_n_stack=1, surgery=False)
    out = capsys.readouterr().out
    assert "❌" in out
    assert "removed" in out


def test_frame_only_change_renders_warning(tmp_path: Path, capsys):
    info = _write_info(tmp_path, DUEL_V1_BODY, n_stack=1)
    # Construct a spec identical to DUEL_V1_BODY except opp_vel_rel uses world frame.
    cur = ObsSpec((ANG_VEL, ANG_POS, LIN_VEL_BODY, LIN_POS, UNIT_TO_GOAL,
                   SIGNED_DIST_NORM, OPP_POS_REL, OPP_VEL_REL_WORLD))
    with pytest.raises(SystemExit):
        check_obs_compat(info, current=cur, current_n_stack=1, surgery=False)
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "frame" in out  # the rendered explanation mentions "frame changed"


def test_notes_only_change_passes(tmp_path: Path, capsys):
    # Parent has SIMPLE_ENV_OBS with the canonical SIGNED_DIST_NORM note.
    info = _write_info(tmp_path, SIMPLE_ENV_OBS, n_stack=1)
    # Current spec: same shape but a different notes string on signed_dist_norm.
    alt = ObsBlock("signed_dist_norm", dim=1, notes="reworded")
    cur = ObsSpec(SIMPLE_ENV_OBS.blocks[:-1] + (alt,))
    parent = check_obs_compat(info, current=cur, current_n_stack=1, surgery=False)
    assert parent is not None   # load proceeds
    out = capsys.readouterr().out
    assert "Notes changed" in out
    assert "reworded" in out


def test_n_stack_mismatch_refuses(tmp_path: Path, capsys):
    info = _write_info(tmp_path, DUEL_V2_WORLD, n_stack=1)
    with pytest.raises(SystemExit):
        check_obs_compat(info, current=DUEL_V2_WORLD, current_n_stack=3, surgery=False)
    out = capsys.readouterr().out
    assert "n_stack" in out
    assert "❌" in out


def test_surgery_flag_allows_added_removed_and_frame(tmp_path: Path):
    info = _write_info(tmp_path, DUEL_V1_BODY, n_stack=1)
    parent = check_obs_compat(info, current=DUEL_V2_WORLD, current_n_stack=3, surgery=True)
    assert parent == (DUEL_V1_BODY, 1)


def test_parent_missing_obs_block_refuses_without_surgery(tmp_path: Path, capsys):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname='legacy'\n")
    with pytest.raises(SystemExit):
        check_obs_compat(info, current=DUEL_V2_WORLD, current_n_stack=3, surgery=False)
    out = capsys.readouterr().out
    assert "no [obs] block" in out.lower() or "missing" in out.lower()


def test_parent_missing_obs_block_passes_with_surgery(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname='legacy'\n")
    parent = check_obs_compat(info, current=DUEL_V2_WORLD, current_n_stack=3, surgery=True)
    # Returning (None, None) is the contract for "parent has no spec but surgery requested".
    assert parent == (None, None)


# ── New Hydra-based compat check (post-cutover) ─────────────────────────────


def test_check_obs_compat_from_hydra_strict_match(tmp_path: Path):
    from scripts.train import _check_obs_compat_from_hydra
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(
        "obs:\n  name: DUEL_V2_WORLD\n  n_stack: 3\n"
    )
    _check_obs_compat_from_hydra(hydra_dir, DUEL_V2_WORLD, current_n_stack=3)


def test_check_obs_compat_from_hydra_raises_on_mismatch(tmp_path: Path):
    from scripts.train import _check_obs_compat_from_hydra
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(
        "obs:\n  name: DUEL_V1_BODY\n  n_stack: 1\n"
    )
    with pytest.raises(SystemExit):
        _check_obs_compat_from_hydra(hydra_dir, DUEL_V2_WORLD, current_n_stack=3)
