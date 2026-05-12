"""Unit tests for scripts/backfill_obs_spec.py."""
from pathlib import Path

import pytest

from scripts.backfill_obs_spec import backfill_one, LEGACY_SPECS


def test_backfill_appends_obs_block_to_legacy_info(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text(
        "[run]\nname = \"ppo_hoop_blue_4\"\ntrial = \"20260511_202612\"\n"
    )
    # Use the canonical mapping for blue_4.
    backfill_one(info, run_dir_name="ppo_hoop_blue_4_20260511_202612")
    text = info.read_text()
    assert "[obs]" in text
    assert "n_stack = 3" in text
    assert "vec_to_hoop" in text
    assert "back-filled" in text


def test_backfill_refuses_to_overwrite(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text(
        "[run]\nname = \"foo\"\n\n[obs]\ndim = 16\nn_stack = 1\nslots = []\n"
    )
    with pytest.raises(RuntimeError, match="already has \\[obs\\]"):
        backfill_one(info, run_dir_name="ppo_hoop_blue_4_20260511_202612")


def test_backfill_unknown_run_dir_raises(tmp_path: Path):
    info = tmp_path / "run_info.toml"
    info.write_text("[run]\nname = \"foo\"\n")
    with pytest.raises(KeyError):
        backfill_one(info, run_dir_name="unknown_run_dir")


def test_legacy_specs_covers_all_seven_models():
    expected_prefixes = {
        "ppo_hoop_fixed_start_20260430_224234",
        "ppo_hoop_fixed_start_20260504_023051",
        "ppo_hoop_rand_start_20260430_234354",
        "ppo_hoop_rand_start_20260505_174509",
        "ppo_hoop_red_1_20260506_103058",
        "ppo_hoop_blue_1_20260507_194423",
        "ppo_hoop_blue_4_20260511_202612",
    }
    assert set(LEGACY_SPECS.keys()) == expected_prefixes
