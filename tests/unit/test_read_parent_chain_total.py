"""read_parent_chain_total walks a pretrain parent's info.toml so the
child's [pretrain].total_steps reflects the full ancestry.  Shared by
single-agent and team training scripts; this test pins the contract."""
from __future__ import annotations

from pathlib import Path


def _write_info(trial: Path, content: str) -> None:
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "info.toml").write_text(content)


def test_prefers_pretrain_total_when_present(tmp_path: Path) -> None:
    """If the parent itself had a [pretrain] block, its total_steps already
    reflects the chain — return that, not [run].steps_trained."""
    from scripts._train_common import read_parent_chain_total

    trial = tmp_path / "ppo_hoop_blue_4" / "20260511_010101"
    _write_info(
        trial,
        '[run]\n'
        'steps_trained = 20000000\n'
        '[pretrain]\n'
        'parent       = "models/blue_3/best_model"\n'
        'parent_steps = 10000000\n'
        'total_steps  = 30000000\n',
    )
    ckpt = trial / "best_model.zip"
    ckpt.write_bytes(b"")  # only path resolution matters

    assert read_parent_chain_total(str(ckpt)) == 30_000_000


def test_falls_back_to_steps_trained_when_no_pretrain(tmp_path: Path) -> None:
    """First link in a chain: parent has no [pretrain] block, only [run]."""
    from scripts._train_common import read_parent_chain_total

    trial = tmp_path / "ppo_hoop_blue_4" / "20260511_010101"
    _write_info(
        trial,
        '[run]\n'
        'steps_trained = 20000000\n',
    )
    ckpt = trial / "best_model.zip"
    ckpt.write_bytes(b"")

    assert read_parent_chain_total(str(ckpt)) == 20_000_000


def test_returns_none_when_no_info_file(tmp_path: Path) -> None:
    """Caller must tolerate missing info.toml — handcrafted /models drops
    without info often have just best_model.zip."""
    from scripts._train_common import read_parent_chain_total

    trial = tmp_path / "models" / "ad_hoc_drop"
    trial.mkdir(parents=True)
    ckpt = trial / "best_model.zip"
    ckpt.write_bytes(b"")

    assert read_parent_chain_total(str(ckpt)) is None


def test_reads_run_info_toml_variant(tmp_path: Path) -> None:
    """Promoted models live under models/<name>/ with the file renamed to
    run_info.toml (matches the existing dual-name handling)."""
    from scripts._train_common import read_parent_chain_total

    trial = tmp_path / "models" / "ppo_hoop_blue_v0"
    trial.mkdir(parents=True)
    (trial / "run_info.toml").write_text(
        '[run]\n'
        'steps_trained = 15000000\n',
    )
    ckpt = trial / "best_model.zip"
    ckpt.write_bytes(b"")

    assert read_parent_chain_total(str(ckpt)) == 15_000_000
