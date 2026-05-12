"""write_run_info emits a [pretrain] block when pretrain= is given.
scripts/lineage.py reads `[pretrain].parent` and recurses; this test pins
the field names so lineage walking can't silently break."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import tomllib


def test_pretrain_block_is_emitted(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "ppo_hoop_blue_5" / "20260512_010101"
    run_dir.mkdir(parents=True)

    args = argparse.Namespace(run_name="ppo_hoop_blue_5", config=None)

    write_run_info(
        run_dir,
        config={},
        args=args,
        extra={"learner": "blue_0", "opponent_spec": "beeline_red"},
        pretrain={
            "parent": "models/ppo_hoop_blue_4/best_model",
            "parent_steps": 20_000_000,
            "total_steps": 30_000_000,  # parent's chain (20M) + this run's contribution (10M)
        },
        started=datetime(2026, 5, 12, 1, 1, 1),
        steps_trained=10_000_000,
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert info["pretrain"]["parent"]       == "models/ppo_hoop_blue_4/best_model"
    assert info["pretrain"]["parent_steps"] == 20_000_000
    assert info["pretrain"]["total_steps"]  == 30_000_000
    # Existing blocks still present and not corrupted.
    assert info["extra"]["learner"] == "blue_0"
    assert info["run"]["steps_trained"] == 10_000_000


def test_pretrain_total_in_progress_when_none(tmp_path: Path) -> None:
    """Before training completes, total_steps is unknown — write_run_info
    emits a sentinel string instead of an int so the file is still valid TOML."""
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "r" / "t"
    run_dir.mkdir(parents=True)

    write_run_info(
        run_dir,
        config={},
        args=argparse.Namespace(run_name="r", config=None),
        pretrain={
            "parent": "models/parent/best_model",
            "parent_steps": 5_000_000,
            "total_steps": None,    # filled in after training
        },
        started=datetime(2026, 5, 12, 1, 1, 1),
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    # Total is written as the sentinel string when None.
    assert info["pretrain"]["total_steps"] == "in progress"


def test_no_pretrain_block_when_not_given(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "r" / "t"
    run_dir.mkdir(parents=True)

    write_run_info(
        run_dir,
        config={},
        args=argparse.Namespace(run_name="r", config=None),
        extra={"learner": "blue_0", "opponent_spec": "beeline_red"},
        started=datetime(2026, 5, 12, 1, 1, 1),
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert "pretrain" not in info
