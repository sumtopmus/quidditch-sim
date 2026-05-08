"""write_run_info emits a [resume] block when resume= is given."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import tomllib


def test_resume_block_is_emitted(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "ppo_hoop_blue_1" / "20260508_010101"
    run_dir.mkdir(parents=True)

    args = argparse.Namespace(run_name="ppo_hoop_blue_1", config=None)
    started = datetime(2026, 5, 8, 1, 1, 1)

    write_run_info(
        run_dir,
        config={},
        args=args,
        extra={"learner": "blue_0",
               "opponent_spec": "frozen:models/red/best_model"},
        resume={"checkpoint": "runs/ppo_hoop_blue_1/20260507_194423/"
                              "checkpoints/ppo_10000000_steps.zip",
                "resumed_at": 10002432},
        started=started,
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert info["resume"]["checkpoint"].endswith("ppo_10000000_steps.zip")
    assert info["resume"]["resumed_at"] == 10002432
    # extra block still present
    assert info["extra"]["learner"] == "blue_0"


def test_no_resume_block_when_not_given(tmp_path: Path) -> None:
    from scripts._train_common import write_run_info

    run_dir = tmp_path / "ppo_hoop_red_1" / "20260508_010101"
    run_dir.mkdir(parents=True)

    args = argparse.Namespace(run_name="ppo_hoop_red_1", config=None)

    write_run_info(
        run_dir, config={}, args=args,
        extra={"learner": "red_0", "opponent_spec": "beeline_blue"},
        started=datetime(2026, 5, 8, 1, 1, 1),
    )

    info = tomllib.loads((run_dir / "info.toml").read_text())
    assert "resume" not in info
    assert info["extra"]["learner"] == "red_0"
