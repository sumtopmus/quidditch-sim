"""End-to-end smoke: scripts.train runs with WANDB_MODE=disabled.

Runs a 2048-step canary_team training to confirm the wandb-callback wiring
doesn't blow up.  No actual wandb network calls.  Eval skipped via
trainer.total_timesteps below first eval threshold.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_train_canary_team_2048_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=canary_team",
            "trainer.total_timesteps=2048",
            "trainer.n_steps=1024",
            "eval.eval_freq_steps=999999999",       # skip eval
            "eval.checkpoint_freq_steps=999999999", # skip checkpoint
            "eval.video.enabled=false",             # skip video
            f"hydra.run.dir={tmp_path}/smoke_run",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout)
        sys.stderr.write(out.stderr)
    assert out.returncode == 0, "scripts.train exited non-zero"
    assert (tmp_path / "smoke_run" / "final_model.zip").exists()
    assert (tmp_path / "smoke_run" / ".hydra" / "config.yaml").exists()
    assert (tmp_path / "smoke_run" / ".hydra" / "meta.yaml").exists()
    # Confirm no events.out.tfevents.* files (TB output retired).
    tb_files = list((tmp_path / "smoke_run").rglob("events.out.tfevents.*"))
    assert tb_files == [], f"unexpected TB event files: {tb_files}"
