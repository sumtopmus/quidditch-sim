"""End-to-end smoke: the blue_oracle_v1 CTDE experiment trains a few steps.

Runs a tiny PPO loop over the dual-view Dict obs (AsymmetricActorCriticPolicy)
against beeline_red, WANDB disabled.  eval_freq is set below the first rollout
boundary so the Dict-obs EvalCallback path is actually exercised — the CTDE
analogue of the 2026-05-18 eval-env obs-shape regression that blue_v7's smoke
was written to catch.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_ctde_experiment_trains_a_few_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=blue_oracle_v1",
            "trainer.total_timesteps=512",
            "trainer.n_steps=128",
            "trainer.batch_size=64",
            "env.n_envs=1",
            # Trigger at least one eval boundary: with n_envs=1 EvalCallback's
            # eval_freq is just env steps, so 128 fires within 512 timesteps —
            # exercising the Dict-obs eval env + actor-only predict.
            "eval.eval_freq_steps=128",
            "eval.n_eval_episodes=1",
            "curriculum.episode_seconds=2.0",
            "eval.checkpoint_freq_steps=999999999",
            "eval.video.enabled=false",
            f"hydra.run.dir={tmp_path}/ctde_smoke",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout)
        sys.stderr.write(out.stderr)
    assert out.returncode == 0, "scripts.train blue_oracle_v1 exited non-zero"
    assert (tmp_path / "ctde_smoke" / "final_model.zip").exists()
    assert (tmp_path / "ctde_smoke" / ".hydra" / "config.yaml").exists()
