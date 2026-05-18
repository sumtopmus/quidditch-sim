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

import pytest

pytestmark = pytest.mark.slow


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
    # MODEL.md is auto-generated at train end (best-effort).
    model_md = tmp_path / "smoke_run" / "MODEL.md"
    assert model_md.exists(), "MODEL.md should be auto-generated at train end"
    text = model_md.read_text()
    assert "# MODEL:" in text
    assert "## Summary" in text
    # Confirm no events.out.tfevents.* files (TB output retired).
    tb_files = list((tmp_path / "smoke_run").rglob("events.out.tfevents.*"))
    assert tb_files == [], f"unexpected TB event files: {tb_files}"


def test_train_smoke_calls_log_run_artifact(tmp_path: Path) -> None:
    """The end-of-run artifact log fires (and silently no-ops in disabled mode)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}

    # Use an offline-mode side process; we can't easily intercept the in-process
    # call from a subprocess, so this test confirms only that the smoke doesn't
    # fail when log_run_artifact is wired in.  (Mocked unit tests cover the
    # actual artifact-construction behavior.)
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=canary_team",
            "trainer.total_timesteps=2048",
            "trainer.n_steps=1024",
            "eval.eval_freq_steps=999999999",
            "eval.checkpoint_freq_steps=999999999",
            "eval.video.enabled=false",
            f"hydra.run.dir={tmp_path}/smoke_artifact",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0


def test_train_blue_v7_smoke(tmp_path: Path) -> None:
    """Smoke: blue_v7 experiment (DUEL_V3_BODY_EGO + InterceptShaping)
    composes via Hydra and runs a tiny PPO loop end-to-end.

    Triggers the eval callback at least once (eval_freq_steps below the
    first rollout boundary) so the eval-env-fn obs shape mismatch can't
    slip past again — see 2026-05-18 fix where the inline eval_env_fn
    in scripts/train.py was missing learner_id + learner_spec and
    produced 22-d obs (× n_stack=3 = 66-d) instead of the model's
    expected 75-d, crashing EvalCallback's first predict().
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "WANDB_MODE": "disabled"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "+experiment=blue_v7",
            "trainer.total_timesteps=512",
            "trainer.n_steps=64",
            "trainer.batch_size=64",
            "env.n_envs=1",
            # Trigger eval after the first 64-step rollout: with n_envs=1
            # EvalCallback's eval_freq is just env_steps, and SB3 also
            # invokes it on training start.  64 is enough that the eval
            # actually fires within total_timesteps=512.
            "eval.eval_freq_steps=64",
            "eval.n_eval_episodes=1",
            # Short episode so the eval rollout doesn't dwarf training.
            "curriculum.episode_seconds=2.0",
            "eval.checkpoint_freq_steps=999999999",
            "eval.video.enabled=false",
            f"hydra.run.dir={tmp_path}/blue_v7_smoke",
        ],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0, "scripts.train blue_v7 exited non-zero"
    assert (tmp_path / "blue_v7_smoke" / "final_model.zip").exists()
    assert (tmp_path / "blue_v7_smoke" / ".hydra" / "config.yaml").exists()
