"""End-to-end: train a tiny team-play model, save a checkpoint, resume from it,
verify num_timesteps continues and info.toml has a [resume] block."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

pytestmark = pytest.mark.slow


def _make_env_fn():
    def _thunk():
        team = QuidditchTeamEnv(cfg=TeamConfig(
            randomise_red_start=False, episode_seconds=2.0))
        return OpponentControlledEnv(team, learner_id="red_0",
                                     opponent=from_spec("beeline_blue"))
    return _thunk


def test_resume_continues_step_counter_and_writes_resume_block(tmp_path: Path) -> None:
    """Train ~256 steps, save checkpoint, resume, train ~256 more, assert
    num_timesteps continues and info.toml has [resume]."""
    # 1) Cold-train a tiny model and save a checkpoint.
    env = DummyVecEnv([_make_env_fn()])
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32,
                n_epochs=1, learning_rate=3e-4, verbose=0)
    model.learn(total_timesteps=128)
    cold_steps = int(model.num_timesteps)
    assert cold_steps >= 128

    parent_dir = tmp_path / "runs" / "team_resume_test" / "20260508_000000"
    (parent_dir / "checkpoints").mkdir(parents=True)
    ckpt = parent_dir / "checkpoints" / f"ppo_{cold_steps}_steps.zip"
    model.save(str(ckpt))

    # Write a parent info.toml so _read_parent_extra works.
    (parent_dir / "info.toml").write_text(
        '[run]\nname = "team_resume_test"\ntrial = "20260508_000000"\n'
        '\n[extra]\n'
        'learner = "red_0"\n'
        'opponent_spec = "beeline_blue"\n'
        'warm_start_from = ""\n'
    )

    # Snapshot a minimal config the resume run will read.
    cfg_path = tmp_path / "training.toml"
    cfg_path.write_text(
        '[training]\n'
        'run_name = "team_resume_test"\n'
        'total_timesteps = 256\n'  # = cold_steps + ~128 more
        'n_envs = 1\n'
        'seed = 42\n'
        '\n[training.ppo]\n'
        'n_steps = 64\nbatch_size = 32\nn_epochs = 1\n'
        'lr = 1.5e-4\n'  # different from cold-train's 3e-4 — verify override
        'gamma = 0.99\ngae_lambda = 0.95\nclip_range = 0.2\nent_coef = 0.01\n'
        '\n[training.eval]\n'
        'eval_freq_steps = 10000\nn_eval_episodes = 1\n'
        '\n[training.callbacks]\n'
        'checkpoint_freq_steps = 10000\nvideo_every_n_evals = 999\nvideo_fps = 20\n'
        '\n[env]\n'
        'randomise_start = false\nepisode_seconds = 2.0\nmode = "team"\n'
        '\n[env.team]\n'
        'red_prefix = "red_0"\nblue_prefix = "blue_0"\nhoop_prefix = "hoop_0"\n'
        'midpoint_alpha = 0.5\ntag_radius = 0.3\ntag_cooldown_s = 1.0\n'
        'crash_vel_thr = 1.5\nwalls_collide = true\n'
        '\n[training.team]\n'
        'learner = "red_0"\nopponent_spec = "beeline_blue"\n'
        'warm_start_from = ""\n'
        '\n[training.team.warm_start]\n'
        'new_dim_init_scale = 0.01\n'
    )

    # 2) Resume from the checkpoint via subprocess (matches real CLI usage).
    env_overrides = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    cwd = tmp_path  # resume writes new trial under tmp_path/runs/...
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train_team_ppo.py"),
         "--resume", str(ckpt),
         "--config", str(cfg_path)],
        cwd=str(cwd), env=env_overrides,
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"resume failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"

    # 3) Find the new trial dir (sibling of the parent under team_resume_test/).
    runs_root = cwd / "runs" / "team_resume_test"
    trials = sorted(p for p in runs_root.iterdir()
                    if p.is_dir() and p.name != "20260508_000000")
    assert len(trials) >= 1, f"no resume trial dir created under {runs_root}"
    resume_trial = trials[-1]

    # 4) Verify info.toml has the [resume] block.
    info = tomllib.loads((resume_trial / "info.toml").read_text())
    assert "resume" in info, f"missing [resume] block: {info.keys()}"
    assert info["resume"]["checkpoint"].endswith(ckpt.name)
    assert info["resume"]["resumed_at"] == cold_steps
    assert info["extra"]["learner"] == "red_0"
    assert info["extra"]["opponent_spec"] == "beeline_blue"

    # 5) Verify num_timesteps progressed past the checkpoint.
    final = PPO.load(str(resume_trial / "final_model.zip"))
    assert final.num_timesteps > cold_steps, (
        f"final num_timesteps {final.num_timesteps} should exceed "
        f"checkpoint's {cold_steps}")
    assert final.num_timesteps >= 256

    # 6) Verify lr override took effect — SB3 stores lr as a callable schedule.
    lr_at_end = float(final.learning_rate(1.0)) if callable(final.learning_rate) \
                else float(final.learning_rate)
    assert lr_at_end == pytest.approx(1.5e-4, rel=1e-6), (
        f"expected lr=1.5e-4 from config to override checkpoint's 3e-4, got {lr_at_end}")
