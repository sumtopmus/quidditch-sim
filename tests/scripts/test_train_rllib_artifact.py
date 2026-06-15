"""Step-5c: train_rllib logs the RLlib checkpoint dir as a :latest artifact."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _cfg() -> "OmegaConf":
    return OmegaConf.create({
        "run_name": "rllib_league_step5",
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1},
        "env": {"learner_id": "red_0"},
        "init": {"mode": "scratch", "parent": None},
        "tune": {"wandb": {"enabled": True, "project": "drone-quidditch"}},
    })


def test_log_best_checkpoint_logs_latest_artifact(tmp_path: Path) -> None:
    from scripts.train_rllib import _log_best_checkpoint

    run_dir = tmp_path / "runs" / "rllib_league_step5" / "20260611_130631"
    ckpt = run_dir / "tune" / "trial_x" / "checkpoint_000030"
    (ckpt / "learner_group" / "learner" / "rl_module" / "main_blue").mkdir(parents=True)
    (run_dir / ".hydra").mkdir(parents=True)

    fake_run = MagicMock(); fake_run.disabled = False
    with patch("scripts.train_rllib.wandb") as wb, \
         patch("scripts.train_rllib.log_rllib_run_artifact") as log_art:
        wb.init.return_value = fake_run
        _log_best_checkpoint(_cfg(), run_dir)

    log_art.assert_called_once()
    kw = log_art.call_args.kwargs
    assert Path(kw["checkpoint_dir"]).name == "checkpoint_000030"
    fake_run.finish.assert_called_once()


def test_log_best_checkpoint_noop_when_wandb_disabled(tmp_path: Path) -> None:
    from scripts.train_rllib import _log_best_checkpoint

    run_dir = tmp_path / "runs" / "x" / "20260611_130631"
    (run_dir / "tune" / "trial_x" / "checkpoint_000010"
     / "learner_group" / "learner" / "rl_module" / "main_red").mkdir(parents=True)
    cfg = _cfg(); cfg.tune.wandb.enabled = False
    with patch("scripts.train_rllib.log_rllib_run_artifact") as log_art:
        _log_best_checkpoint(cfg, run_dir)
    log_art.assert_not_called()
