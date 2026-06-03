"""train.py's resume branch must not select its own freshly-created run_dir
as the parent trial.

When `init.parent_run == run_name` (the `dsim resume` default — it passes both
`init.parent_run={run}` and `run_name={run}`), Hydra has already created
`runs/<run>/<new_timestamp>/` before `_build_or_load_model` runs.  That dir is
the lex-newest, so a naive `max(run_root.iterdir())` picks the current run's own
empty checkpoints/ dir and fails to find any checkpoint.  The resume branch must
exclude run_dir and pick the real prior trial.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

DUEL_V2_BLOCKS = [
    "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
    "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
    "OPP_VEL_REL_WORLD", "CLOSING_RATE",
]


def _cfg(parent_run: str) -> OmegaConf:
    return OmegaConf.create({
        "trainer": {"n_steps": 1, "batch_size": 1, "n_epochs": 1, "lr": 1e-4,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                    "ent_coef": 0.01},
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3, "blocks": DUEL_V2_BLOCKS},
        "init": {"mode": "resume", "parent_run": parent_run,
                 "parent_checkpoint": None},
    })


def test_resume_excludes_current_run_dir_and_picks_highest_step(
    tmp_path: Path, monkeypatch,
) -> None:
    """parent_run == run_name: skip the current (empty) run_dir, load the prior
    trial's *highest-step* checkpoint (not the lexicographic-max name)."""
    from scripts.train import _build_or_load_model

    monkeypatch.chdir(tmp_path)
    run = "ppo_hoop_blue_5"

    # Real prior trial with two checkpoints whose lexicographic and numeric
    # orderings disagree: "..._1000000_..." sorts BEFORE "..._900000_...".
    prior = tmp_path / "runs" / run / "20260514_120000"
    (prior / "checkpoints").mkdir(parents=True)
    (prior / "checkpoints" / "ppo_hoop_900000_steps.zip").write_bytes(b"")
    (prior / "checkpoints" / "ppo_hoop_1000000_steps.zip").write_bytes(b"")

    # Current run dir: lex-newest timestamp, empty checkpoints/ (mirrors the
    # state Hydra leaves before _build_or_load_model runs under dsim resume).
    run_dir = tmp_path / "runs" / run / "20260514_180000"
    (run_dir / "checkpoints").mkdir(parents=True)

    vec_env = MagicMock()
    with patch("scripts.train._check_obs_compat_from_hydra"):
        with patch("scripts.train.PPO") as mock_ppo:
            mock_ppo.load.return_value = MagicMock(num_timesteps=1_000_000)
            _build_or_load_model(_cfg(run), vec_env, run_dir, seed=0)

    loaded = Path(mock_ppo.load.call_args.args[0])
    assert loaded.parent.parent == prior.resolve(), \
        "resume picked its own run_dir instead of the prior trial"
    assert "1000000" in loaded.name, \
        "resume used lexicographic sort instead of highest-step checkpoint"
