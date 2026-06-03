"""train.py's three init.parent load branches all route through resolve_parent."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest


def test_pretrain_branch_calls_resolve_parent(tmp_path: Path) -> None:
    """If init.mode=pretrain and parent is a wandb URI, resolve_parent
    is called before PPO.load."""
    from scripts.train import _build_or_load_model
    from omegaconf import OmegaConf

    duel_v2_blocks = [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
        "OPP_VEL_REL_WORLD", "CLOSING_RATE",
    ]
    cfg = OmegaConf.create({
        "trainer": {"n_steps": 1, "batch_size": 1, "n_epochs": 1, "lr": 1e-4,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                    "ent_coef": 0.01},
        "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3, "blocks": duel_v2_blocks},
        "init": {"mode": "pretrain", "parent": "wandb://ppo_hoop_blue_4:prod",
                 "new_dim_init_scale": 0.01},
    })

    fake_local = tmp_path / "models" / "ppo_hoop_blue_4" / "best_model.zip"
    fake_local.parent.mkdir(parents=True)
    fake_local.write_bytes(b"")

    fake_hydra = tmp_path / "models" / "ppo_hoop_blue_4" / ".hydra"
    fake_hydra.mkdir()
    blocks_yaml = "\n".join(f"    - {b}" for b in duel_v2_blocks)
    (fake_hydra / "config.yaml").write_text(
        "obs:\n  name: DUEL_V2_WORLD\n  n_stack: 3\n  blocks:\n" + blocks_yaml + "\n"
    )

    vec_env = MagicMock()

    with patch("scripts.train.resolve_parent", return_value=fake_local) as mock_resolve:
        with patch("scripts.train.PPO") as mock_ppo:
            mock_ppo.load.return_value = MagicMock(num_timesteps=0)
            with patch("scripts.train.read_parent_chain_total_from_hydra",
                       return_value=42):
                _build_or_load_model(cfg, vec_env, tmp_path, seed=42)

    mock_resolve.assert_called_once_with("wandb://ppo_hoop_blue_4:prod")
