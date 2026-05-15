"""scripts.promote: alias the artifact, copy best_model + .hydra into
models/<run_name>/, write pinned `_wandb_metadata.json`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_run_dir(tmp_path: Path, run_name: str) -> Path:
    """Build a fake completed run dir."""
    run_dir = tmp_path / "runs" / run_name / "20260514_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.zip").write_bytes(b"model-bytes")
    hydra = run_dir / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text(f"run_name: {run_name}\n")
    (hydra / "meta.yaml").write_text("git_hash: abc123\nparent_chain_total: 0\n")
    return run_dir


def test_promote_copies_into_models_dir(tmp_path: Path) -> None:
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_5")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v0"
    art.aliases = ["latest"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_blue_5",
                        models_root=models_root)

    dest = models_root / "ppo_hoop_blue_5"
    assert (dest / "best_model.zip").read_bytes() == b"model-bytes"
    assert (dest / ".hydra" / "config.yaml").exists()
    meta = json.loads((dest / "_wandb_metadata.json").read_text())
    assert meta["name"] == "ppo_hoop_blue_5"
    assert meta["version"] == "v0"             # IMMUTABLE, not the alias
    assert "prod" in meta["aliases"]
    assert "ppo_hoop_blue_5" in meta["aliases"]


def test_promote_adds_prod_and_run_name_aliases(tmp_path: Path) -> None:
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_red_2")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v3"
    art.aliases = ["latest", "v3"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_red_2",
                        models_root=models_root)

    # Verify the aliases list passed to art.save (the wandb API updates
    # aliases by mutating .aliases and calling .save()).
    assert "prod" in art.aliases
    assert "ppo_hoop_red_2" in art.aliases
    art.save.assert_called_once()


def test_promote_idempotent_alias_add(tmp_path: Path) -> None:
    """Adding `:prod` when it's already there should not duplicate."""
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_4")
    models_root = tmp_path / "models"

    art = MagicMock()
    art.version = "v2"
    art.aliases = ["latest", "v2", "prod", "ppo_hoop_blue_4"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_blue_4",
                        models_root=models_root)

    assert art.aliases.count("prod") == 1
    assert art.aliases.count("ppo_hoop_blue_4") == 1


def test_promote_no_best_model_fails(tmp_path: Path) -> None:
    import pytest
    from scripts.promote import promote_run_dir

    run_dir = tmp_path / "runs" / "x" / "20260514_120000"
    run_dir.mkdir(parents=True)
    # No best_model.zip.

    with pytest.raises(FileNotFoundError, match="best_model.zip"):
        promote_run_dir(run_dir=run_dir, run_name="x", models_root=tmp_path / "models")
