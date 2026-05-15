"""upload_legacy_models builds the right wandb.Artifact + writes
_wandb_metadata.json pinning v0."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_legacy_dir(tmp_path: Path, name: str, obs_spec: str = "AUGMENTED_OBS") -> Path:
    d = tmp_path / "models" / name
    d.mkdir(parents=True)
    (d / "best_model.zip").write_bytes(b"model-bytes")
    hydra = d / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text(
        f"run_name: {name}\nobs:\n  name: {obs_spec}\n  n_stack: 3\n"
        f"init:\n  mode: scratch\n  parent: null\n"
    )
    (hydra / "meta.yaml").write_text(
        "git_hash: legacy\nparent_chain_total: 0\n"
        "final_stats:\n  completed_steps: 10000000\n  wall_time_s: 3600\n"
        "  best_eval_reward: null\n  peak_eval_step: null\n"
    )
    return d


def test_upload_creates_artifact_with_prod_and_run_name_aliases(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_red_1", obs_spec="TEAM_ENV_OBS")

    art = MagicMock()
    art.version = "v0"
    run = MagicMock()
    run.log_artifact.return_value = art

    with patch("wandb.Artifact", return_value=art) as mock_art_cls:
        with patch("wandb.init", return_value=run) as mock_init:
            upload_one(d)

    mock_art_cls.assert_called_once()
    name = mock_art_cls.call_args.kwargs.get("name") or mock_art_cls.call_args.args[0]
    assert name == "ppo_hoop_red_1"
    # The aliases list passed to log_artifact must include prod and the run name.
    aliases = run.log_artifact.call_args.kwargs.get("aliases")
    assert aliases is not None
    assert "prod" in aliases
    assert "ppo_hoop_red_1" in aliases


def test_upload_writes_pinned_metadata_in_place(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_blue_4", obs_spec="AUGMENTED_OBS")

    art = MagicMock()
    art.version = "v0"
    run = MagicMock()
    run.log_artifact.return_value = art

    with patch("wandb.Artifact", return_value=art):
        with patch("wandb.init", return_value=run):
            upload_one(d)

    meta = json.loads((d / "_wandb_metadata.json").read_text())
    assert meta["name"] == "ppo_hoop_blue_4"
    assert meta["version"] == "v0"
    assert "prod" in meta["aliases"]


def test_upload_idempotent_skips_if_metadata_present(tmp_path: Path) -> None:
    from scripts.upload_legacy_models import upload_one

    d = _make_legacy_dir(tmp_path, "ppo_hoop_red_1", obs_spec="TEAM_ENV_OBS")
    (d / "_wandb_metadata.json").write_text(json.dumps({"name": "ppo_hoop_red_1", "version": "v0"}))

    with patch("wandb.init") as mock_init:
        skipped = upload_one(d)

    assert skipped is True
    mock_init.assert_not_called()
