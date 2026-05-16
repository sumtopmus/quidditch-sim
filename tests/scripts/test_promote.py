"""scripts.promote: alias the artifact, copy best_model + .hydra into
models/<run_name>/, write pinned `_wandb_metadata.json`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_run_dir(
    tmp_path: Path,
    run_name: str,
    *,
    wandb_project: str = "drone-quidditch",
    entity_override: str | None = None,
) -> Path:
    """Build a fake completed run dir.  Includes a wandb config block
    matching what a real Hydra-composed run writes."""
    run_dir = tmp_path / "runs" / run_name / "20260514_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.zip").write_bytes(b"model-bytes")
    hydra = run_dir / ".hydra"
    hydra.mkdir()
    wandb_yaml = f"  project: {wandb_project}\n"
    if entity_override is not None:
        wandb_yaml += f"  entity_override: {entity_override}\n"
    (hydra / "config.yaml").write_text(
        f"run_name: {run_name}\nwandb:\n{wandb_yaml}"
    )
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


def test_promote_qualifies_artifact_path_with_project_and_entity(
    tmp_path: Path, monkeypatch,
) -> None:
    """Regression: a bare `name:latest` lookup falls through to wandb's
    workspace-default project (often `uncategorized`) and fails to find
    artifacts logged to cfg.wandb.project.  We must qualify with
    entity/project."""
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_5",
                             wandb_project="drone-quidditch")
    models_root = tmp_path / "models"

    monkeypatch.setenv("WANDB_ENTITY", "gridcom")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    art = MagicMock()
    art.version = "v0"
    art.aliases = ["latest"]
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        promote_run_dir(run_dir=run_dir, run_name="ppo_hoop_blue_5",
                        models_root=models_root)

    # Single api.artifact call with fully-qualified path.
    api.artifact.assert_called_once_with(
        "gridcom/drone-quidditch/ppo_hoop_blue_5:latest"
    )


def test_promote_falls_back_to_drone_quidditch_when_cfg_lacks_wandb_block(
    tmp_path: Path, monkeypatch,
) -> None:
    """Old runs (pre-Part-2) may not have a wandb block in .hydra/config.yaml.
    The resolver should still produce a usable project (default fallback)."""
    from scripts.promote import _resolve_entity_project

    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    hydra = run_dir / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text("run_name: legacy\n")

    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    entity, project = _resolve_entity_project(run_dir)
    assert project == "drone-quidditch"
    assert entity is None


def test_promote_env_overrides_take_precedence_over_cfg(
    tmp_path: Path, monkeypatch,
) -> None:
    from scripts.promote import _resolve_entity_project

    run_dir = _make_run_dir(tmp_path, "x",
                             wandb_project="drone-quidditch",
                             entity_override="team-a")

    monkeypatch.setenv("WANDB_PROJECT", "scratch")
    monkeypatch.setenv("WANDB_ENTITY", "team-b")

    entity, project = _resolve_entity_project(run_dir)
    assert project == "scratch"
    assert entity == "team-b"


def test_promote_path_without_entity_when_only_project_set(
    tmp_path: Path, monkeypatch,
) -> None:
    """If WANDB_ENTITY isn't set and cfg has no entity_override, the
    qualified path uses the two-part form (project/name:alias)."""
    from scripts.promote import _find_run_artifact

    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    api = MagicMock()
    with patch("wandb.Api", return_value=api):
        _find_run_artifact("foo", "20260101_000000",
                            entity=None, project="drone-quidditch")
    api.artifact.assert_called_once_with("drone-quidditch/foo:latest")


def test_promote_copies_model_doc_when_present(tmp_path: Path) -> None:
    """When <run_dir>/MODEL.md exists, promote copies it to models/<name>/MODEL.md."""
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_5")
    (run_dir / "MODEL.md").write_text("# MODEL: ppo_hoop_blue_5_20260514_120000\n")
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
    assert (dest / "MODEL.md").exists()
    assert "ppo_hoop_blue_5" in (dest / "MODEL.md").read_text()


def test_promote_skips_model_doc_when_absent(tmp_path: Path) -> None:
    """When the run dir lacks MODEL.md, promote completes without copying it."""
    from scripts.promote import promote_run_dir

    run_dir = _make_run_dir(tmp_path, "ppo_hoop_blue_5")
    # No MODEL.md created.
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
    assert (dest / "best_model.zip").exists()
    assert not (dest / "MODEL.md").exists()
