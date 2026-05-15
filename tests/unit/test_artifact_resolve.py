"""resolve_parent maps `wandb://run:alias` URIs to local cache paths,
preferring committed `models/<run>/` when its `_wandb_metadata.json`
pins the same immutable version."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── URI parsing ──────────────────────────────────────────────────────────────
def test_parse_filesystem_path_returns_as_is(tmp_path: Path) -> None:
    from scripts._artifact_io import resolve_parent
    target = tmp_path / "models" / "x" / "best_model.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    out = resolve_parent(str(target))
    assert out == target


def test_parse_relative_path_returns_as_is() -> None:
    from scripts._artifact_io import resolve_parent
    out = resolve_parent("models/ppo_hoop_blue_4/best_model")
    assert str(out) == "models/ppo_hoop_blue_4/best_model"


def test_parse_wandb_shorthand_uri() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    qual = _parse_wandb_uri("wandb://ppo_hoop_blue_4:prod")
    assert qual.entity is None
    assert qual.project is None
    assert qual.name == "ppo_hoop_blue_4"
    assert qual.alias == "prod"


def test_parse_wandb_fully_qualified_uri() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    qual = _parse_wandb_uri("wandb-artifact://me/proj/ppo_hoop_blue_4:v3")
    assert qual.entity == "me"
    assert qual.project == "proj"
    assert qual.name == "ppo_hoop_blue_4"
    assert qual.alias == "v3"


def test_parse_uri_missing_alias_raises() -> None:
    from scripts._artifact_io import _parse_wandb_uri
    with pytest.raises(ValueError, match="missing alias"):
        _parse_wandb_uri("wandb://ppo_hoop_blue_4")


# ── Committed-cache hit ──────────────────────────────────────────────────────
def test_resolve_hits_committed_dir_when_version_matches(tmp_path: Path) -> None:
    """If models/<run>/_wandb_metadata.json pins v3 and alias :prod resolves
    to v3 on the API, return the committed path without download."""
    from scripts._artifact_io import resolve_parent

    # Set up a committed dir with metadata pinning v3.
    committed = tmp_path / "models" / "ppo_hoop_blue_4"
    committed.mkdir(parents=True)
    (committed / "best_model.zip").write_bytes(b"committed-bytes")
    (committed / "_wandb_metadata.json").write_text(json.dumps({
        "name": "ppo_hoop_blue_4",
        "version": "v3",
        "entity": "me",
        "project": "drone-quidditch",
    }))

    # Mock the wandb API: artifact for :prod resolves to v3.
    art = MagicMock()
    art.version = "v3"
    art.name = "ppo_hoop_blue_4:v3"
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_blue_4:prod", models_root=tmp_path / "models")

    assert out == committed / "best_model.zip"
    # use_artifact still recorded for lineage.
    run.use_artifact.assert_called_once()


def test_resolve_falls_through_when_committed_pins_different_version(tmp_path: Path) -> None:
    """If committed dir pins v2 but :prod points to v3, download v3."""
    from scripts._artifact_io import resolve_parent

    committed = tmp_path / "models" / "ppo_hoop_blue_4"
    committed.mkdir(parents=True)
    (committed / "_wandb_metadata.json").write_text(json.dumps({
        "name": "ppo_hoop_blue_4",
        "version": "v2",
    }))

    cache_root = tmp_path / "models" / ".cache"
    cache_dir = cache_root / "ppo_hoop_blue_4_v3"

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"downloaded-bytes")
        return root

    art = MagicMock()
    art.version = "v3"
    art.name = "ppo_hoop_blue_4:v3"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()
    run.use_artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_blue_4:prod", models_root=tmp_path / "models")

    assert out == cache_dir / "best_model.zip"
    assert out.read_bytes() == b"downloaded-bytes"


# ── No committed dir → download ──────────────────────────────────────────────
def test_resolve_downloads_when_no_committed_dir(tmp_path: Path) -> None:
    from scripts._artifact_io import resolve_parent

    cache_dir = tmp_path / "models" / ".cache" / "ppo_hoop_red_1_v0"

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"downloaded")
        return root

    art = MagicMock()
    art.version = "v0"
    art.name = "ppo_hoop_red_1:v0"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art
    run = MagicMock()
    run.use_artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", run):
            out = resolve_parent("wandb://ppo_hoop_red_1:prod", models_root=tmp_path / "models")

    assert out == cache_dir / "best_model.zip"


def test_resolve_no_active_wandb_run_still_works(tmp_path: Path) -> None:
    """When wandb.run is None (offline / disabled outside training), the
    resolver skips use_artifact but still downloads."""
    from scripts._artifact_io import resolve_parent

    def fake_download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "best_model.zip").write_bytes(b"x")
        return root

    art = MagicMock()
    art.version = "v0"
    art.download.side_effect = fake_download
    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", None):
            out = resolve_parent("wandb://ppo_hoop_red_1:prod", models_root=tmp_path / "models")

    assert (tmp_path / "models" / ".cache" / "ppo_hoop_red_1_v0" / "best_model.zip").exists()
    assert out.exists()
