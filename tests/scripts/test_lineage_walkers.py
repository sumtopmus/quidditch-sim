"""Lineage walker: local walks `_wandb_metadata.json`; wandb walks artifact DAG;
make lineage dispatches between them."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Walker A: local-only ─────────────────────────────────────────────────────
def test_walker_a_walks_metadata_chain(tmp_path: Path) -> None:
    from scripts.lineage import walk_chain_local

    # Build a two-link chain: red_v1 → rand_start.
    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "best_model.zip").write_bytes(b"")
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0", "parent_uri": "wandb://rand_start:prod",
    }))
    rand = tmp_path / "models" / "rand_start"
    rand.mkdir()
    (rand / "best_model.zip").write_bytes(b"")
    (rand / "_wandb_metadata.json").write_text(json.dumps({
        "name": "rand_start", "version": "v0", "parent_uri": None,
    }))

    chain = walk_chain_local(red, models_root=tmp_path / "models")

    assert len(chain) == 2
    # Oldest-first ordering.
    assert chain[0]["name"] == "rand_start"
    assert chain[1]["name"] == "red_v1"


def test_walker_a_truncates_on_missing_parent(tmp_path: Path) -> None:
    from scripts.lineage import walk_chain_local

    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0",
        "parent_uri": "wandb://nonexistent_parent:prod",
    }))

    chain = walk_chain_local(red, models_root=tmp_path / "models")

    # Walk truncated at red_v1; warning printed (not asserted here).
    assert len(chain) == 1
    assert chain[0]["name"] == "red_v1"


# ── Walker B: wandb API ──────────────────────────────────────────────────────
def test_walker_b_walks_artifact_dag() -> None:
    from scripts.lineage import walk_chain_wandb

    # Mock a two-link DAG: art_red -[used]-> art_rand.
    art_rand = MagicMock()
    art_rand.name = "rand_start:v0"
    art_rand.version = "v0"
    art_rand.metadata = {"obs_spec": "SIMPLE_ENV_OBS", "parent_chain_total": 20_000_000}
    art_rand.logged_by.return_value = MagicMock(used_artifacts=lambda: [])

    art_red = MagicMock()
    art_red.name = "red_v1:v0"
    art_red.version = "v0"
    art_red.metadata = {"obs_spec": "TEAM_ENV_OBS", "parent_chain_total": 30_000_000}
    red_run = MagicMock()
    red_run.used_artifacts.return_value = [art_rand]
    art_red.logged_by.return_value = red_run

    api = MagicMock()
    api.artifact.return_value = art_red

    with patch("wandb.Api", return_value=api):
        chain = walk_chain_wandb("wandb://red_v1:prod")

    assert len(chain) == 2
    assert chain[0]["name"].startswith("rand_start")
    assert chain[1]["name"].startswith("red_v1")


def test_walker_b_falls_back_to_local_on_api_error(tmp_path: Path) -> None:
    """When wandb.Api() raises, the dispatch falls back to walker A."""
    from scripts.lineage import walk_dispatch

    red = tmp_path / "models" / "red_v1"
    red.mkdir(parents=True)
    (red / "_wandb_metadata.json").write_text(json.dumps({
        "name": "red_v1", "version": "v0", "parent_uri": None,
    }))

    with patch("wandb.Api", side_effect=RuntimeError("network down")):
        chain = walk_dispatch("wandb://red_v1:prod",
                              models_root=tmp_path / "models",
                              prefer="wandb")

    # Local-fallback chain.
    assert len(chain) == 1
    assert chain[0]["name"] == "red_v1"


def test_walker_b_qualifies_uri_with_env_project_and_entity(monkeypatch) -> None:
    """Regression: walk_chain_wandb must qualify the URI with entity/project
    before calling api.artifact, otherwise wandb's default project takes
    over and lookups go to `uncategorized` instead of drone-quidditch."""
    from scripts.lineage import walk_chain_wandb

    monkeypatch.setenv("WANDB_ENTITY", "gridcom")
    monkeypatch.setenv("WANDB_PROJECT", "drone-quidditch")

    art = MagicMock()
    art.name = "ppo_hoop_blue_4:v3"
    art.version = "v3"
    art.metadata = {"obs_spec": "AUGMENTED_OBS"}
    art.logged_by.return_value = None        # leaf node, walk terminates

    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", None):       # offline / no live run
            chain = walk_chain_wandb("wandb://ppo_hoop_blue_4:prod")

    api.artifact.assert_called_once_with("gridcom/drone-quidditch/ppo_hoop_blue_4:prod")
    assert len(chain) == 1


# ── Resume collapse ──────────────────────────────────────────────────────────
def test_resume_chain_collapses_in_render(tmp_path: Path) -> None:
    """`init.mode=resume` segments collapse into one line in the rendered output."""
    from scripts.lineage import render

    chain = [
        {"name": "red_v1", "version": "v0", "init_mode": "scratch",
         "steps": 10_000_000, "obs_spec": "TEAM_ENV_OBS"},
        {"name": "red_v1", "version": "v1", "init_mode": "resume",
         "steps": 5_000_000, "obs_spec": "TEAM_ENV_OBS"},
    ]

    out = render(chain)

    assert "resumed" in out.lower()
    # Same name; not two separate red_v1 rows.
    assert out.count("red_v1") <= 2   # name might appear in header + summary
