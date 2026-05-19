"""Behavior contract for core.lineage walkers.

walk_chain_local(start_path) -> list[LineageNode]
  Reads .hydra/config.yaml `init.parent` chain in models/<run>/
  (offline-survivable).

walk_chain_wandb(target_uri) -> list[LineageNode]
  Uses wandb.Api().artifact().logged_by().used_artifacts().  Network required.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _make_chain(tmp_path: Path, names: list[str], parents: list[str | None]) -> None:
    """Build a synthetic model chain where models[i]'s parent points to models[i-1]."""
    models = tmp_path / "models"
    for name, parent in zip(names, parents):
        d = models / name
        (d / ".hydra").mkdir(parents=True)
        cfg: dict = {"run_name": name, "obs": {"name": "DUEL_V2_WORLD", "n_stack": 3}}
        if parent is not None:
            cfg["init"] = {"mode": "pretrain", "parent": parent}
        else:
            cfg["init"] = {"mode": "scratch"}
        OmegaConf.save(OmegaConf.create(cfg), d / ".hydra" / "config.yaml")
        OmegaConf.save(OmegaConf.create({"final_steps": 1_000_000}),
                       d / ".hydra" / "meta.yaml")


def test_walk_chain_local_walks_back_via_parent_field(tmp_path: Path) -> None:
    from core.lineage import walk_chain_local
    _make_chain(
        tmp_path,
        names=["A", "B", "C"],
        parents=[None, str(tmp_path / "models" / "A"), str(tmp_path / "models" / "B")],
    )
    chain = walk_chain_local(tmp_path / "models" / "C")
    assert [n.name for n in chain] == ["C", "B", "A"]


def test_walk_chain_local_stops_at_scratch(tmp_path: Path) -> None:
    from core.lineage import walk_chain_local
    _make_chain(tmp_path, names=["A", "B"], parents=[None, str(tmp_path / "models" / "A")])
    chain = walk_chain_local(tmp_path / "models" / "B")
    assert chain[-1].parent is None
    assert chain[-1].name == "A"


def test_walk_chain_local_handles_missing_intermediate(tmp_path: Path) -> None:
    """If a parent path doesn't exist, the chain truncates with a sentinel."""
    from core.lineage import walk_chain_local
    _make_chain(
        tmp_path,
        names=["B"],
        parents=[str(tmp_path / "models" / "VANISHED_PARENT")],
    )
    chain = walk_chain_local(tmp_path / "models" / "B")
    assert chain[0].name == "B"
    assert chain[-1].truncated is True


# ── Walker B: wandb API ──────────────────────────────────────────────────────
def test_walk_chain_wandb_walks_artifact_dag(monkeypatch) -> None:
    """The wandb walker hops via art.logged_by().used_artifacts() and
    yields oldest-first LineageNode rows."""
    from unittest.mock import MagicMock, patch
    from core.lineage import walk_chain_wandb

    monkeypatch.setenv("WANDB_ENTITY", "gridcom")
    monkeypatch.setenv("WANDB_PROJECT", "drone-quidditch")

    art_rand = MagicMock()
    art_rand.name = "rand_start:v0"
    art_rand.version = "v0"
    art_rand.metadata = {"obs_spec": "SIMPLE_ENV_OBS", "parent_chain_total": 20_000_000}
    art_rand.logged_by.return_value = MagicMock(used_artifacts=lambda: [])

    art_red = MagicMock()
    art_red.name = "red_v1:v0"
    art_red.version = "v0"
    art_red.metadata = {"obs_spec": "DUEL_V1_BODY", "parent_chain_total": 30_000_000,
                        "init_mode": "pretrain", "parent_uri": "wandb://rand_start:prod"}
    red_run = MagicMock()
    red_run.used_artifacts.return_value = [art_rand]
    art_red.logged_by.return_value = red_run

    api = MagicMock()
    api.artifact.return_value = art_red

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", None):
            chain = walk_chain_wandb("wandb://red_v1:prod")

    assert len(chain) == 2
    assert chain[0].name == "rand_start"
    assert chain[1].name == "red_v1"
    assert chain[1].init_mode == "pretrain"
    assert chain[1].obs_spec == "DUEL_V1_BODY"


def test_walk_chain_wandb_qualifies_uri_with_env(monkeypatch) -> None:
    """walk_chain_wandb must qualify the URI with entity/project before hitting
    api.artifact, otherwise wandb's default project takes over."""
    from unittest.mock import MagicMock, patch
    from core.lineage import walk_chain_wandb

    monkeypatch.setenv("WANDB_ENTITY", "gridcom")
    monkeypatch.setenv("WANDB_PROJECT", "drone-quidditch")

    art = MagicMock()
    art.name = "ppo_hoop_blue_4:v3"
    art.version = "v3"
    art.metadata = {"obs_spec": "DUEL_V2_WORLD"}
    art.logged_by.return_value = None

    api = MagicMock()
    api.artifact.return_value = art

    with patch("wandb.Api", return_value=api):
        with patch("wandb.run", None):
            chain = walk_chain_wandb("wandb://ppo_hoop_blue_4:prod")

    api.artifact.assert_called_once_with("gridcom/drone-quidditch/ppo_hoop_blue_4:prod")
    assert len(chain) == 1
