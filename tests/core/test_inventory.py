"""Behavior contract for core.inventory.

inventory() walks models/<name>/ (vendored) and optionally models/.cache/<name>/
(downloaded), reads each model's .hydra/config.yaml + .hydra/meta.yaml +
_wandb_metadata.json via core.run_context.load_run_context, and returns a
list[ModelInfo] sorted by short_name then trial timestamp descending.

Legacy migrated models (.hydra/config.yaml hand-written by
scripts/migrate_legacy_models.py) are handled the same way; their obs_spec
field may be missing or sparse, which inventory() reports as obs_spec="?"
(string sentinel, not None — keeps the table renderer simple).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _make_model_dir(
    root: Path,
    name: str,
    *,
    obs_name: str = "DUEL_V2_WORLD",
    n_stack: int = 3,
    parent: str | None = None,
    final_steps: int = 1_000_000,
    chain_total: int = 1_000_000,
    wandb_alias: str | None = "prod",
    wandb_version: str | None = "v0",
    has_model_doc: bool = False,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "best_model.zip").write_bytes(b"\x50\x4b\x03\x04stub-zip")
    h = d / ".hydra"
    h.mkdir()
    cfg: dict = {
        "run_name": name.rsplit("_", 2)[0],
        "obs": {"name": obs_name, "n_stack": n_stack},
        "init": (
            {"mode": "pretrain", "parent": parent} if parent is not None
            else {"mode": "scratch"}
        ),
    }
    OmegaConf.save(OmegaConf.create(cfg), h / "config.yaml")
    OmegaConf.save(OmegaConf.create({
        "final_steps": final_steps,
        "parent_chain_total": chain_total,
    }), h / "meta.yaml")
    if wandb_alias is not None:
        import json
        (d / "_wandb_metadata.json").write_text(json.dumps({
            "name": name.rsplit("_", 2)[0],
            "alias": wandb_alias,
            "version": wandb_version,
        }))
    if has_model_doc:
        (d / "MODEL.md").write_text("# MODEL: " + name + "\n")
    return d


def test_inventory_lists_vendored_models(tmp_path: Path) -> None:
    from core.inventory import inventory

    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir, "ppo_hoop_red_1_20260506_103058",
                    obs_name="DUEL_V1_BODY", n_stack=1)

    rows = inventory(models_dir=models_dir)

    names = [r.name for r in rows]
    assert "ppo_hoop_blue_4_20260511_202612" in names
    assert "ppo_hoop_red_1_20260506_103058" in names
    assert len(rows) == 2


def test_inventory_short_name_strips_timestamp(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    rows = inventory(models_dir=models_dir)
    assert rows[0].short_name == "blue_4"


def test_inventory_excludes_cache_by_default(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir / ".cache", "ppo_hoop_blue_5_v3")

    rows = inventory(models_dir=models_dir)
    assert all(r.source == "vendored" for r in rows)
    assert len(rows) == 1


def test_inventory_include_cache_picks_up_downloaded(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir / ".cache", "ppo_hoop_blue_5_v3")

    rows = inventory(models_dir=models_dir, include_cache=True)
    sources = {r.source for r in rows}
    assert sources == {"vendored", "cache"}


def test_inventory_carries_parent_chain_and_wandb_metadata(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_5_20260514_120000",
                    parent="wandb://ppo_hoop_blue_4:prod",
                    chain_total=15_000_000,
                    wandb_alias="prod", wandb_version="v3")
    rows = inventory(models_dir=models_dir)
    r = rows[0]
    assert r.parent == "wandb://ppo_hoop_blue_4:prod"
    assert r.parent_chain_total == 15_000_000
    assert r.wandb_alias == "prod"
    assert r.wandb_version == "v3"


def test_inventory_detects_model_doc_presence(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "with_doc", has_model_doc=True)
    _make_model_dir(models_dir, "without_doc", has_model_doc=False)
    rows = {r.name: r for r in inventory(models_dir=models_dir)}
    assert rows["with_doc"].has_model_doc is True
    assert rows["without_doc"].has_model_doc is False


def test_inventory_skips_dirs_without_hydra_config(tmp_path: Path) -> None:
    """A models/<x>/ dir with no .hydra/config.yaml is not a model — skip it."""
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "real_model")
    (models_dir / "junk").mkdir()
    (models_dir / "junk" / "README.txt").write_text("not a model")
    rows = inventory(models_dir=models_dir)
    assert [r.name for r in rows] == ["real_model"]


def test_inventory_sorted_by_short_name_then_timestamp_desc(tmp_path: Path) -> None:
    from core.inventory import inventory
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260501_000000")
    _make_model_dir(models_dir, "ppo_hoop_blue_4_20260511_202612")
    _make_model_dir(models_dir, "ppo_hoop_red_1_20260506_103058")
    rows = inventory(models_dir=models_dir)
    # blue_4 entries come first (sort by short_name), latest timestamp first.
    assert rows[0].name == "ppo_hoop_blue_4_20260511_202612"
    assert rows[1].name == "ppo_hoop_blue_4_20260501_000000"
    assert rows[2].name == "ppo_hoop_red_1_20260506_103058"


def test_load_model_doc_reads_when_present(tmp_path: Path) -> None:
    from core.inventory import inventory, load_model_doc
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_x", has_model_doc=True)
    [info] = inventory(models_dir=models_dir)
    assert load_model_doc(info) == "# MODEL: ppo_hoop_blue_4_x\n"


def test_load_model_doc_returns_none_when_absent(tmp_path: Path) -> None:
    from core.inventory import inventory, load_model_doc
    models_dir = tmp_path / "models"
    _make_model_dir(models_dir, "ppo_hoop_blue_4_x", has_model_doc=False)
    [info] = inventory(models_dir=models_dir)
    assert load_model_doc(info) is None
