"""log_run_artifact picks the right source file and tags model_kind correctly.

Five behavioral cases:
  - best_model.zip present                       → logged as `best`
  - only final_model.zip present                 → logged as `final`,
                                                   renamed to best_model.zip
                                                   inside the artifact
  - neither file present                         → no-op
  - run is None                                  → no-op
  - run.disabled (WANDB_MODE=disabled stub)      → no-op
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _make_cfg() -> "OmegaConf":
    return OmegaConf.create({
        "run_name": "ppo_hoop_blue_5",
        "obs": {"name": "AUGMENTED_OBS", "n_stack": 3},
        "env": {"learner_id": "blue_0"},
        "init": {"mode": "pretrain", "parent": "wandb://ppo_hoop_blue_4:prod"},
    })


def _make_run_dir(tmp_path: Path, *, best: bool, final: bool) -> Path:
    d = tmp_path / "runs" / "ppo_hoop_blue_5" / "20260515_120000"
    d.mkdir(parents=True)
    if best:
        (d / "best_model.zip").write_bytes(b"best-bytes")
    if final:
        (d / "final_model.zip").write_bytes(b"final-bytes")
    hydra = d / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text("run_name: ppo_hoop_blue_5\n")
    return d


def test_logs_best_model_when_present(tmp_path: Path) -> None:
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=True, final=True)
    run = MagicMock()
    run.disabled = False

    art = MagicMock()
    with patch("wandb.Artifact", return_value=art) as mock_art_cls:
        log_run_artifact(run=run, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=10_000_000, best_eval_reward=8.42)

    # Metadata records model_kind=best.
    md = mock_art_cls.call_args.kwargs["metadata"]
    assert md["model_kind"] == "best"
    # add_file pointed at best_model.zip.
    add_calls = art.add_file.call_args_list
    assert len(add_calls) == 1
    src = add_calls[0].args[0]
    assert src.endswith("/best_model.zip")
    assert add_calls[0].kwargs["name"] == "best_model.zip"
    # Aliases default to latest.
    run.log_artifact.assert_called_once()
    aliases = run.log_artifact.call_args.kwargs.get("aliases")
    assert aliases == ["latest"]


def test_falls_back_to_final_when_best_absent(tmp_path: Path) -> None:
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=False, final=True)
    run = MagicMock()
    run.disabled = False

    art = MagicMock()
    with patch("wandb.Artifact", return_value=art) as mock_art_cls:
        log_run_artifact(run=run, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=0, best_eval_reward=None)

    md = mock_art_cls.call_args.kwargs["metadata"]
    assert md["model_kind"] == "final"
    assert md["best_eval_reward"] is None
    # Source path is final_model.zip but is renamed inside the artifact.
    add_call = art.add_file.call_args
    assert add_call.args[0].endswith("/final_model.zip")
    assert add_call.kwargs["name"] == "best_model.zip"
    run.log_artifact.assert_called_once()


def test_noop_when_neither_file_present(tmp_path: Path) -> None:
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=False, final=False)
    run = MagicMock()
    run.disabled = False

    with patch("wandb.Artifact") as mock_art_cls:
        log_run_artifact(run=run, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=0, best_eval_reward=None)

    mock_art_cls.assert_not_called()
    run.log_artifact.assert_not_called()


def test_noop_when_run_is_none(tmp_path: Path) -> None:
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=True, final=True)
    with patch("wandb.Artifact") as mock_art_cls:
        log_run_artifact(run=None, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=0, best_eval_reward=None)

    mock_art_cls.assert_not_called()


def test_noop_when_run_disabled(tmp_path: Path) -> None:
    """WANDB_MODE=disabled gives a stub with .disabled=True."""
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=True, final=True)
    run = MagicMock()
    run.disabled = True

    with patch("wandb.Artifact") as mock_art_cls:
        log_run_artifact(run=run, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=0, best_eval_reward=None)

    mock_art_cls.assert_not_called()
    run.log_artifact.assert_not_called()


def test_hydra_dir_added_when_present(tmp_path: Path) -> None:
    from scripts._artifact_io import log_run_artifact

    run_dir = _make_run_dir(tmp_path, best=True, final=False)
    run = MagicMock()
    run.disabled = False
    art = MagicMock()

    with patch("wandb.Artifact", return_value=art):
        log_run_artifact(run=run, run_dir=run_dir, cfg=_make_cfg(),
                         parent_chain_total=0, best_eval_reward=None)

    art.add_dir.assert_called_once()
    add_dir_args = art.add_dir.call_args
    assert add_dir_args.args[0].endswith("/.hydra")
    assert add_dir_args.kwargs["name"] == ".hydra"
