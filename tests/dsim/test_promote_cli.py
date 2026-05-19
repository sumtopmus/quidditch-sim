from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def test_promote_invokes_core_promote(tmp_path: Path) -> None:
    """promote resolves the trial dir and calls core.promote.promote_run."""
    trial = tmp_path / "runs" / "blue_5" / "20260514_120000"
    (trial / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"run_name": "blue_5"}),
                   trial / ".hydra" / "config.yaml")
    (trial / "best_model.zip").write_bytes(b"")

    fake_result = MagicMock(
        run_name="blue_5", wandb_version="v0", wandb_alias="prod",
        target_dir=tmp_path / "models" / "blue_5",
        copied_files=["best_model.zip"],
    )

    runner = CliRunner()
    with patch("dsim.commands.promote.promote_run",
               return_value=fake_result) as mock_promote:
        result = runner.invoke(app, ["promote", "blue_5",
                                     "--runs-dir", str(tmp_path / "runs"),
                                     "--models-dir", str(tmp_path / "models")])

    assert result.exit_code == 0
    mock_promote.assert_called_once()
    # The trial dir gets passed as the first positional.
    call_args, call_kwargs = mock_promote.call_args
    assert call_kwargs["alias"] == "prod"
    assert "promoted blue_5" in result.output
