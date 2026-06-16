from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def test_resume_not_supported_on_rllib(tmp_path: Path) -> None:
    """resume fails fast on the RLlib path (Tuner.restore not yet wired)."""
    trial = tmp_path / "runs" / "blue_5" / "20260514_120000"
    (trial / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"run_name": "blue_5"}),
                   trial / ".hydra" / "config.yaml")
    # Synthetic hydra.yaml carrying the experiment choice.
    OmegaConf.save(OmegaConf.create({
        "hydra": {"runtime": {"choices": {"experiment": "blue_v5"}}}
    }), trial / ".hydra" / "hydra.yaml")
    (trial / "checkpoints").mkdir()
    (trial / "checkpoints" / "ppo_hoop_100000_steps.zip").write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(app, ["resume", "blue_5",
                                 "--runs-dir", str(tmp_path / "runs")])

    assert result.exit_code == 2
    assert "not yet supported on the RLlib path" in result.output


def test_resume_errors_when_experiment_unresolvable(tmp_path: Path) -> None:
    """If hydra.yaml doesn't carry an experiment choice and --exp isn't passed,
    exit 2 with a friendly error."""
    trial = tmp_path / "runs" / "x" / "20260514_120000"
    (trial / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"run_name": "x"}),
                   trial / ".hydra" / "config.yaml")
    # No hydra.yaml.
    (trial / "checkpoints").mkdir()
    (trial / "checkpoints" / "ppo_hoop_10000_steps.zip").write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(app, ["resume", "x", "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 2
