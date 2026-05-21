from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from dsim.cli import app


def test_lineage_local_renders_chain(tmp_path: Path) -> None:
    # Build a synthetic two-node chain: B -> A.
    a = tmp_path / "models" / "A"
    (a / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"run_name": "A", "init": {"mode": "scratch"}}),
                   a / ".hydra" / "config.yaml")
    b = tmp_path / "models" / "B"
    (b / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_name": "B",
        "init": {"mode": "pretrain", "parent": str(a)},
    }), b / ".hydra" / "config.yaml")

    runner = CliRunner()
    result = runner.invoke(app, ["lineage", "--target", str(b), "--local"])
    assert result.exit_code == 0
    assert "A" in result.output
    assert "B" in result.output
