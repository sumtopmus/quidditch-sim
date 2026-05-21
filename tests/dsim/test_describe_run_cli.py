from pathlib import Path

from typer.testing import CliRunner

from dsim.cli import app


def test_describe_run_prints_existing_model_md(tmp_path: Path) -> None:
    md = tmp_path / "models" / "ppo_hoop_blue_4" / "MODEL.md"
    md.parent.mkdir(parents=True)
    md.write_text("# MODEL: ppo_hoop_blue_4\n\nstub")
    runner = CliRunner()
    result = runner.invoke(app, ["describe-run", "ppo_hoop_blue_4",
                                 "--models-dir", str(tmp_path / "models")])
    assert result.exit_code == 0
    assert "ppo_hoop_blue_4" in result.output


def test_describe_run_errors_without_model_md(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["describe-run", "nope",
                                 "--models-dir", str(tmp_path / "models"),
                                 "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 2
    assert "render_model_doc" in result.output or "no such run" in result.output
