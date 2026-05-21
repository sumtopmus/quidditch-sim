from pathlib import Path

from typer.testing import CliRunner

from dsim.cli import app


def test_list_runs_renders(tmp_path: Path) -> None:
    (tmp_path / "runs" / "blue_5" / "20260514_120000").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 0
    assert "blue_5" in result.output


def test_list_runs_filter(tmp_path: Path) -> None:
    (tmp_path / "runs" / "blue_5" / "20260514_120000").mkdir(parents=True)
    (tmp_path / "runs" / "red_2" / "20260514_120000").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path / "runs"),
                                 "--run", "blue"])
    assert result.exit_code == 0
    assert "blue_5" in result.output
    assert "red_2" not in result.output
