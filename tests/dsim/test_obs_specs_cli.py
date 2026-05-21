from typer.testing import CliRunner

from dsim.cli import app


def test_obs_specs_lists_known_specs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["obs-specs"])
    assert result.exit_code == 0
    # At least three known specs should appear.
    assert "DUEL_V1_BODY" in result.output
    assert "DUEL_V2_WORLD" in result.output
    assert "DUEL_V3_BODY_EGO" in result.output
