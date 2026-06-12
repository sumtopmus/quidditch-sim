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


def test_describe_run_surfaces_rllib_module_provenance(tmp_path):
    import json
    from typer.testing import CliRunner
    from dsim.cli import app

    model_dir = tmp_path / "models" / "rllib_league_step5"
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(
        "run_name: rllib_league_step5\nobs:\n  name: DUEL_V1_BODY\n  n_stack: 1\n")
    (model_dir / "_wandb_metadata.json").write_text(json.dumps({
        "name": "rllib_league_step5", "version": "v0",
        "checkpoint_format": "rllib",
        "module_ids": ["main_red", "main_blue", "blue_pop_v1"],
        "aliases": ["prod", "rllib_league_step5"],
    }))

    result = CliRunner().invoke(app, ["describe-run", str(model_dir)])
    assert result.exit_code == 0
    assert "rllib" in result.stdout
    assert "blue_pop_v1" in result.stdout
