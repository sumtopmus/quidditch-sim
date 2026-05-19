from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dsim.cli import app


def test_sweep_create_shells_to_wandb() -> None:
    runner = CliRunner()
    with patch("subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(app, ["sweep", "create", "ppo_lr"])
    assert result.exit_code == 0
    cmd = mock_call.call_args.args[0]
    assert cmd == ["wandb", "sweep", "sweeps/ppo_lr.yaml"]


def test_sweep_agent_shells_to_wandb() -> None:
    runner = CliRunner()
    with patch("subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(app, ["sweep", "agent", "abc123"])
    assert result.exit_code == 0
    cmd = mock_call.call_args.args[0]
    assert cmd == ["wandb", "agent", "abc123"]


def test_sweep_agents_spawns_n_parallel() -> None:
    runner = CliRunner()
    proc = MagicMock()
    proc.wait.return_value = 0
    with patch("subprocess.Popen", return_value=proc) as mock_popen:
        result = runner.invoke(app, ["sweep", "agents", "abc123", "--n", "3"])
    assert result.exit_code == 0
    assert mock_popen.call_count == 3
