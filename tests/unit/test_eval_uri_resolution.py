"""eval_team's frozen: spec routes through resolve_parent so wandb:// URIs
are accepted alongside filesystem paths."""
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_frozen_filesystem_path_unchanged(tmp_path: Path) -> None:
    """Plain `frozen:models/foo/best_model` still works."""
    from envs.quidditch.opponents import from_spec
    fake_model = tmp_path / "fake.zip"
    fake_model.write_bytes(b"")

    with patch("envs.quidditch.opponents.PPO") as mock_ppo:
        mock_ppo.load.return_value = MagicMock()
        opp = from_spec(f"frozen:{fake_model}")

    call_args = mock_ppo.load.call_args
    # PPO.load is called with the path (positional or via path=); just confirm
    # the path passed in is the same string.
    args = call_args.args + tuple(call_args.kwargs.values())
    assert any(str(fake_model) in str(a) for a in args)


def test_frozen_wandb_uri_routes_through_resolve_parent(tmp_path: Path) -> None:
    from envs.quidditch.opponents import from_spec

    fake_resolved = tmp_path / "models" / ".cache" / "x_v0" / "best_model.zip"
    fake_resolved.parent.mkdir(parents=True)
    fake_resolved.write_bytes(b"")

    with patch("scripts._artifact_io.resolve_parent", return_value=fake_resolved) as mock_r:
        with patch("envs.quidditch.opponents.PPO") as mock_ppo:
            mock_ppo.load.return_value = MagicMock()
            from_spec("frozen:wandb://x:prod")

    mock_r.assert_called_once_with("wandb://x:prod")
