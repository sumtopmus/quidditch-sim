"""Table-driven tests: action.build_argv(form_values) → expected argv."""
from __future__ import annotations

import pytest

from tui.actions import ACTIONS


def _action(key: str):
    for a in ACTIONS:
        if a.key == key:
            return a
    raise KeyError(key)


def test_registry_has_16_actions() -> None:
    assert len(ACTIONS) == 16
    groups = {a.group for a in ACTIONS}
    assert groups == {"Demo", "Train", "Eval", "Manage"}


@pytest.mark.parametrize("key,values,expected", [
    ("hover", {}, ["mjpython", "demo/menu.py", "hover"]),
    ("waypoint", {}, ["mjpython", "demo/menu.py", "waypoint"]),
    ("takedown", {}, ["mjpython", "demo/menu.py", "takedown"]),
    ("score-through-tag", {}, ["mjpython", "demo/menu.py", "score-through-tag"]),
])
def test_demo_argvs(key, values, expected):
    assert _action(key).build_argv(values) == expected


def test_train_single_no_optional_args() -> None:
    a = _action("train")
    assert a.is_training is True
    argv = a.build_argv({"RUN_NAME": "", "PRETRAIN": ""})
    assert argv == ["python", "scripts/train_ppo.py"]


def test_train_single_with_run_name() -> None:
    a = _action("train")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop_x", "PRETRAIN": ""})
    assert argv == ["python", "scripts/train_ppo.py", "--run-name", "ppo_hoop_x"]


def test_train_single_with_pretrain() -> None:
    a = _action("train")
    argv = a.build_argv({"RUN_NAME": "", "PRETRAIN": "models/red_v1"})
    assert argv == ["python", "scripts/train_ppo.py",
                    "--pretrain", "models/red_v1/best_model"]


def test_train_team_red_with_warm_start() -> None:
    a = _action("train-team-red")
    argv = a.build_argv({"RUN_NAME": "phase2a", "WARM_START": "models/rs"})
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--learner", "red_0", "--opponent", "beeline_blue",
        "--run-name", "phase2a",
        "--warm-start", "models/rs/best_model",
    ]


def test_train_team_blue_requires_red() -> None:
    a = _action("train-team-blue")
    argv = a.build_argv({"RUN_NAME": "phase2b", "RED": "models/red_v1"})
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--learner", "blue_0",
        "--opponent", "frozen:models/red_v1/best_model",
        "--run-name", "phase2b",
    ]


def test_resume_argv() -> None:
    a = _action("resume")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260101_000000",
        "CHECKPOINT": "runs/ppo_hoop/20260101_000000/checkpoints/ppo_5000_steps",
    })
    assert argv == [
        "python", "scripts/train_ppo.py",
        "--run-name", "ppo_hoop",
        "--resume", "runs/ppo_hoop/20260101_000000/checkpoints/ppo_5000_steps",
    ]


def test_resume_team_argv() -> None:
    a = _action("resume-team")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop_blue_1",
        "TRIAL": "20260101_000000",
        "CHECKPOINT": "runs/ppo_hoop_blue_1/20260101_000000/checkpoints/ppo_10000_steps",
        "LEARNER": "", "OPPONENT": "",
    })
    assert argv == [
        "python", "scripts/train_team_ppo.py",
        "--resume", "runs/ppo_hoop_blue_1/20260101_000000/checkpoints/ppo_10000_steps",
        "--run-name", "ppo_hoop_blue_1",
    ]


def test_eval_gui_uses_mjpython() -> None:
    a = _action("eval")
    assert a.requires_mjpython is True  # the action declares it; runner picks mjpython
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260202_000000",
        "EPISODES": 10,
        "GUI": True,
    })
    assert argv == [
        "mjpython", "scripts/eval_ppo.py",
        "--model", "runs/ppo_hoop/20260202_000000/best_model",
        "--episodes", "10",
    ]


def test_eval_headless_uses_python_and_no_render() -> None:
    a = _action("eval")
    argv = a.build_argv({
        "RUN_NAME": "ppo_hoop",
        "TRIAL": "20260202_000000",
        "EPISODES": 50,
        "GUI": False,
    })
    assert argv == [
        "python", "scripts/eval_ppo.py",
        "--model", "runs/ppo_hoop/20260202_000000/best_model",
        "--episodes", "50",
        "--no-render",
    ]


def test_eval_team_with_flags() -> None:
    a = _action("eval-team")
    argv = a.build_argv({
        "RED": "models/red_v1", "BLUE": "models/blue_v1",
        "EPISODES": 5, "GUI": True, "DETERMINISTIC": True,
    })
    assert argv == [
        "mjpython", "scripts/eval_team.py",
        "--red", "models/red_v1", "--blue", "models/blue_v1",
        "--episodes", "5",
        "--gui",
        "--deterministic",
    ]


def test_tensorboard_all_runs() -> None:
    a = _action("tensorboard")
    assert a.is_tensorboard is True
    argv = a.build_argv({"RUN_NAME": ""})
    assert argv == ["tensorboard", "--logdir", "runs"]


def test_tensorboard_single_run() -> None:
    a = _action("tensorboard")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop"})
    assert argv == ["tensorboard", "--logdir", "runs/ppo_hoop"]


def test_promote_argv() -> None:
    a = _action("promote")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop", "TRIAL": "20260101_000000"})
    assert argv == [
        "python", "scripts/promote.py",
        "--trial", "runs/ppo_hoop/20260101_000000",
    ]


def test_list_runs_argv() -> None:
    a = _action("list-runs")
    assert a.build_argv({}) == ["python", "scripts/list_runs.py"]


def test_lineage_argv() -> None:
    a = _action("lineage")
    argv = a.build_argv({"RUN_NAME": "ppo_hoop", "TRIAL": "20260101_000000"})
    assert argv == ["python", "scripts/lineage.py", "runs/ppo_hoop/20260101_000000"]


def test_repro_argv() -> None:
    a = _action("repro")
    argv = a.build_argv({"MODEL": "models/red_v1"})
    assert argv == ["python", "scripts/repro.py", "--model-dir", "models/red_v1"]
