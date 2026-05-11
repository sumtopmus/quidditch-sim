"""Training-slot actions: train (single), train-team-red, train-team-blue, resume, resume-team."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _train_single_argv(v: dict) -> list[str]:
    out = ["python", "scripts/train_ppo.py"]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    if v.get("PRETRAIN"):
        out += ["--pretrain", f"{v['PRETRAIN']}/best_model"]
    return out


def _train_team_red_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--learner", "red_0", "--opponent", "beeline_blue",
    ]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    if v.get("WARM_START"):
        out += ["--warm-start", f"{v['WARM_START']}/best_model"]
    return out


def _train_team_blue_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--learner", "blue_0",
        "--opponent", f"frozen:{v['RED']}/best_model",
    ]
    if v.get("RUN_NAME"):
        out += ["--run-name", v["RUN_NAME"]]
    return out


def _resume_argv(v: dict) -> list[str]:
    return [
        "python", "scripts/train_ppo.py",
        "--run-name", v["RUN_NAME"],
        "--resume", v["CHECKPOINT"],
    ]


def _resume_team_argv(v: dict) -> list[str]:
    out = [
        "python", "scripts/train_team_ppo.py",
        "--resume", v["CHECKPOINT"],
        "--run-name", v["RUN_NAME"],
    ]
    if v.get("LEARNER"):
        out += ["--learner", v["LEARNER"]]
    if v.get("OPPONENT"):
        out += ["--opponent", v["OPPONENT"]]
    return out


TRAIN = [
    Action(
        key="train",
        label="train single-agent",
        group="Train",
        glyph_attr="TRAIN",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="PRETRAIN", label="Pretrain from", kind=FieldKind.PICKER_MODELS),
        ),
        is_training=True,
        build_argv=_train_single_argv,
    ),
    Action(
        key="train-team-red",
        label="train-team-red",
        group="Train",
        glyph_attr="TRAIN_TEAM",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="WARM_START", label="Warm start from",
                      kind=FieldKind.PICKER_MODELS),
        ),
        is_training=True,
        build_argv=_train_team_red_argv,
    ),
    Action(
        key="train-team-blue",
        label="train-team-blue",
        group="Train",
        glyph_attr="TRAIN_TEAM",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run name", kind=FieldKind.TEXT),
            FieldSpec(name="RED", label="Red opponent (frozen)",
                      kind=FieldKind.PICKER_MODELS, required=True),
        ),
        is_training=True,
        build_argv=_train_team_blue_argv,
    ),
    Action(
        key="resume",
        label="resume",
        group="Train",
        glyph_attr="RESUME",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="CHECKPOINT", label="Checkpoint",
                      kind=FieldKind.PICKER_CHECKPOINTS_IN_TRIAL, required=True,
                      depends_on=("RUN_NAME", "TRIAL")),
        ),
        is_training=True,
        build_argv=_resume_argv,
    ),
    Action(
        key="resume-team",
        label="resume-team",
        group="Train",
        glyph_attr="RESUME",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="CHECKPOINT", label="Checkpoint",
                      kind=FieldKind.PICKER_CHECKPOINTS_IN_TRIAL, required=True,
                      depends_on=("RUN_NAME", "TRIAL")),
            FieldSpec(name="LEARNER", label="Learner override (optional)",
                      kind=FieldKind.TEXT),
            FieldSpec(name="OPPONENT", label="Opponent spec override (optional)",
                      kind=FieldKind.TEXT),
        ),
        is_training=True,
        build_argv=_resume_team_argv,
    ),
]
