"""Manage actions: tensorboard, promote, list-runs, lineage, repro."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _tensorboard_argv(v: dict) -> list[str]:
    logdir = f"runs/{v['RUN_NAME']}" if v.get("RUN_NAME") else "runs"
    return ["tensorboard", "--logdir", logdir]


def _promote_argv(v: dict) -> list[str]:
    return ["python", "scripts/promote.py",
            "--trial", f"runs/{v['RUN_NAME']}/{v['TRIAL']}"]


def _list_runs_argv(_v: dict) -> list[str]:
    return ["python", "scripts/list_runs.py"]


def _lineage_argv(v: dict) -> list[str]:
    return ["python", "scripts/lineage.py", f"runs/{v['RUN_NAME']}/{v['TRIAL']}"]


def _repro_argv(v: dict) -> list[str]:
    return ["python", "scripts/repro.py", "--model-dir", v["MODEL"]]


MANAGE = [
    Action(
        key="tensorboard",
        label="tensorboard",
        group="Manage",
        glyph_attr="TENSORBOARD",
        fields=(FieldSpec(name="RUN_NAME", label="Run (optional — all if blank)",
                          kind=FieldKind.PICKER_RUNS),),
        is_tensorboard=True,
        build_argv=_tensorboard_argv,
    ),
    Action(
        key="promote",
        label="promote",
        group="Manage",
        glyph_attr="PROMOTE",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
        ),
        build_argv=_promote_argv,
    ),
    Action(
        key="list-runs",
        label="list-runs",
        group="Manage",
        glyph_attr="LIST_RUNS",
        fields=(),
        build_argv=_list_runs_argv,
    ),
    Action(
        key="lineage",
        label="lineage",
        group="Manage",
        glyph_attr="LINEAGE",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
        ),
        build_argv=_lineage_argv,
    ),
    Action(
        key="repro",
        label="repro",
        group="Manage",
        glyph_attr="REPRO",
        fields=(FieldSpec(name="MODEL", label="Promoted model",
                          kind=FieldKind.PICKER_MODELS, required=True),),
        build_argv=_repro_argv,
    ),
]
