"""Eval actions: single-agent eval + team eval. GUI flag selects python vs mjpython."""
from __future__ import annotations

from tui.actions.base import Action, FieldKind, FieldSpec


def _eval_argv(v: dict) -> list[str]:
    interp = "mjpython" if v.get("GUI") else "python"
    out = [
        interp, "scripts/eval_ppo.py",
        "--model", f"runs/{v['RUN_NAME']}/{v['TRIAL']}/best_model",
        "--episodes", str(int(v.get("EPISODES") or 10)),
    ]
    if not v.get("GUI"):
        out.append("--no-render")
    return out


def _eval_team_argv(v: dict) -> list[str]:
    interp = "mjpython" if v.get("GUI") else "python"
    out = [
        interp, "scripts/eval_team.py",
        "--red", v["RED"], "--blue", v["BLUE"],
        "--episodes", str(int(v.get("EPISODES") or (5 if v.get("GUI") else 100))),
    ]
    if v.get("GUI"):
        out.append("--gui")
    if v.get("DETERMINISTIC"):
        out.append("--deterministic")
    return out


EVAL = [
    Action(
        key="eval",
        label="eval",
        group="Eval",
        glyph_attr="EVAL",
        fields=(
            FieldSpec(name="RUN_NAME", label="Run", kind=FieldKind.PICKER_RUNS, required=True),
            FieldSpec(name="TRIAL", label="Trial",
                      kind=FieldKind.PICKER_TRIALS_IN_RUN, required=True,
                      depends_on=("RUN_NAME",)),
            FieldSpec(name="EPISODES", label="Episodes", kind=FieldKind.INT, default=10),
            FieldSpec(name="GUI", label="GUI viewer", kind=FieldKind.BOOL, default=True),
        ),
        requires_mjpython=True,  # when GUI=True; runner inspects argv[0]
        build_argv=_eval_argv,
    ),
    Action(
        key="eval-team",
        label="eval-team",
        group="Eval",
        glyph_attr="EVAL",
        fields=(
            FieldSpec(name="RED", label="Red model", kind=FieldKind.PICKER_MODELS, required=True),
            FieldSpec(name="BLUE", label="Blue model", kind=FieldKind.PICKER_MODELS, required=True),
            FieldSpec(name="EPISODES", label="Episodes", kind=FieldKind.INT, default=5),
            FieldSpec(name="GUI", label="GUI viewer", kind=FieldKind.BOOL, default=True),
            FieldSpec(name="DETERMINISTIC", label="Deterministic", kind=FieldKind.BOOL, default=True),
        ),
        requires_mjpython=True,
        build_argv=_eval_team_argv,
    ),
]
