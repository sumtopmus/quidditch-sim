"""RLlib directory-checkpoint mechanics (Step 5c).

RLlib new-stack checkpoints are DIRECTORIES, not SB3 .zip files. Tune writes
them under runs/<run>/<ts>/tune/<trial>/checkpoint_<N>/, with one module
sub-checkpoint per id under learner_group/learner/rl_module/<module_id>/.

The dir-finding / path-building / detection helpers are pure (pathlib + regex,
no Ray import) so run_listing / promote / inventory can use them without pulling
Ray. Only load_rl_module imports Ray, lazily.
"""
from __future__ import annotations

import re
from pathlib import Path

_CKPT_DIR_RE = re.compile(r"^checkpoint_(\d+)$")
_RL_MODULE_REL = ("learner_group", "learner", "rl_module")


def find_latest_checkpoint_dir(run_root: Path | str) -> Path | None:
    """The highest-ordinal checkpoint_<N> directory anywhere under `run_root`
    (Tune nests them under tune/<trial>/). None when there are none."""
    run_root = Path(run_root)
    if not run_root.exists():
        return None
    best: tuple[int, Path] | None = None
    for d in run_root.rglob("checkpoint_*"):
        if not d.is_dir():
            continue
        m = _CKPT_DIR_RE.match(d.name)
        if not m:
            continue
        ordinal = int(m.group(1))
        if best is None or ordinal > best[0]:
            best = (ordinal, d)
    return best[1].resolve() if best else None


def module_subpath(checkpoint_dir: Path | str, module_id: str) -> Path:
    """Path to a single module's sub-checkpoint inside a checkpoint dir."""
    return Path(checkpoint_dir).joinpath(*_RL_MODULE_REL, module_id)


def module_ids(checkpoint_dir: Path | str) -> set[str]:
    """Module ids present in a checkpoint dir (the league population)."""
    rl = Path(checkpoint_dir).joinpath(*_RL_MODULE_REL)
    if not rl.is_dir():
        return set()
    return {d.name for d in rl.iterdir() if d.is_dir()}


def is_rllib_checkpoint(path: Path | str) -> bool:
    """True when `path` is an RLlib checkpoint directory (has the rl_module
    layout). False for files (.zip) and non-checkpoint dirs."""
    p = Path(path)
    return p.is_dir() and p.joinpath(*_RL_MODULE_REL).is_dir()


def load_rl_module(checkpoint_dir: Path | str, module_id: str):
    """Load a single frozen RLModule for inference. Ray imported lazily.

    NOTE: the rl_module subpath layout is RLlib-version-specific. If this raises,
    verify the on-disk layout under <checkpoint_dir>/learner_group/learner/
    rl_module/ against the installed Ray version and adjust _RL_MODULE_REL.
    """
    from ray.rllib.core.rl_module.rl_module import RLModule

    sub = module_subpath(checkpoint_dir, module_id)
    if not sub.is_dir():
        raise FileNotFoundError(
            f"no module sub-checkpoint for {module_id!r} at {sub}")
    return RLModule.from_checkpoint(str(sub))
