"""Shared training infrastructure used by scripts/train.py.

Owns:
  - the standard callback set (checkpoint + eval + optional video)
  - .hydra/meta.yaml helpers (write + append + parent-chain walk)
  - legacy info.toml obs-spec reader + compat check (used by Phase 6 migrator)

The Hydra entrypoint (`scripts/train.py`) owns run-dir creation, config
loading, and lifecycle management; this module is a thin helpers library.
"""
from __future__ import annotations

import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.quidditch.obs_spec import ObsBlock, ObsSpec


def _format_slot(block: ObsBlock) -> str:
    """Return the inline-table TOML representation of one ObsBlock."""
    parts = [f'name = "{block.name}"', f"dim = {block.dim}"]
    if block.frame is not None:
        parts.append(f'frame = "{block.frame}"')
    if block.notes is not None:
        # Escape backslashes and double-quotes for TOML basic strings.
        escaped = block.notes.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'notes = "{escaped}"')
    return "{" + ", ".join(parts) + "}"


def format_obs_block(spec: ObsSpec, n_stack: int) -> str:
    """Return the [obs] block text (including the [obs] header) for run_info.toml.

    Inline-array-of-tables form: one {name=..., dim=..., frame=..., notes=...}
    entry per block.  Optional fields (frame, notes) are omitted when None.
    """
    lines = ["\n[obs]", f"dim     = {spec.dim}", f"n_stack = {n_stack}",
             "# slots is an inline array-of-tables; each entry is one ObsBlock.",
             "slots = ["]
    for block in spec.blocks:
        lines.append(f"  {_format_slot(block)},")
    lines.append("]\n")
    return "\n".join(lines)


def read_obs_spec(info_path: Path | str) -> tuple[ObsSpec, int] | None:
    """Parse run_info.toml and return (ObsSpec, n_stack) or None if [obs] absent."""
    data = tomllib.loads(Path(info_path).read_text())
    obs = data.get("obs")
    if obs is None:
        return None
    blocks = tuple(
        ObsBlock(
            name=s["name"], dim=s["dim"],
            frame=s.get("frame"), notes=s.get("notes"),
        )
        for s in obs["slots"]
    )
    spec = ObsSpec(blocks)
    if spec.dim != obs["dim"]:
        raise ValueError(
            f"[obs].dim={obs['dim']} disagrees with sum of slot dims ({spec.dim}) "
            f"in {info_path}"
        )
    return spec, int(obs["n_stack"])


def _block_summary(block: ObsBlock | None) -> str:
    if block is None:
        return "—"  # em-dash
    parts = [block.name.ljust(14), f"dim={block.dim}"]
    if block.frame is not None:
        parts.append(block.frame)
    return " ".join(parts)


def _render_diff(parent_spec: ObsSpec | None, parent_n_stack: int | None,
                 current_spec: ObsSpec, current_n_stack: int,
                 parent_path: Path) -> str:
    """Render the column-by-column diff including emoji status indicators."""
    out: list[str] = []
    out.append("Obs spec mismatch between parent and current env.")
    out.append(f"  parent: {parent_path}")
    if parent_spec is None:
        out.append(f"  (parent has no [obs] block — older run, pre-obs-spec feature)")
        out.append(f"  current: {current_spec.dim}-d  (n_stack={current_n_stack})")
        out.append("")
        out.append("  Run with --obs-surgery to copy matching blocks and "
                   "small-init the rest.")
        return "\n".join(out)

    out.append(f"  current: {current_spec.dim}-d  (n_stack={current_n_stack})")
    out.append("")
    header = f"  {'offset':<8} {'parent':<30}  {'current':<30}"
    out.append(header)
    out.append(f"  {'-' * 6:<8} {'-' * 28:<30}  {'-' * 28:<30}")

    # Parallel walk by name+dim.  Same-name-same-dim aligns; otherwise look
    # ahead to decide whether to emit "removed" (advance parent) or "added"
    # (advance current) — the choice depends on which side carries the upcoming
    # match.
    p_blocks = list(parent_spec.blocks)
    c_blocks = list(current_spec.blocks)
    pi = ci = 0
    p_off = c_off = 0
    notes_diffs: list[tuple[str, str | None, str | None]] = []

    def _pair_aligned(pb: ObsBlock, cb: ObsBlock) -> bool:
        return pb.name == cb.name and pb.dim == cb.dim

    while pi < len(p_blocks) or ci < len(c_blocks):
        pb = p_blocks[pi] if pi < len(p_blocks) else None
        cb = c_blocks[ci] if ci < len(c_blocks) else None
        if pb is not None and cb is not None and _pair_aligned(pb, cb):
            # Aligned: either exact match, frame-only diff, or notes-only diff.
            offset = f"{p_off}:{p_off + pb.dim}"
            if pb.frame == cb.frame:
                status = "✅"
                if pb.notes != cb.notes:
                    notes_diffs.append((pb.name, pb.notes, cb.notes))
            else:
                status = "⚠️"  # frame changed
            out.append(
                f"  {offset:<8} {_block_summary(pb):<30}  {_block_summary(cb):<30}  "
                f"{status}{'  frame changed' if status == '⚠️' else ''}"
            )
            pi += 1; ci += 1
            p_off += pb.dim; c_off += cb.dim
            continue
        # Decide which side to advance: if parent's head will be matched by a
        # later current block, emit "added" for current's head first; otherwise
        # parent's head is gone — emit "removed" and advance parent.
        parent_head_in_current = (
            pb is not None
            and any(_pair_aligned(pb, c) for c in c_blocks[ci:])
        )
        advance_parent = cb is None or (pb is not None and not parent_head_in_current)
        if advance_parent and pb is not None:
            offset = f"{p_off}:{p_off + pb.dim}"
            out.append(
                f"  {offset:<8} {_block_summary(pb):<30}  {_block_summary(None):<30}  "
                f"❌  removed"
            )
            pi += 1; p_off += pb.dim
        else:
            offset = f"{c_off}:{c_off + cb.dim}"
            out.append(
                f"  {offset:<8} {_block_summary(None):<30}  {_block_summary(cb):<30}  "
                f"❌  added"
            )
            ci += 1; c_off += cb.dim

    if parent_n_stack != current_n_stack:
        out.append(
            f"  {'n_stack':<8} {str(parent_n_stack):<30}  {str(current_n_stack):<30}  "
            f"❌"
        )

    if notes_diffs:
        out.append("")
        out.append("  ⚠️  Notes changed on matching blocks "
                   "(informational, load proceeds):")
        for name, pn, cn in notes_diffs:
            out.append(f"       {name}:")
            out.append(f"         parent:  {pn or '(none)'}")
            out.append(f"         current: {cn or '(none)'}")

    out.append("")
    out.append("  Run with --obs-surgery to copy matching blocks and "
               "small-init the rest.")
    return "\n".join(out)


def _is_compat(parent_spec: ObsSpec, parent_n_stack: int,
               current_spec: ObsSpec, current_n_stack: int) -> bool:
    """Return True iff load can proceed without surgery."""
    if parent_n_stack != current_n_stack:
        return False
    if len(parent_spec.blocks) != len(current_spec.blocks):
        return False
    for pb, cb in zip(parent_spec.blocks, current_spec.blocks):
        if pb.name != cb.name or pb.dim != cb.dim or pb.frame != cb.frame:
            return False
    # Notes differences are tolerated (informational).
    return True


def check_obs_compat(parent_info: Path | str, current: ObsSpec, current_n_stack: int,
                     *, surgery: bool) -> tuple[ObsSpec | None, int | None]:
    """Compare parent's [obs] block to (current, current_n_stack).

    Returns (parent_spec, parent_n_stack) on success.  On strict-mode failure
    (mismatch with surgery=False), prints the diff to stdout and calls
    sys.exit(2).  When surgery=True, accepts any mismatch (including a parent
    that lacks an [obs] block — returns (None, None) in that case).
    """
    parent_path = Path(parent_info)
    parsed = read_obs_spec(parent_path)

    if parsed is None:
        if surgery:
            return None, None
        print(_render_diff(None, None, current, current_n_stack, parent_path))
        sys.exit(2)

    parent_spec, parent_n_stack = parsed
    if _is_compat(parent_spec, parent_n_stack, current, current_n_stack):
        # Even on compat, surface notes diffs as informational.
        diff = _render_diff(parent_spec, parent_n_stack, current, current_n_stack,
                            parent_path)
        if "Notes changed" in diff:
            # Print only the notes-changed section (compat passed otherwise).
            tail = diff.split("⚠️  Notes changed", 1)
            print("⚠️  Notes changed" + tail[1])
        return parent_spec, parent_n_stack

    if surgery:
        return parent_spec, parent_n_stack

    print(_render_diff(parent_spec, parent_n_stack, current, current_n_stack,
                       parent_path))
    sys.exit(2)


def build_callbacks(
    *,
    run_dir: Path,
    eval_env_fn: Callable[[], Any],
    config: dict[str, Any],
    n_envs: int,
    video_env_fn: Callable[[], Any] | None = None,
    verbose: int = 0,
    frame_stack: int = 1,
) -> list:
    """Build the standard SB3 callback set: checkpoint + eval + (optional) video.

    When ``video_env_fn`` is provided, a ``VideoRecorderCallback`` is appended
    using the ``[training.callbacks.video]`` sub-block.  ``video_env_fn`` should
    return an env in ``rgb_array`` mode (single-agent ``QuidditchSimpleEnv`` or
    team-agent ``OpponentControlledEnv``); the callback resets and rolls out one
    deterministic episode at every ``video_every_n_evals``-th eval trigger.

    ``verbose`` is propagated to every callback so a progress-bar run (verbose=0)
    stays silent while ``--verbose`` keeps SB3's per-eval / per-checkpoint /
    per-video chatter.
    """
    eval_freq = max(config["training"]["eval"]["eval_freq_steps"] // n_envs, 1)
    ckpt_freq = max(
        config["training"]["callbacks"]["checkpoint_freq_steps"] // n_envs, 1
    )

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from stable_baselines3.common.monitor import Monitor
    # Monitor-wrap eval env so SB3's evaluate_policy stops warning about it —
    # eval reward/length numbers stay correct either way (no other wrapper
    # mutates them) but the wrapper is the canonical fix.
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_fn())])
    # Must match the training-env stack depth or SB3 will reject the eval env
    # at EvalCallback construction (obs-space shape mismatch).
    if frame_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack)

    cbs: list = [
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            verbose=verbose,
        ),
    ]
    # SB3 emits an unconditional UserWarning when train (SubprocVecEnv) and eval
    # (DummyVecEnv) types differ.  Intentional here — single-process eval is
    # cheaper and behaves the same.  The warning fires from EvalCallback's
    # _init_callback() (invoked by model.learn(), not by __init__), so a
    # scoped `with warnings.catch_warnings()` around construction is too short-
    # lived — install a persistent filter for the rest of this training run.
    warnings.filterwarnings(
        "ignore",
        message="Training and eval env are not of the same type",
        category=UserWarning,
    )
    cbs.append(
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=eval_freq,
            n_eval_episodes=config["training"]["eval"]["n_eval_episodes"],
            deterministic=True,
            verbose=verbose,
        )
    )

    if video_env_fn is not None:
        # Local import to avoid a hard dep on imageio/moviepy when video is off.
        from scripts.callbacks import VideoRecorderCallback
        from core.world import CONTROL_HZ

        video_cfg = config["training"]["callbacks"].get("video", {})
        video_freq = eval_freq * config["training"]["callbacks"]["video_every_n_evals"]
        cbs.append(
            VideoRecorderCallback(
                env_fn=video_env_fn,
                video_dir=str(run_dir / "videos"),
                record_freq=video_freq,
                fps=config["training"]["callbacks"]["video_fps"],
                sim_hz=CONTROL_HZ,
                grid=video_cfg.get("grid", True),
                grid_cams=tuple(video_cfg["cells"]) if "cells" in video_cfg else None,
                cell_width=video_cfg.get("cell_width", 960),
                cell_height=video_cfg.get("cell_height", 540),
                verbose=verbose,
            )
        )

    return cbs


# ── meta.yaml helpers (Phase 4 additions; TOML helpers stay until Phase 5) ───
def _git_hash() -> str:
    """Return the current HEAD short hash, or '<unknown>' if not in a repo."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "<unknown>"


def write_meta_yaml(
    run_dir: Path | str,
    *,
    git_hash: str | None = None,
    parent_chain_total: int = 0,
    init_mode: str = "scratch",
    parent_path: str | None = None,
) -> None:
    """Write the start-of-run meta.yaml into `<run_dir>/.hydra/meta.yaml`.

    Final stats are appended later by `append_meta_yaml_final_stats`.
    """
    import yaml
    run_dir = Path(run_dir)
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    payload = {
        "git_hash":           git_hash if git_hash is not None else _git_hash(),
        "parent_chain_total": int(parent_chain_total),
        "init_mode":          str(init_mode),
        "parent_path":        parent_path,
    }
    (run_dir / ".hydra" / "meta.yaml").write_text(yaml.safe_dump(payload))


def append_meta_yaml_final_stats(
    run_dir: Path | str,
    *,
    wall_time_s: float,
    completed_steps: int,
    best_eval_reward: float | None = None,
    peak_eval_step: int | None = None,
) -> None:
    """Merge final-stats fields into `<run_dir>/.hydra/meta.yaml`."""
    import yaml
    run_dir = Path(run_dir)
    p = run_dir / ".hydra" / "meta.yaml"
    if p.exists():
        payload = yaml.safe_load(p.read_text()) or {}
    else:
        payload = {}
    payload["final_stats"] = {
        "wall_time_s":      float(wall_time_s),
        "completed_steps":  int(completed_steps),
        "best_eval_reward": None if best_eval_reward is None else float(best_eval_reward),
        "peak_eval_step":   None if peak_eval_step  is None else int(peak_eval_step),
    }
    p.write_text(yaml.safe_dump(payload))


def read_parent_chain_total_from_hydra(parent_path: str | Path) -> int | None:
    """Walk up from a parent best_model path to its `.hydra/meta.yaml` and
    return parent_chain_total + final_stats.completed_steps.

    parent_path forms:
      - models/X/best_model[.zip]        → models/X/.hydra/meta.yaml
      - runs/X/<trial>/checkpoints/c.zip → runs/X/<trial>/.hydra/meta.yaml
      - runs/X/<trial>/best_model[.zip]  → runs/X/<trial>/.hydra/meta.yaml
    """
    import yaml
    parent_path = Path(parent_path).resolve()
    # Strip a trailing checkpoints/ if present.
    candidates = [parent_path.parent, parent_path.parent.parent]
    for cand in candidates:
        meta_path = cand / ".hydra" / "meta.yaml"
        if meta_path.exists():
            data = yaml.safe_load(meta_path.read_text()) or {}
            ancestry = int(data.get("parent_chain_total", 0))
            this_run = int(data.get("final_stats", {}).get("completed_steps", 0))
            return ancestry + this_run
    return None
