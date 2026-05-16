"""Pure renderer for per-model MODEL.md spec sheets.

Reads .hydra/{config,meta,hydra}.yaml + _wandb_metadata.json from a run dir
and returns a Markdown string.  No filesystem writes, no wandb calls.

See docs/superpowers/specs/2026-05-16-model-doc-generator-design.md.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _load_run_context(run_dir: Path) -> dict[str, Any]:
    """Gather config + meta + hydra-choices + wandb-meta into one dict.

    Required: `.hydra/config.yaml`.  All other inputs optional; missing ones
    surface as `None` in the returned ctx.
    """
    hdir = run_dir / ".hydra"
    cfg_path = hdir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"required input missing: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    meta_path = hdir / "meta.yaml"
    meta = (
        OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
        if meta_path.exists() else None
    )

    hydra_yaml_path = hdir / "hydra.yaml"
    hydra_yaml = (
        OmegaConf.to_container(OmegaConf.load(hydra_yaml_path), resolve=True)
        if hydra_yaml_path.exists() else None
    )

    wandb_meta_path = run_dir / "_wandb_metadata.json"
    wandb_meta = (
        json.loads(wandb_meta_path.read_text()) if wandb_meta_path.exists() else None
    )

    return {
        "cfg": cfg,
        "meta": meta,
        "hydra_yaml": hydra_yaml,
        "wandb_meta": wandb_meta,
        "run_dir": run_dir,
    }


def _section_header(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    wandb_meta = ctx["wandb_meta"]
    run_dir = ctx["run_dir"]

    # run_dir basename is <timestamp> (under runs/<name>/<ts>/) or <name>
    # (under models/<name>/).  Join with run_name unless the basename already
    # IS the run_name (e.g., the vendored models/<name>/ layout).
    timestamp = run_dir.name
    title = (
        f"# MODEL: {cfg.run_name}_{timestamp}"
        if timestamp != cfg.run_name else f"# MODEL: {cfg.run_name}"
    )

    status = "promoted" if wandb_meta else "run-only"
    git = meta.get("git_hash", "(unknown)") if meta else "(unknown)"

    lines = [
        title,
        "",
        f"**Status:** {status}  ·  **Git:** `{git}`",
    ]
    if wandb_meta:
        name = wandb_meta["name"]
        version = wandb_meta["version"]
        aliases = ", ".join(wandb_meta.get("aliases", []))
        lines.append(f"**W&B:** `wandb://{name}:prod` ({version}, aliases: {aliases})")
    return "\n".join(lines)


def _section_summary(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    hydra_yaml = ctx["hydra_yaml"]

    description = ""
    if hasattr(cfg, "get"):
        description = (cfg.get("description") or "").strip()
    elif hasattr(cfg, "description"):
        description = (cfg.description or "").strip()
    if description:
        return "## Summary\n\n" + description

    # Auto-template path.
    learner_id = "drone_0"
    if hasattr(cfg, "env") and cfg.env is not None and hasattr(cfg.env, "get"):
        learner_id = cfg.env.get("learner_id", "drone_0")

    init_mode = "scratch"
    parent = None
    if hasattr(cfg, "init") and cfg.init is not None:
        init_mode = cfg.init.mode
        if hasattr(cfg.init, "get"):
            parent = cfg.init.get("parent", None)

    obs_name = cfg.obs.name
    n_stack = cfg.obs.n_stack
    total_steps = int(cfg.trainer.total_timesteps)
    lr = cfg.trainer.lr
    curriculum = "(unknown)"
    if hasattr(cfg, "curriculum") and cfg.curriculum is not None and hasattr(cfg.curriculum, "get"):
        curriculum = cfg.curriculum.get("name", "(unknown)")

    # Opponent shorthand: read `spec` field if present, else _target_ class name.
    opp_short = None
    opp = cfg.get("opponent", None) if hasattr(cfg, "get") else None
    if opp is not None:
        if hasattr(opp, "get"):
            opp_short = opp.get("spec", None) or opp.get("_target_", "").rsplit(".", 1)[-1]
        else:
            opp_short = str(opp)

    # Reward group choice from hydra.yaml; fall back if absent.
    reward_choice = "(unknown)"
    if hydra_yaml:
        reward_choice = (
            hydra_yaml.get("hydra", {})
            .get("runtime", {})
            .get("choices", {})
            .get("reward", "(unknown)")
        )

    parts = [f"{learner_id} learner trained from {init_mode}"]
    if parent:
        parts.append(f"(parent: {parent})")
    if opp_short:
        parts.append(f"against {opp_short}")
    parts.append(f"on {obs_name} × n_stack={n_stack}")
    parts.append(f"reward stack {reward_choice}")
    parts.append(f"lr={lr}")
    parts.append(curriculum)
    parts.append(f"{total_steps:,} steps.")

    body = ", ".join(parts)
    return "## Summary\n\n" + body


def _section_lineage(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    meta = ctx["meta"] or {}
    init_mode = cfg.init.mode if hasattr(cfg, "init") and cfg.init is not None else "scratch"
    if init_mode == "scratch":
        return "## Lineage\n\n- **init mode:** `scratch` — no parent"

    parent = None
    if hasattr(cfg.init, "get"):
        parent = cfg.init.get("parent", None)
    chain_total = meta.get("parent_chain_total", None) if meta else None
    this_total = int(cfg.trainer.total_timesteps)

    lines = ["## Lineage", "", f"- **init mode:** `{init_mode}`"]
    if parent:
        lines.append(f"- **parent:** `{parent}`")
    if chain_total is not None:
        lines.append(
            f"- **parent chain total:** {chain_total:,} steps "
            f"(this run is {this_total:,} of that)"
        )
    return "\n".join(lines)


def _section_obs_spec(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    from envs.quidditch.obs_spec import SPEC_BY_NAME

    name = cfg.obs.name
    n_stack = int(cfg.obs.n_stack)
    spec = SPEC_BY_NAME.get(name)
    if spec is None:
        return (
            "## Obs spec\n\n"
            f"> ⚠ unknown obs spec name `{name}` — not in SPEC_BY_NAME registry. "
            "See `.hydra/config.yaml:obs` for the recorded name."
        )

    header = (
        "## Obs spec\n\n"
        f"**Name:** `{name}` ({spec.dim}-d)  ·  **n_stack:** {n_stack}  ·  "
        f"**Input dim:** {spec.dim * n_stack}"
    )
    rows = [
        "| Slot | Block | Dim | Frame | Notes |",
        "|------|-------|-----|-------|-------|",
    ]
    for block, sl in spec.offsets():
        notes = block.notes or ""
        frame = block.frame or ""
        rows.append(f"| {sl.start}:{sl.stop} | {block.name} | {block.dim} | {frame} | {notes} |")
    return header + "\n\n" + "\n".join(rows)


def _term_coefficient_summary(term: Any) -> str:
    """Render a term's numeric / dict fields as `name=value` joined by commas."""
    parts: list[str] = []
    for f in dataclasses.fields(term):
        val = getattr(term, f.name)
        if isinstance(val, bool):
            # bool is a subclass of int — surface it as-is rather than 0/1.
            parts.append(f"{f.name}={val}")
        elif isinstance(val, (int, float)):
            parts.append(f"{f.name}={val}")
        elif isinstance(val, dict):
            parts.append(f"{f.name}={dict(val)}")
    return ", ".join(parts)


def _term_agents_summary(term: Any) -> str:
    """Render the term's agent-identifying fields as human shorthand."""
    out: list[str] = []
    for f in dataclasses.fields(term):
        val = getattr(term, f.name)
        if f.name in ("gainer", "loser", "scorer", "zero_sum_opponent",
                       "aggressor", "victim"):
            if val:
                out.append(f"{val} ({f.name})")
        elif f.name == "agents":
            if val:
                out.append(", ".join(val))
        elif f.name in ("agent_to_target", "agent_to_crash_flags"):
            if val:
                out.append(", ".join(f"{k}→{v}" for k, v in val.items()))
    return "; ".join(out)


def _section_reward_stack(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    hydra_yaml = ctx["hydra_yaml"]
    from hydra.utils import instantiate

    # Source line: reward group choice from hydra.yaml, or fallback.
    if hydra_yaml:
        choice = (
            hydra_yaml.get("hydra", {})
            .get("runtime", {})
            .get("choices", {})
            .get("reward", None)
        )
        source = f"`conf/reward/{choice}.yaml`" if choice else "(in-line override / unknown source)"
    else:
        source = "(in-line override / unknown source)"

    stack = instantiate(cfg.reward, _convert_="all")

    rows = [
        "| # | Term | Key coefficients | Agents |",
        "|---|------|------------------|--------|",
    ]
    for i, term in enumerate(stack.terms, start=1):
        cls = type(term).__name__
        coeffs = _term_coefficient_summary(term)
        agents = _term_agents_summary(term)
        rows.append(f"| {i} | {cls} | {coeffs} | {agents} |")

    return f"## Reward stack\n\n**Source:** {source}\n\n" + "\n".join(rows)


def _opponent_short(cfg) -> str:
    opp = cfg.get("opponent", None) if hasattr(cfg, "get") else None
    if opp is None:
        return "(none)"
    if hasattr(opp, "get") and opp.get("spec"):
        return opp.get("spec")
    target = opp.get("_target_", "") if hasattr(opp, "get") else ""
    return target.rsplit(".", 1)[-1] if target else "(unknown)"


def _section_env_config(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    learner = "drone_0"
    if hasattr(cfg, "env") and cfg.env is not None and hasattr(cfg.env, "get"):
        learner = cfg.env.get("learner_id", "drone_0")
    team_cfg = None
    if hasattr(cfg, "env") and cfg.env is not None and hasattr(cfg.env, "get"):
        team_cfg = cfg.env.get("team_cfg", None)
    curriculum = "(unknown)"
    if hasattr(cfg, "curriculum") and cfg.curriculum is not None and hasattr(cfg.curriculum, "get"):
        curriculum = cfg.curriculum.get("name", "(unknown)")
    opp = _opponent_short(cfg)

    lines = ["## Env config", ""]
    lines.append(f"- **Opponent:** `{opp}`  ·  **Learner:** `{learner}`")
    if team_cfg:
        episode = team_cfg.get("episode_seconds", "(unknown)")
        lines.append(f"- **Episode:** {episode} s  ·  **Curriculum:** `{curriculum}`")
        tag_r = team_cfg.get("tag_radius", None)
        crash_thr = team_cfg.get("crash_vel_thr", None)
        if tag_r is not None and crash_thr is not None:
            lines.append(f"- **Tag radius:** {tag_r} m  ·  **Crash velocity threshold:** {crash_thr} m/s")
        midpoint = team_cfg.get("midpoint_alpha", None)
        aftermath = team_cfg.get("crash_aftermath_seconds", None)
        if midpoint is not None or aftermath is not None:
            lines.append(f"- **Midpoint α:** {midpoint}  ·  **Crash aftermath:** {aftermath} s")
    else:
        lines.append(f"- **Curriculum:** `{curriculum}`")
    return "\n".join(lines)


def _section_hyperparams(ctx: dict[str, Any]) -> str:
    t = ctx["cfg"].trainer
    total = int(t.total_timesteps)
    return "\n".join([
        "## Training hyperparams",
        "",
        f"- **Algorithm:** PPO  ·  **lr:** {t.lr}  ·  **total_timesteps:** {total:,}",
        f"- **n_envs:** {t.get('n_envs', '(unknown)')}  ·  "
        f"**batch_size:** {t.get('batch_size', '(unknown)')}  ·  "
        f"**n_epochs:** {t.get('n_epochs', '(unknown)')}",
        f"- **gamma:** {t.get('gamma', '(unknown)')}  ·  "
        f"**gae_lambda:** {t.get('gae_lambda', '(unknown)')}  ·  "
        f"**ent_coef:** {t.get('ent_coef', '(unknown)')}  ·  "
        f"**clip_range:** {t.get('clip_range', '(unknown)')}",
    ])


def _format_wall_clock(seconds: float | int) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s_ = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s_:02d}s"
    return f"{m}m {s_:02d}s"


def _section_eval_results(ctx: dict[str, Any]) -> str:
    meta = ctx["meta"]
    if meta is None or "final_stats" not in meta:
        return "## Eval results\n\n(meta.yaml absent — eval section unavailable)"

    fs = meta["final_stats"]
    model_kind = fs.get("model_kind", "(unknown)")
    lines = ["## Eval results", ""]
    if model_kind == "best":
        reward = fs.get("best_eval_reward", "(unknown)")
        best_step = fs.get("best_step", None)
        if isinstance(best_step, int):
            lines.append(f"- **best_eval_reward:** {reward} @ step {best_step:,}")
        else:
            lines.append(f"- **best_eval_reward:** {reward}")
    completed = fs.get("completed_steps", "(unknown)")
    wall = fs.get("wall_clock_seconds", None)
    wall_str = _format_wall_clock(wall) if isinstance(wall, (int, float)) else "(unknown)"
    completed_str = f"{completed:,}" if isinstance(completed, int) else str(completed)
    lines.append(f"- **completed_steps:** {completed_str}  ·  **wall_clock:** {wall_str}")
    lines.append(f"- **model_kind:** `{model_kind}`")
    return "\n".join(lines)


def _section_wandb(ctx: dict[str, Any]) -> str:
    wandb_meta = ctx["wandb_meta"]
    if not wandb_meta:
        return ""

    cfg = ctx["cfg"]
    entity = wandb_meta.get("entity", "(unknown)")
    project = wandb_meta.get("project", "(unknown)")
    name = wandb_meta.get("name", "(unknown)")
    version = wandb_meta.get("version", "(unknown)")
    aliases = ", ".join(wandb_meta.get("aliases", []))
    run_id = f"{cfg.run_name}_{ctx['run_dir'].name}" if hasattr(cfg, "run_name") else "(unknown)"

    return "\n".join([
        "## W&B",
        "",
        f"- **Project:** `{entity}/{project}`",
        f"- **Run id:** `{run_id}`",
        f"- **Artifact:** `{name}:{version}`  (aliases: {aliases})",
    ])


_SECTION_FUNCTIONS = [
    ("header",       _section_header),
    ("summary",      _section_summary),
    ("lineage",      _section_lineage),
    ("obs_spec",     _section_obs_spec),
    ("reward_stack", _section_reward_stack),
    ("env_config",   _section_env_config),
    ("hyperparams",  _section_hyperparams),
    ("eval_results", _section_eval_results),
    ("wandb",        _section_wandb),
]


def render_model_doc(run_dir: Path) -> str:
    """Render a per-model MODEL.md spec sheet for one run dir.

    Pure: reads inputs from `run_dir`, returns a Markdown string.  No
    filesystem writes, no wandb network calls.  Per-section try/except so
    a bad section doesn't kill the whole doc.
    """
    ctx = _load_run_context(run_dir)
    parts: list[str] = []
    for name, fn in _SECTION_FUNCTIONS:
        try:
            out = fn(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("section %s render failed: %s", name, e)
            out = f"## (Section `{name}` failed to render)\n\n> ⚠ {type(e).__name__}: {e}"
        if out:  # _section_wandb returns "" when meta absent
            parts.append(out)
    return "\n\n".join(parts) + "\n"
