"""Walk a trial's pretrain ancestry and print one row per training segment.

Each row shows the cumulative step range, the trial dir, and a few key config
knobs that typically vary across pretrain steps (lr, ent_coef, randomise_start,
episode_seconds).  The Hydra-shaped config lives in `<trial>/.hydra/config.yaml`;
legacy info.toml/config.toml are still read for un-migrated promoted models.

Usage:
    python scripts/lineage.py runs/ppo_hoop_rand_start/20260430_234354
    python scripts/lineage.py models/ppo_hoop_rand_start_20260430_234354
    make lineage RUN_NAME=ppo_hoop_rand_start          # latest trial in that run
    make lineage TRIAL=20260430_234354 RUN_NAME=...    # specific trial

The chain is reconstructed by reading the parent edge from each trial's
.hydra/config.yaml `init.parent` (or `[pretrain].parent` in legacy info.toml)
and recursing.  If a parent dir is missing the chain is truncated with a warning.
"""

import argparse
import sys
import tomllib
from pathlib import Path

import yaml


# (tuple-of-dotted-fallback-paths, column-header, format-string)
# Hydra layout uses `trainer.*` / `curriculum.*`; legacy snapshots used
# `training.ppo.*` / `env.*`.  Tuples let us try both.
DISPLAYED_KEYS = [
    (("trainer.lr",            "training.ppo.lr",       "ppo.lr"),       "lr",         "{:.1e}"),
    (("trainer.ent_coef",      "training.ppo.ent_coef", "ppo.ent_coef"), "ent_coef",   "{:.2g}"),
    (("curriculum.randomise_start", "env.randomise_start"),              "rand_start", "{}"),
    (("curriculum.episode_seconds", "env.episode_seconds"),              "episode_s",  "{:.1f}"),
]


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except Exception as exc:
        print(f"WARN: failed to parse {path}: {exc}", file=sys.stderr)
        return {}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        print(f"WARN: failed to parse {path}: {exc}", file=sys.stderr)
        return {}


def _find_first(dir_: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        c = dir_ / name
        if c.exists():
            return c
    return None


def _resolve_dotted(data: dict, dotted: str):
    cur = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_first(data: dict, paths: tuple[str, ...]):
    """Return the first dotted-path that resolves to a non-None value."""
    for p in paths:
        v = _resolve_dotted(data, p)
        if v is not None:
            return v
    return None


def walk_chain(start: Path) -> list[dict]:
    """Walk pretrain ancestry from `start` backwards. Returns oldest-first.

    Each segment reads .hydra/config.yaml + .hydra/meta.yaml when present
    (Hydra layout, post-Phase-6 migration).  Falls back to info.toml +
    config.toml for un-migrated dirs.
    """
    chain: list[dict] = []
    cur: Path | None = start.resolve()
    visited: set[Path] = set()

    while cur is not None:
        if cur in visited:
            print(f"WARN: cycle detected at {cur} — stopping walk", file=sys.stderr)
            break
        visited.add(cur)

        if not cur.exists():
            print(f"WARN: {cur} does not exist — chain truncated", file=sys.stderr)
            break

        hydra_dir = cur / ".hydra"
        hydra_cfg = _load_yaml(hydra_dir / "config.yaml")
        hydra_meta = _load_yaml(hydra_dir / "meta.yaml")

        info_path = _find_first(cur, ("info.toml", "run_info.toml"))
        cfg_path = _find_first(cur, ("config_snapshot.toml", "config.toml"))
        env_path = _find_first(cur, ("env_snapshot.toml", "env.toml"))
        info = _load_toml(info_path) if info_path else {}
        cfg = _load_toml(cfg_path) if cfg_path else {}
        if env_path:
            env_data = _load_toml(env_path)
            if "env" in env_data:
                cfg.setdefault("env", {}).update(env_data["env"])

        # Merge Hydra config into legacy cfg so _resolve_first lookups see both.
        # `trainer.*` / `curriculum.*` (Hydra) take precedence over legacy keys.
        merged_cfg = dict(cfg)
        merged_cfg.update(hydra_cfg)

        # Steps-trained comes from meta.yaml's final_stats (new) or
        # run.steps_trained (legacy).
        steps = None
        if hydra_meta:
            steps = hydra_meta.get("final_stats", {}).get("completed_steps")
        if steps is None:
            steps = info.get("run", {}).get("steps_trained")

        chain.append({
            "dir": cur,
            "info": info,
            "config": merged_cfg,
            "hydra_meta": hydra_meta,
            "steps_trained": steps,
        })

        # Parent edge: Hydra init.parent first, then legacy [pretrain].parent.
        parent_path = (
            hydra_cfg.get("init", {}).get("parent")
            or info.get("pretrain", {}).get("parent")
        )
        if not parent_path:
            break
        cur = Path(parent_path).resolve().parent

    chain.reverse()
    return chain


def render(chain: list[dict]) -> str:
    if not chain:
        return "(empty chain — no info.toml found at the given path)"

    rows: list[list[str]] = []
    cumulative = 0
    cumulative_known = True

    for seg in chain:
        steps = seg.get("steps_trained")

        if isinstance(steps, int):
            seg_start = cumulative
            cumulative += steps
            range_label = f"[{seg_start:>11,}, {cumulative:>11,}]"
        else:
            cumulative_known = False
            range_label = "[?, ?]" if not isinstance(steps, str) else f"[?, {steps}]"

        cells = [range_label, str(seg["dir"])]
        for paths, _, fmt in DISPLAYED_KEYS:
            v = _resolve_first(seg["config"], paths)
            cells.append("-" if v is None else fmt.format(v))
        rows.append(cells)

    headers = ["step range", "trial dir", *(label for _, label, _ in DISPLAYED_KEYS)]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    fmt_row = "  ".join(f"{{:<{w}}}" for w in widths)

    out = [fmt_row.format(*headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append(fmt_row.format(*r))
    out.append("")
    out.append(
        f"Total steps: {cumulative:,}"
        if cumulative_known
        else f"Total steps: {cumulative:,}+ (some segments missing steps_trained)"
    )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Walk a trial's pretrain ancestry chain.",
    )
    p.add_argument(
        "trial_dir",
        help="Path to a trial dir (runs/<run>/<trial>) or promoted model dir (models/<name>).",
    )
    args = p.parse_args()
    chain = walk_chain(Path(args.trial_dir))
    print(render(chain))


if __name__ == "__main__":
    main()
