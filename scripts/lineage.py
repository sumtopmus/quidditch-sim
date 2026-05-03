"""Walk a trial's pretrain ancestry and print one row per training segment.

Each row shows the cumulative step range, the trial dir, and a few key config
knobs that typically vary across pretrain steps (lr, ent_coef, randomise_start,
episode_seconds).  The full per-trial config lives in `<trial>/config_snapshot.toml`
(or `config.toml` for promoted models).

Usage:
    python scripts/lineage.py runs/ppo_hoop_rand_start/20260430_234354
    python scripts/lineage.py models/ppo_hoop_rand_start_20260430_234354
    make lineage RUN_NAME=ppo_hoop_rand_start          # latest trial in that run
    make lineage TRIAL=20260430_234354 RUN_NAME=...    # specific trial

The chain is reconstructed by reading `[pretrain].parent` from each trial's
info.toml (or run_info.toml in promoted dirs) and recursing.  If a parent dir
is missing the chain is truncated with a warning.
"""

import argparse
import sys
import tomllib
from pathlib import Path


# (tuple-of-dotted-fallback-paths, column-header, format-string)
# Older snapshots used flat `[ppo]`; current ones use nested `[training.ppo]` —
# tuple lets us try both.  Env settings come from env_snapshot.toml (merged in).
DISPLAYED_KEYS = [
    (("training.ppo.lr",       "ppo.lr"),       "lr",         "{:.1e}"),
    (("training.ppo.ent_coef", "ppo.ent_coef"), "ent_coef",   "{:.2g}"),
    (("env.randomise_start",),                  "rand_start", "{}"),
    (("env.episode_seconds",),                  "episode_s",  "{:.1f}"),
]


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
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
    """Walk pretrain ancestry from `start` backwards. Returns oldest-first."""
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

        info_path = _find_first(cur, ("info.toml", "run_info.toml"))
        cfg_path = _find_first(cur, ("config_snapshot.toml", "config.toml"))
        env_path = _find_first(cur, ("env_snapshot.toml", "env.toml"))
        info = _load_toml(info_path) if info_path else {}
        cfg = _load_toml(cfg_path) if cfg_path else {}
        # Older trial layouts split env into a separate snapshot; merge so
        # _resolve_first can find env.* keys with one lookup.
        if env_path:
            env_data = _load_toml(env_path)
            if "env" in env_data:
                cfg.setdefault("env", {}).update(env_data["env"])

        chain.append({"dir": cur, "info": info, "config": cfg})

        parent_path = info.get("pretrain", {}).get("parent")
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
        run = seg["info"].get("run", {})
        steps = run.get("steps_trained")

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
