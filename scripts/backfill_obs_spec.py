"""One-off: write [obs] blocks into the seven legacy run_info.toml files in
models/, so they participate in the obs-spec compatibility check.

The mapping is hand-written because the body-mixed → world-frame opp_vel_rel
change happened mid-Phase-2b without a saved-config flag, so two team-mode
runs from the same training script have different specs.  Hand-written = auditable.

Usage:
    python scripts/backfill_obs_spec.py [--dry-run]

Idempotent: refuses to overwrite an existing [obs] block.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow imports from repo root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.quidditch import obs_spec as obs_spec_module
from scripts._train_common import format_obs_block


# run_dir name → (spec attribute name on obs_spec module, n_stack)
LEGACY_SPECS: dict[str, tuple[str, int]] = {
    "ppo_hoop_fixed_start_20260430_224234": ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_fixed_start_20260504_023051": ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_rand_start_20260430_234354":  ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_rand_start_20260505_174509":  ("SIMPLE_ENV_OBS", 1),
    "ppo_hoop_red_1_20260506_103058":       ("TEAM_ENV_OBS",   1),
    "ppo_hoop_blue_1_20260507_194423":      ("TEAM_ENV_OBS",   1),
    "ppo_hoop_blue_4_20260511_202612":      ("AUGMENTED_OBS",  3),
}


def backfill_one(info_path: Path, *, run_dir_name: str) -> str:
    """Append the appropriate [obs] block to info_path and return the text appended.

    Raises:
      RuntimeError if info_path already contains an [obs] block.
      KeyError    if run_dir_name has no entry in LEGACY_SPECS.
    """
    existing = info_path.read_text()
    if "\n[obs]" in existing or existing.startswith("[obs]"):
        raise RuntimeError(f"{info_path} already has [obs] block")

    spec_attr, n_stack = LEGACY_SPECS[run_dir_name]  # KeyError on miss
    spec = getattr(obs_spec_module, spec_attr)
    block_text = format_obs_block(spec, n_stack)
    appended = (
        f"\n# back-filled by scripts/backfill_obs_spec.py on "
        f"{date.today().isoformat()}\n"
        f"{block_text.lstrip()}"
    )
    info_path.write_text(existing.rstrip() + "\n" + appended)
    return appended


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without modifying files.")
    p.add_argument("--models-dir", default="models",
                   help="Directory containing promoted runs (default: models)")
    args = p.parse_args(argv)

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        print(f"ERROR: {models_dir} is not a directory", file=sys.stderr)
        return 2

    summary: list[tuple[str, str]] = []  # (run_dir_name, status)
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = run_dir / "run_info.toml"
        if not info.exists():
            summary.append((run_dir.name, "SKIP (no run_info.toml)"))
            continue
        if run_dir.name not in LEGACY_SPECS:
            summary.append((run_dir.name, "SKIP (not in LEGACY_SPECS)"))
            continue
        try:
            if args.dry_run:
                if "\n[obs]" in info.read_text():
                    summary.append((run_dir.name, "ALREADY HAS [obs]"))
                else:
                    spec_attr, n_stack = LEGACY_SPECS[run_dir.name]
                    summary.append((run_dir.name,
                                    f"WOULD BACK-FILL: {spec_attr} n_stack={n_stack}"))
            else:
                backfill_one(info, run_dir_name=run_dir.name)
                spec_attr, n_stack = LEGACY_SPECS[run_dir.name]
                summary.append((run_dir.name,
                                f"BACK-FILLED: {spec_attr} n_stack={n_stack}"))
        except RuntimeError:
            summary.append((run_dir.name, f"SKIP (already has [obs])"))

    print("\nBack-fill summary:")
    print("-" * 70)
    for name, status in summary:
        print(f"  {name:<50}  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
