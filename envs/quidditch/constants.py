"""Canonical Quidditch arena constants — single source of truth.

Anything that needs to know "where is the hoop" or "how big is
the arena" should import from here.
"""

from __future__ import annotations

import numpy as np


# ── Arena ────────────────────────────────────────────────────────────────────
ARENA_RADIUS: float = 3.0  # m (6 m diameter)
ARENA_WALL_HEIGHT: float = 4.5  # m

# ── Hoop geometry ────────────────────────────────────────────────────────────
# Vertical ring at x=2, y=0, z=2; outward normal points along +x (away from
# arena centre).  Drones score by crossing the ring from -x to +x.
HOOP_CENTER         = np.array([2.0, 0.0, 2.0], dtype=np.float64)
HOOP_OUTWARD_NORMAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
HOOP_DIAMETER: float = 0.5  # m
HOOP_RADIUS:   float = HOOP_DIAMETER / 2.0  # 0.25 m

# ── Scoring trigger volume ───────────────────────────────────────────────────
# Half-length (along the hoop's outward normal) of the cylinder used for
# mj_geomDistance-based hoop scoring.  See decisions.md 2026-04-30 for the ADR.
HOOP_SCORE_TUBE_HALF_LEN: float = 0.1  # ± m around the hoop plane

# ── Team-play (Phase 2) ──────────────────────────────────────────────────────
# Soft-tag: Blue scoring on proximity to Red.
TAG_RADIUS: float = 0.3                # m, radius of the tag sphere on each drone
TAG_COOLDOWN_SECONDS: float = 1.0      # post-exit gate on the tag-entry pulse

# Crashes: drone-vs-{drone, wall} contact magnitudes.  At |v_rel| > threshold
# the contact is treated as a decisive crash; below threshold contacts apply
# physics but do not terminate the episode.
CRASH_VEL_THR: float = 1.5             # m/s, |v_rel · contact_normal| threshold

# Default Blue start (hovering 1 m in front of the hoop, slightly below hoop height).
BLUE_START_POS = np.array([1.0, 0.0, 1.5], dtype=np.float64)
BLUE_START_YAW: float = float(np.pi)   # facing arena center (−x direction)

# Reward magnitudes (per-event) live in envs/quidditch/rewards.py — they're
# parameters of the RL objective, not of the scene.
