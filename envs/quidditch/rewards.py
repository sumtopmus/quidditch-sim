"""Canonical Quidditch reward magnitudes — single source of truth.

Physical / geometric thresholds that *gate* reward events
(TAG_RADIUS, CRASH_VEL_THR, TAG_COOLDOWN_SECONDS, …) live in constants.py.
Only the per-event reward / penalty magnitudes (and the shaping coefficient
that ties them together) live here.

Sign convention: each constant carries its own sign.  A *_REWARD constant is
positive (gain to the recipient); a *_PENALTY constant is negative (loss to
the recipient).  This avoids the legacy `+= X / -= X` sign-flip pattern and
lets the two sides of a zero-sum event be tuned independently if needed.
"""

from __future__ import annotations

# ── Shared (simple_env + team_env) ──────────────────────────────────────────
SCORE_REWARD:           float = 10.0  # drone scores through the hoop
CRASH_PENALTY:          float = -20.0  # drone hits wall, floor, or goes OOB
DIST_REWARD_SCALE:      float = 0.01  # multiplier on −dist/ARENA_RADIUS shaping

# ── Team-only (team_env) ────────────────────────────────────────────────────
# Soft-tag (proximity-based): zero-sum between blue (defender) and red (attacker).
TAG_ENTRY_REWARD:       float = 5.0  # one-shot pulse on first entry into zone
TAG_DURATION_REWARD:    float = 0.02  # per simulation step while in zone

# Take-down: drone-drone collision with |v_rel| > CRASH_VEL_THR.
# Currently zero-sum (same magnitude, opposite signs); split into two named
# constants so they can diverge later without touching team_env's reward code.
TAKE_DOWN_REWARD:       float = 20.0  # aggressor (the drone that took the other down)
TAKE_DOWN_PENALTY:      float = -20.0  # victim (the drone that was taken down)

# Defender shaping/tactics: target midpoint = α·red_pos + (1−α)·hoop_center.
# Lives here (not in constants.py) because it parameterises a reward term,
# not the scene geometry.
DEFAULT_MIDPOINT_ALPHA: float = 0.5
