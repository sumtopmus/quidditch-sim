"""Centralized Ray initialization for the project.

The macOS libomp double-init guard (KMP_DUPLICATE_LIB_OK) must reach every
Ray worker process — env-runners and learners run in separate processes that
do NOT inherit a module-level os.environ set in the driver. We propagate it
explicitly via runtime_env.
"""
from __future__ import annotations

import os

import ray

# Driver-side guard (mirrors tests/conftest.py and scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WORKER_ENV_VARS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    # MuJoCo must render off-screen inside headless workers; keep GL out of
    # env-runners entirely (rendering happens in a dedicated eval worker).
    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
}


def ray_init_for_project(**kwargs) -> None:
    """Idempotent ray.init with the project's worker env-var propagation."""
    if ray.is_initialized():
        return
    runtime_env = {"env_vars": dict(_WORKER_ENV_VARS)}
    ray.init(runtime_env=runtime_env, ignore_reinit_error=True, **kwargs)
