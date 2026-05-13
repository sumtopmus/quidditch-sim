"""Hydra-instantiable env factories.

`SimpleEnvFactory` and `TeamEnvFactory` are populated in Phase 3.  This file
exists in Phase 1 so `_target_: envs.quidditch.env_factories.SimpleEnvFactory`
in the conf/env/*.yaml files (added in Phase 4) resolves to *something* import-
able even before factories exist.
"""
from __future__ import annotations
