"""Tiny helper to load and cache the per-source schema maps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCHEMA_MAPS_DIR = Path(__file__).parent / "schema_maps"


@lru_cache(maxsize=8)
def load_schema_map(name: str) -> dict[str, Any]:
    """Load a schema map by name (e.g. 'profiles', 'demands', 'jds')."""
    path = SCHEMA_MAPS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Schema map not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
