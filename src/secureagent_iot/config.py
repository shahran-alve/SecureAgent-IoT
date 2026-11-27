"""Configuration loading and seeding helpers."""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def stable_seed(*parts: Any) -> int:
    """Seed derived from the parts with SHA-256 (Python's hash() is salted)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def rng_for(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))
