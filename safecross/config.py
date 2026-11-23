"""Load the YAML configuration so every script uses the same settings."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read configs/config.yaml (or another file) and apply optional overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _merge(cfg, overrides or {})


def resolve(path: str | Path) -> Path:
    """Turn a path from the config into an absolute path inside the project."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
