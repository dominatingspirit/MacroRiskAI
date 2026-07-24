"""IO and configuration helpers.

Centralizes config loading and path resolution so no module hardcodes
filesystem layout. All paths in ``config.yaml`` are resolved relative to the
project root, which is inferred as two levels above this file unless the config
overrides it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


# Project root = <repo>/  (this file lives at <repo>/src/utils/io.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration into a plain dict.

    Parameters
    ----------
    config_path:
        Optional explicit path. Defaults to ``config/config.yaml`` at the
        project root.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a config-relative path against the project root."""
    base = Path(root).resolve() if root else PROJECT_ROOT
    p = Path(relative)
    return p if p.is_absolute() else (base / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Serialize ``obj`` to JSON, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return p


def write_text(text: str, path: str | Path) -> Path:
    """Write UTF-8 text to ``path``, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def _json_default(o: Any) -> Any:
    """Fallback serializer for numpy/pandas scalar types."""
    # Handle numpy scalars and other objects that expose .item()/.tolist().
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
