from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve paths relative to the repository."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_config_path"] = str(path)
    config["_repo_root"] = str(path.parent.parent)
    return config


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def merge_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides parsed as YAML scalars."""
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot override nested key {dotted_key!r}")
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_repo_root"]) / path

