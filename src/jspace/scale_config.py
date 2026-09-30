"""Frozen configuration helpers for the version-2 and version-3 scale extensions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "extension" / "config.json"


def resolved_config_path() -> Path:
    """The frozen v2 file unless JSPACE_SCALE_CONFIG names another committed registration."""
    override = os.environ.get("JSPACE_SCALE_CONFIG")
    if not override:
        return CONFIG_PATH
    path = Path(override)
    return path if path.is_absolute() else REPO_ROOT / path


def load_scale_config(path: Path | None = None) -> dict:
    path = path or resolved_config_path()
    with path.open() as handle:
        config = json.load(handle)
    if config.get("schema_version") != 2:
        raise ValueError("unsupported scale-extension configuration")
    return config


def config_sha256(path: Path | None = None) -> str:
    path = path or resolved_config_path()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmatory_profiles(config: dict | None = None) -> tuple[str, ...]:
    config = config or load_scale_config()
    models = config["models"]
    return tuple(models["primary"] + models["secondary"])


def transferred_band(n_layers: int, config: dict | None = None) -> tuple[int, int]:
    """Map the fixed 4B band as a half-open depth interval, then return inclusive.

    The source band 14..19 is represented as [14, 20). Both boundaries are
    multiplied by target/source depth and floored. This maps 36-layer 8B to
    14..19 and 40-layer 14B to 15..21 without inspecting either model's results.
    """
    config = config or load_scale_config()
    intervention = config["intervention"]
    source_layers = intervention["source_layers"]
    first, last = intervention["source_band_inclusive"]
    start = first * n_layers // source_layers
    stop = (last + 1) * n_layers // source_layers
    if not 0 <= start < stop <= n_layers:
        raise ValueError(f"invalid transferred band for {n_layers} layers")
    return start, stop - 1
