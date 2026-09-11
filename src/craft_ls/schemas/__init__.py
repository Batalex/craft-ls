"""Jsonschemas versions."""

from __future__ import annotations

import json
from importlib.resources import files

SCHEMA_VERSIONS: dict[str, str] = json.loads(
    files("craft_ls.schemas").joinpath("versions.json").read_text(encoding="utf-8")
)
