"""Types module."""

from dataclasses import dataclass
from typing import Any, NewType

from yaml import Token

# We can probably do better
YamlDocument = NewType("YamlDocument", dict[str, Any])
Schema = NewType("Schema", dict[str, Any])


@dataclass
class ScanResult:
    """Token scan result container."""

    tokens: list[Token]
    instance: YamlDocument


@dataclass
class CompleteScan(ScanResult):
    """Indicate a complete scan of the document."""

    pass


@dataclass
class IncompleteScan(ScanResult):
    """Indicate an incomplete scan of the document."""

    pass
