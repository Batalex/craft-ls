"""Types module."""

from dataclasses import dataclass
from typing import Any, NewType

from yaml import Token

# We can probably do better
YamlDocument = NewType("YamlDocument", dict[str, Any])
Schema = NewType("Schema", dict[str, Any])


@dataclass
class ParsedResult:
    """Token parsed result container."""

    tokens: list[Token]
    instance: YamlDocument


@dataclass
class CompleteParsedResult(ParsedResult):
    """Indicate a complete document parsing."""

    pass


@dataclass
class IncompleteParsedResult(ParsedResult):
    """Indicate an incomplete parsing of the document."""

    pass
