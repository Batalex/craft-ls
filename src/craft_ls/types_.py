"""Types module."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Generator, NewType

from jsonschema import ValidationError, Validator
from lsprotocol import types as lsp
from yaml import CollectionNode, Mark, Token

# We can probably do better, but that will do for now
YamlDocument = NewType("YamlDocument", dict[str, Any])
Schema = NewType("Schema", dict[str, Any])


@dataclass
class ParsedResult:
    """Token parsed result container."""

    tokens: list[Token]
    instance: YamlDocument
    nodes: CollectionNode


@dataclass
class CompleteParsedResult(ParsedResult):
    """Indicate a complete document parsing."""

    pass


@dataclass
class IncompleteParsedResult(ParsedResult):
    """Indicate an incomplete parsing of the document."""

    pass


@dataclass
class DocumentNode:
    """Document node."""

    value: str
    start: Mark
    end: Mark
    selection_end: Mark

    def contains(self, position: lsp.Position) -> bool:
        """Is position contained in node range?"""
        range_start_before = self.start.line < position.line or (
            self.start.line == position.line and self.start.column <= position.character
        )

        range_end_after = self.selection_end.line > position.line or (
            self.selection_end.line == position.line
            and self.selection_end.column >= position.character
        )

        return range_start_before and range_end_after


@dataclass
class IndexEntry:
    """Document index entry."""

    validator: Validator
    tokens: list[Token]
    instance: YamlDocument
    segments: dict[tuple[str, ...], DocumentNode]
    version: int | None


class MissingTypeCharmcraftValidator:
    """No op implementation.

    Used if charmcraft.yaml is missing the 'type' key or is set to 'bundle'.
    """

    def iter_errors(
        self, instance: Any, _schema: Any = None
    ) -> Generator[ValidationError, None, None]:
        """Lazily yield each of the validation errors in the given instance."""
        yield ValidationError(
            validator="required",
            path=deque([]),
            message="'type' key is mandatory and must be 'charm'",
            schema={},
        )


class MissingTypeSnapcraftValidator:
    """No op implementation.

    Used if snapcraft.yaml is missing the 'base' or 'build-base' key.
    """

    def iter_errors(
        self, instance: Any, _schema: Any = None
    ) -> Generator[ValidationError, None, None]:
        """Lazily yield each of the validation errors in the given instance."""
        yield ValidationError(
            validator="required",
            path=deque([]),
            message="Filling 'base' and/or 'build-base' key(s) is mandatory.",
            schema={},
        )
