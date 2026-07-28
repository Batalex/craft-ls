"""Types module."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Generator, NamedTuple, NewType, TypeAlias

from jsonschema import ValidationError, Validator
from tree_sitter import Tree

# We can probably do better, but that will do for now
YamlDocument = NewType("YamlDocument", dict[str, Any])
Schema = NewType("Schema", dict[str, Any])


class IndexEntry(NamedTuple):
    """Document index entry."""

    tree: Tree
    validator: Validator | None
    instance: YamlDocument
    text: str
    version: int | None


DocumentsIndex: TypeAlias = dict[str, IndexEntry]


@dataclass
class MissingTypeCharmcraftValidator:
    """No op implementation.

    Used if charmcraft.yaml is missing the 'type' key or is set to 'bundle'.
    """

    schema: Any

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


MISSING_TYPE_MSG = "Missing or unsupported 'base' and/or 'build-base' key(s)."


@dataclass
class MissingTypeSnapcraftValidator:
    """No op implementation.

    Used if snapcraft.yaml is:
    - missing the 'base' or 'build-base' key
    - using an unsupported base
    """

    schema: Any

    def iter_errors(
        self, instance: Any, _schema: Any = None
    ) -> Generator[ValidationError, None, None]:
        """Lazily yield each of the validation errors in the given instance."""
        yield ValidationError(
            validator="required",
            path=deque([]),
            message=MISSING_TYPE_MSG,
            schema={},
        )
