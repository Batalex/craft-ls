"""Parser-validator core logic."""

import json
import logging
import re
from collections import deque
from importlib.resources import files
from itertools import tee
from textwrap import shorten
from typing import Iterable, cast

import yaml
from jsonschema import ValidationError
from jsonschema.exceptions import best_match
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from lsprotocol import types as lsp
from yaml.events import (
    DocumentEndEvent,
    Event,
    MappingEndEvent,
    MappingStartEvent,
    SequenceEndEvent,
    SequenceStartEvent,
    StreamEndEvent,
)
from yaml.scanner import ScannerError
from yaml.tokens import (
    BlockEndToken,
    BlockMappingStartToken,
    BlockSequenceStartToken,
    ScalarToken,
    Token,
    ValueToken,
)

from craft_ls.types_ import (
    CompleteScan,
    IncompleteScan,
    ScanResult,
    Schema,
    YamlDocument,
)

SOURCE = "craft-ls"
FILE_TYPES = ["snapcraft", "rockcraft", "charmcraft"]
MISSING_DESC = "No description to display"
SIZE = 79
DEFAULT_RANGE = lsp.Range(
    start=lsp.Position(line=0, character=0),
    end=lsp.Position(line=0, character=0),
)

logger = logging.getLogger(__name__)

validators: dict[str, Validator] = {}
for file_type in FILE_TYPES:
    schema = json.loads(
        files("craft_ls.schemas").joinpath(f"{file_type}.json").read_text()
    )
    validators[file_type] = validator_for(schema)(schema)


def scan_for_tokens(instance_document: str) -> ScanResult:
    """Scan the document for yaml tokens."""
    tokens = []
    tokens_iter = yaml.scan(instance_document)

    try:
        for event in tokens_iter:
            tokens.append(event)
    except ScannerError:
        instance = robust_load(instance_document)
        return IncompleteScan(tokens=tokens, instance=instance)

    instance = cast(YamlDocument, yaml.safe_load(instance_document))
    return CompleteScan(tokens=tokens, instance=instance)


def robust_load(instance_document: str) -> YamlDocument:
    """Parse the valid portion of the stream and construct a Python object."""
    events = []
    events_iter = yaml.parse(instance_document)

    closing_sequence: deque[Event] = deque()

    try:
        for event in events_iter:
            match event:
                case MappingStartEvent():
                    closing_sequence.append(MappingEndEvent())
                case SequenceStartEvent():
                    closing_sequence.append(SequenceEndEvent())
                case MappingEndEvent():
                    closing_sequence.pop()
                case SequenceEndEvent():
                    closing_sequence.pop()

            events.append(event)
    except ScannerError:
        closing_sequence.extendleft([DocumentEndEvent(), StreamEndEvent()])

    truncated_file = yaml.emit(events + list(reversed(closing_sequence)))
    return cast(YamlDocument, yaml.safe_load(truncated_file))


def get_diagnostics(
    validator: Validator, instance_document: str
) -> list[lsp.Diagnostic]:
    """Validate a document against its schema."""
    scanned_tokens = scan_for_tokens(instance_document)
    tokens = list(scanned_tokens.tokens)
    diagnostics = []

    for error in validator.iter_errors(scanned_tokens.instance):
        if error.context:
            error = cast(ValidationError, best_match(error.context))

        match error:
            case ValidationError(
                validator="additionalProperties", path=path, message=message
            ):
                ranges = [DEFAULT_RANGE]
                pattern = r"\((?P<keys>.*) (was|were) unexpected\)"
                if (match := re.search(pattern, message or "")) and (
                    keys := match.group("keys")
                ):
                    keys_cleaned = [key.strip(" '") for key in keys.split(",")]
                    ranges = [
                        get_faulty_token_range(tokens, list(path) + [key])
                        for key in keys_cleaned
                    ]

                for range_ in ranges:
                    diagnostics.append(
                        lsp.Diagnostic(
                            message=shorten(message, SIZE),
                            severity=lsp.DiagnosticSeverity.Error,
                            range=range_,
                            source=SOURCE,
                        )
                    )

            case ValidationError(validator="required", path=path, message=message):
                range_ = get_faulty_token_range(tokens, path) if path else DEFAULT_RANGE

                diagnostics.append(
                    lsp.Diagnostic(
                        message=shorten(message, SIZE),
                        severity=lsp.DiagnosticSeverity.Error,
                        range=range_,
                        source=SOURCE,
                    )
                )

            case ValidationError(path=path, message=str(message), schema={**schema}):
                # The sub-error might have a path we should highlight
                path = deque(cast(Iterable[str], schema.get("err_path", path)))
                range_ = get_faulty_token_range(tokens, path) if path else DEFAULT_RANGE
                message = str(schema.get("err_msg", message))
                range_ = get_faulty_token_range(tokens, path)

                diagnostics.append(
                    lsp.Diagnostic(
                        message=shorten(message, SIZE),
                        severity=lsp.DiagnosticSeverity.Error,
                        range=range_,
                        source=SOURCE,
                    )
                )

            case error:
                # yet to implement
                logger.debug(error.message)

    return diagnostics


def peek(tee_iterator: Iterable[Token]) -> Token | None:
    """Return the next value without moving the input forward."""
    [forked_iterator] = tee(tee_iterator, 1)
    return next(forked_iterator, None)


def get_faulty_token_range(
    tokens: list[Token], path_segments: Iterable[str | int]
) -> lsp.Range:
    """Link the validation error to the position in the original document."""
    target_level: int | None
    segment: str | int | None

    path_iterator = iter(enumerate(path_segments))
    target_level, segment = next(path_iterator)
    # We keep track of the nested elements by incrementing/decrementing the level
    # every time we encounted a block token. The very start of the document
    # counts as a mapping, hence the -1 offset
    current_level = -1

    # Create a peekable iterator
    [token_iterator] = tee(tokens, 1)

    for token in token_iterator:
        match token:
            case BlockMappingStartToken() | BlockSequenceStartToken():
                current_level += 1

            case BlockEndToken():
                current_level -= 1

            case ScalarToken(value=value) if value == segment:
                nested_level_mismatch = current_level != target_level
                is_not_key = not isinstance(peek(token_iterator), ValueToken)

                if nested_level_mismatch or is_not_key:
                    continue

                target_level, segment = next(path_iterator, (None, None))
                if segment is not None:
                    continue

                else:  # found our culprit
                    # TODO(ux): flag up to the next token/block?
                    range = lsp.Range(
                        start=lsp.Position(
                            line=token.start_mark.line,
                            character=token.start_mark.column,
                        ),
                        end=lsp.Position(
                            line=token.end_mark.line, character=token.end_mark.column
                        ),
                    )
                    return range

            # TODO(array)

    return DEFAULT_RANGE


def get_description_from_path(path: Iterable[str | int], schema: Schema) -> str:
    """Given an element path, get its description."""
    sub = schema
    for segment in path:
        sub = sub.get("properties", {}).get(segment, {})

    return str(sub.get("description", MISSING_DESC))


def get_schema_path_from_token_position(
    position: lsp.Position, instance_document: str
) -> deque[str] | None:
    """Parse the document to find the path to the current position."""
    scanned_tokens = scan_for_tokens(instance_document)
    current_path: deque[str] = deque()
    last_scalar_token: str = ""
    start_mark: yaml.Mark
    end_mark: yaml.Mark

    for token in scanned_tokens.tokens:
        match token:
            case BlockMappingStartToken() | BlockSequenceStartToken():
                current_path.append(last_scalar_token)

            case BlockEndToken():
                current_path.pop()

            case ScalarToken(value=value, start_mark=start_mark, end_mark=end_mark):
                is_line_matching = start_mark.line == position.line
                is_col_matching = (
                    start_mark.column <= position.character <= end_mark.column
                )
                if is_line_matching and is_col_matching:
                    current_path.append(value)
                    current_path.remove("")
                    return current_path

                else:
                    last_scalar_token = value

            case _:
                continue
    return None
