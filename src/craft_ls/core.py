"""Parser-validator core logic."""

import json
import logging
import re
from dataclasses import dataclass
from importlib.resources import read_text
from typing import Iterable

import yaml
from jsonschema import ValidationError, Validator
from jsonschema.validators import validator_for
from lsprotocol import types
from yaml.events import (
    DocumentEndEvent,
    Event,
    MappingEndEvent,
    MappingStartEvent,
    StreamEndEvent,
)
from yaml.scanner import ScannerError
from yaml.tokens import BlockEndToken, BlockMappingStartToken, ScalarToken, Token

validators: dict[str, Validator] = {}
for file_type in ["snapcraft", "rockcraft"]:
    schema = json.loads(read_text("craft_ls.schemas", f"{file_type}.json"))
    validators[file_type] = validator_for(schema)(schema)

logger = logging.getLogger(__name__)


DEFAULT_RANGE = types.Range(
    start=types.Position(line=0, character=0),
    end=types.Position(line=0, character=0),
)


@dataclass
class ScanResult:
    """Token scan result container."""

    tokens: list[Token]


@dataclass
class CompleteScan(ScanResult):
    """Indicate a complete scan of the document."""

    pass


@dataclass
class IncompleteScan(ScanResult):
    """Indicate an incomplete scan of the document."""

    pass


def scan_for_tokens(instance_document: str) -> ScanResult:
    """Scan the document for yaml tokens."""
    tokens = []
    tokens_iter = yaml.scan(instance_document)

    try:
        for event in tokens_iter:
            tokens.append(event)
    except ScannerError:
        return IncompleteScan(tokens=tokens)

    return CompleteScan(tokens=tokens)


def robust_load(instance_document: str) -> dict:
    """Parse the valid portion of the stream and construct a Python object."""
    events = []
    events_iter = yaml.parse(instance_document)
    current_level = 0

    try:
        for event in events_iter:
            match event:
                case MappingStartEvent():
                    current_level += 1
                case MappingEndEvent():
                    current_level -= 1

            events.append(event)
    except ScannerError:
        pass

    end_of_sequence: list[Event] = [MappingEndEvent()] * current_level
    end_of_sequence.extend([DocumentEndEvent(), StreamEndEvent()])
    truncated_file = yaml.emit(events + end_of_sequence)
    return yaml.safe_load(truncated_file)


def get_diagnostics(
    validator: Validator, instance_document: str
) -> list[types.Diagnostic]:
    """Validate a document against its schema."""
    instance = {}
    match scanned_tokens := scan_for_tokens(instance_document):
        case CompleteScan():
            instance = yaml.safe_load(instance_document)

        case IncompleteScan():
            instance = robust_load(instance_document)

    tokens = list(scanned_tokens.tokens)
    diagnostics = []

    for error in validator.iter_errors(instance):
        match error:
            case ValidationError(
                validator="additionalProperties", path=path, message=message
            ):
                range_ = DEFAULT_RANGE
                pattern = r"\('(?P<key>.*)' was unexpected\)"
                if key := re.search(pattern, message).group("key"):
                    range_ = get_faulty_token_range(tokens, list(path) + [key])

                diagnostics.append(
                    types.Diagnostic(
                        message=f"{message}",
                        severity=types.DiagnosticSeverity.Error,
                        range=range_,
                    )
                )

            case ValidationError(validator="required", path=path, message=message):
                range_ = get_faulty_token_range(tokens, path) if path else DEFAULT_RANGE

                diagnostics.append(
                    types.Diagnostic(
                        message=f"{message}",
                        severity=types.DiagnosticSeverity.Error,
                        range=range_,
                    )
                )

            case ValidationError(path=path, message=message) if path:
                range_ = get_faulty_token_range(tokens, path)

                diagnostics.append(
                    types.Diagnostic(
                        message=f"{message}",
                        severity=types.DiagnosticSeverity.Error,
                        range=range_,
                    )
                )

            case error:
                # yet to implement
                logger.debug(error.message)

    return diagnostics


def get_faulty_token_range(tokens: list[Token], path_segments: Iterable) -> types.Range:
    """Link the validation error to the position in the original document."""
    path_iterator = iter(enumerate(path_segments))
    target_level, segment = next(path_iterator)
    # We keep track of the nested elements by incrementing/decrementing the level
    # every time we encounted a block token. The very start of the document
    # counts as a mapping, hence the -1 offset
    current_level = -1

    for token in tokens:
        match token:
            case BlockMappingStartToken():
                current_level += 1

            case BlockEndToken():
                current_level -= 1

            case ScalarToken(value=value) if value == segment:
                if current_level != target_level:
                    continue

                target_level, segment = next(path_iterator, (None, None))
                if segment is not None:
                    continue

                else:  # found our culprit
                    # TODO(ux): flag the next token/block?
                    range = types.Range(
                        start=types.Position(
                            line=token.start_mark.line,
                            character=token.start_mark.column,
                        ),
                        end=types.Position(
                            line=token.end_mark.line, character=token.end_mark.column
                        ),
                    )
                    return range

            # TODO(array)

    return DEFAULT_RANGE
