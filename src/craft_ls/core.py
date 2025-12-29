"""Parser-validator core logic."""

import logging
import re
from collections import deque
from importlib.resources import files
from itertools import chain, tee
from typing import Iterable, cast

import jsonref
import yaml
from jsonpath_ng import parse
from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import relevance
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from lsprotocol import types as lsp
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
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
    CompleteParsedResult,
    IncompleteParsedResult,
    MissingTypeCharmcraftValidator,
    MissingTypeSnapcraftValidator,
    Node,
    ParsedResult,
    Schema,
    YamlDocument,
)

SOURCE = "craft-ls"
FILE_TYPES = ["snapcraft", "rockcraft", "charmcraft"]
MISSING_DESC = "No description to display"
DEFAULT_RANGE = lsp.Range(
    start=lsp.Position(line=0, character=0),
    end=lsp.Position(line=0, character=0),
)
SPECIAL_SYMBOL_PARENTS = {"parts", "apps", "services"}

logger = logging.getLogger(__name__)

default_validators: dict[str, Validator] = {}
charmcraft_registry: Registry
for file_type in FILE_TYPES:
    schema_str = files("craft_ls.schemas").joinpath(f"{file_type}.json").read_text()
    schema = jsonref.loads(
        files("craft_ls.schemas").joinpath(f"{file_type}.json").read_text()
    )
    default_validators[file_type] = validator_for(schema)(schema)

    if file_type == "charmcraft":
        schema = Resource.from_contents(
            jsonref.loads(schema_str), default_specification=DRAFT202012
        )
        charmcraft_registry = schema @ Registry()

    if file_type == "snapcraft":
        schema = Resource.from_contents(
            jsonref.loads(schema_str), default_specification=DRAFT202012
        )
        snapcraft_registry = schema @ Registry()


def get_validator_and_parse(  # noqa: C901
    file_stem: str, instance_document: str
) -> tuple[Validator, ParsedResult] | None:
    """Get the most appropriate validator for the current document."""
    if file_stem not in FILE_TYPES:
        return None

    scanned_tokens = parse_tokens(instance_document)

    if file_stem == "rockcraft":
        return default_validators[file_stem], scanned_tokens

    elif file_stem == "snapcraft":
        base = scanned_tokens.instance.get("base", None)
        build_base = scanned_tokens.instance.get("build-base", None)
        validator: Draft202012Validator | MissingTypeSnapcraftValidator
        match base, build_base:
            case "core22", _:
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:core22")
                    .contents
                )
            case "core24", _:
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:core24")
                    .contents
                )
            case "bare", "core22":
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:bare22")
                    .contents
                )
            case "bare", "core24":
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:bare24")
                    .contents
                )
            case _, "core22":
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:base22")
                    .contents
                )
            case _, "core24":
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:base24")
                    .contents
                )
            case _, "devel":
                validator = Draft202012Validator(
                    schema=snapcraft_registry.resolver()
                    .lookup("urn:snapcraft:devel")
                    .contents
                )

            case _:
                validator = MissingTypeSnapcraftValidator()

        return cast(Validator, validator), scanned_tokens

    else:
        # by elimination, file_stem is charmcraft
        if scanned_tokens.instance.get("type") != "charm":
            return cast(Validator, MissingTypeCharmcraftValidator()), scanned_tokens

        validator = Draft202012Validator(
            schema=charmcraft_registry.resolver()
            .lookup("urn:charmcraft:platformcharm")
            .contents
        )
    return cast(Validator, validator), scanned_tokens


def parse_tokens(instance_document: str) -> ParsedResult:
    """Scan the document for yaml tokens."""
    tokens = []
    tokens_iter = yaml.scan(instance_document)

    try:
        for event in tokens_iter:
            tokens.append(event)
    except ScannerError:
        instance, events = robust_load(instance_document)
        nodes = yaml.compose(events)
        return IncompleteParsedResult(tokens=tokens, instance=instance, nodes=nodes)

    instance = cast(YamlDocument, yaml.safe_load(instance_document))
    nodes = yaml.compose(instance_document)
    return CompleteParsedResult(tokens=tokens, instance=instance, nodes=nodes)


def robust_load(instance_document: str) -> tuple[YamlDocument, list[Event]]:
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
    return cast(YamlDocument, yaml.safe_load(truncated_file)), truncated_file


def segmentize_nodes(root: yaml.CollectionNode) -> list[tuple[tuple[str, ...], Node]]:
    """Flatten graph into path segments."""
    segments: list[tuple[tuple[str, ...], Node]] = []
    nodes = list(root.value)

    for node_pair in nodes:
        segments.extend(_do_segmentize_nodes(*node_pair))

    return segments


def _do_segmentize_nodes(
    first: yaml.CollectionNode,
    second: yaml.CollectionNode,
    prefix: tuple[str, ...] | None = None,
) -> list[tuple[tuple[str, ...], Node]]:
    """Recursive node segmentation.

    Craft tools don't usually go over three levels, so we don't have to worry about recursion limits.
    """
    segments = []
    prefix = prefix or ()

    match second:
        case yaml.ScalarNode(end_mark=selection_end):
            current_node = Node(
                value=first.value,
                start=first.start_mark,
                end=first.end_mark,
                selection_end=selection_end,
            )
            segments.append((prefix + (str(first.value),), current_node))

        case yaml.MappingNode(end_mark=selection_end, value=children):
            current_node = Node(
                value=first.value,
                start=first.start_mark,
                end=first.end_mark,
                selection_end=selection_end,
            )
            segments.append((prefix + (str(first.value),), current_node))
            segments.extend(
                list(
                    chain.from_iterable(
                        [
                            _do_segmentize_nodes(
                                child[0], child[1], prefix=prefix + (str(first.value),)
                            )
                            for child in children
                        ]
                    )
                )
            )

        case yaml.CollectionNode(end_mark=selection_end):
            current_node = Node(
                value=first.value,
                start=first.start_mark,
                end=first.end_mark,
                selection_end=selection_end,
            )
            segments.append((prefix + (str(first.value),), current_node))
        case other:
            logger.error(other)

    return segments


def get_diagnostics(
    validator: Validator, tokens: list[Token], instance: YamlDocument
) -> list[lsp.Diagnostic]:
    """Validate a document against its schema."""
    diagnostics = []

    for error in validator.iter_errors(instance):
        if error.context:
            error = sorted(error.context, key=relevance)[0]

        match error:
            case ValidationError(
                validator="additionalProperties", absolute_path=path, message=message
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
                            message=message,
                            severity=lsp.DiagnosticSeverity.Error,
                            range=range_,
                            source=SOURCE,
                        )
                    )

            case ValidationError(
                validator="required",
                absolute_path=path,
                message=message,
                schema={**schema},
            ):
                range_ = get_faulty_token_range(tokens, path) if path else DEFAULT_RANGE
                message = str(schema.get("err_msg", message))

                diagnostics.append(
                    lsp.Diagnostic(
                        message=message,
                        severity=lsp.DiagnosticSeverity.Error,
                        range=range_,
                        source=SOURCE,
                    )
                )

            case ValidationError(
                absolute_path=path, message=str(message), schema={**schema}
            ):
                # The sub-error might have a path we should highlight
                path = deque(cast(Iterable[str], schema.get("err_path", path)))
                range_ = get_faulty_token_range(tokens, path) if path else DEFAULT_RANGE
                message = str(schema.get("err_msg", message))
                range_ = get_faulty_token_range(tokens, path)

                diagnostics.append(
                    lsp.Diagnostic(
                        message=message,
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

    if not path_segments:
        return DEFAULT_RANGE
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


def sanatize_key(key: str) -> str:
    """Sanatize key."""
    return re.sub(r"""['"\\*]""", "", key)


def get_description_from_path(path: Iterable[str | int], schema: Schema) -> str:
    """Given an element path, get its description."""
    # The first part of the query must always be a perfect match according to all
    # schemas. It's also better for performance.
    head, *tail = path
    query = f"$.properties.{sanatize_key(str(head))}"
    if tail:
        sub_query = "..".join(
            [
                f"'{sanatize_key(str(p))}'|additionalProperties|patternProperties"
                for p in tail
            ]
        )
        query = f"{query}..{sub_query}"
    query = f"{query}.description|title"
    parser = parse(query)
    candidates = parser.find(schema)

    if candidates:
        return str(candidates[0].value).capitalize()
    else:
        return MISSING_DESC


def get_schema_path_from_token_position(
    position: lsp.Position, instance_document: str
) -> deque[str] | None:
    """Parse the document to find the path to the current position."""
    scanned_tokens = parse_tokens(instance_document)
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


def list_symbols(
    instance: YamlDocument, segments: dict[tuple[str, ...], Node]
) -> list[lsp.DocumentSymbol]:
    """List document symbols.

    We are only interested in keys up to the second level at most, so we don't need anything
    fancy here.
    """
    symbols = []
    for top_level_key in instance.keys():
        node = segments[(top_level_key,)]
        symbol = lsp.DocumentSymbol(
            name=node.value,
            kind=lsp.SymbolKind.Key,
            range=lsp.Range(
                start=lsp.Position(node.start.line, node.start.column),
                end=lsp.Position(node.end.line, node.end.column),
            ),
            selection_range=lsp.Range(
                start=lsp.Position(node.start.line, node.start.column),
                end=lsp.Position(node.selection_end.line, node.selection_end.column),
            ),
        )
        if top_level_key in SPECIAL_SYMBOL_PARENTS:
            children_symbols = []
            for second_level_key in instance[top_level_key].keys():
                child_node = segments[(top_level_key, second_level_key)]
                child_symbol = lsp.DocumentSymbol(
                    name=child_node.value,
                    kind=lsp.SymbolKind.Key,
                    range=lsp.Range(
                        start=lsp.Position(
                            child_node.start.line, child_node.start.column
                        ),
                        end=lsp.Position(child_node.end.line, child_node.end.column),
                    ),
                    selection_range=lsp.Range(
                        start=lsp.Position(
                            child_node.start.line, child_node.start.column
                        ),
                        end=lsp.Position(
                            child_node.selection_end.line,
                            child_node.selection_end.column,
                        ),
                    ),
                )
                children_symbols.append(child_symbol)

            symbol.children = children_symbols
        symbols.append(symbol)

    return symbols
