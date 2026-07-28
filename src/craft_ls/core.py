"""Core logic for crafting LS responses."""

import logging
import re
from importlib.resources import files
from typing import Any, Iterable, cast

import jsonref
import lsprotocol.types as lsp
from jsonpath_ng import parse as jq  # zuban: ignore[attr-defined]
from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import relevance
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from tree_sitter import Node, Tree

from craft_ls.helpers import sanatize_key
from craft_ls.parser import (
    query_charm_type_keys,
    query_pairs,
    query_snap_base_keys,
)
from craft_ls.types_ import (
    MissingTypeCharmcraftValidator,
    MissingTypeSnapcraftValidator,
    Schema,
)

SOURCE = "craft-ls"
FILE_TYPES = ["snapcraft", "rockcraft", "charmcraft"]
MISSING_DESC = "No description to display"
SPECIAL_SYMBOL_PARENTS = {"parts", "apps", "services"}
DEFAULT_RANGE = lsp.Range(
    start=lsp.Position(line=0, character=0),
    end=lsp.Position(line=0, character=0),
)

logger = logging.getLogger(__name__)

default_validators: dict[str, Validator] = {}
charmcraft_registry: Registry
for file_type in FILE_TYPES:
    schema_str = files("craft_ls.schemas").joinpath(f"{file_type}.json").read_text()
    schema = jsonref.loads(
        files("craft_ls.schemas").joinpath(f"{file_type}.json").read_text()
    )
    default_validators[file_type] = validator_for(schema)(schema)  # type: ignore

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


def get_snap_bases(tree: Tree) -> tuple[str | None, str | None]:
    """Get the snapcraft base and build-base keys and values."""
    matches = query_snap_base_keys.matches(tree.root_node)
    bases = {}

    for _, captures in matches:
        k_nodes = captures.get("key_node", [])
        v_nodes = captures.get("value_node", [])

        k_node = k_nodes[0] if k_nodes else None
        v_node = v_nodes[0] if v_nodes else None

        if (
            not k_node
            or not v_node
            or not k_node.text
            or not v_node.text
            # The base keys are only at depth=0 of the yaml doc
            or k_node.start_point.column != 0
        ):
            continue

        key_name = k_node.text.decode("utf-8").strip()
        bases[key_name] = v_node.text.decode("utf-8").strip()

    return bases.get("base", None), bases.get("build-base", None)


def get_charm_type(tree: Tree) -> str | None:
    """Get the charmcraft type key and value."""
    matches = query_charm_type_keys.matches(tree.root_node)
    charm_type: str | None = None

    for _, captures in matches:
        k_nodes = captures.get("key_node", [])
        v_nodes = captures.get("value_node", [])

        k_node = k_nodes[0] if k_nodes else None
        v_node = v_nodes[0] if v_nodes else None

        if (
            not k_node
            or not v_node
            or not v_node.text
            # The base keys are only at depth=0 of the yaml doc
            or k_node.start_point.column != 0
        ):
            continue

        charm_type = v_node.text.decode("utf-8").strip()

    return charm_type


def get_snapcraft_validator(tree: Tree) -> Validator:
    """Get the most appropriate snapcraft validator for the current document."""
    validator: Draft202012Validator | MissingTypeSnapcraftValidator
    match get_snap_bases(tree):
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
        case "core26", _:
            validator = Draft202012Validator(
                schema=snapcraft_registry.resolver()
                .lookup("urn:snapcraft:core26")
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
        case "bare", "core26":
            validator = Draft202012Validator(
                schema=snapcraft_registry.resolver()
                .lookup("urn:snapcraft:bare26")
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
                .lookup("urn:snapcraft:basedevel")
                .contents
            )

        case _:
            validator = MissingTypeSnapcraftValidator(
                schema=snapcraft_registry.resolver()
                .lookup("urn:snapcraft:core26")
                .contents
            )

    return cast(Validator, validator)


def get_validator_from_tree(file_stem: str, tree: Tree) -> Validator | None:
    """Get the most appropriate validator for the current document."""
    if file_stem not in FILE_TYPES:
        return None

    if file_stem == "rockcraft":
        return default_validators[file_stem]

    elif file_stem == "snapcraft":
        validator = get_snapcraft_validator(tree)

    else:
        # by elimination, file_stem is charmcraft
        if get_charm_type(tree) != "charm":
            return cast(
                Validator,
                MissingTypeCharmcraftValidator(
                    charmcraft_registry.resolver()
                    .lookup("urn:charmcraft:platformcharm")
                    .contents
                ),
            )

        validator = cast(
            Validator,
            Draft202012Validator(
                schema=charmcraft_registry.resolver()
                .lookup("urn:charmcraft:platformcharm")
                .contents
            ),
        )
    return validator


def get_diagnostics(
    tree: Tree,
    validator: Validator,
    instance: Any,
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
                        get_diagnostic_range(tree, cast(list[str], list(path)) + [key])
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
            ):
                pattern = "'(?P<key>.*)' key is mandatory"
                if path:
                    range_ = get_diagnostic_range(tree, cast(list[str], list(path)))
                elif (match := re.search(pattern, message or "")) and (
                    key := match.group("key")
                ):
                    # Const errors can be wrongfully handled here with an empty path,
                    # so we try to handle those cases gracefully
                    range_ = get_diagnostic_range(
                        tree, cast(list[str], list(path)) + [key]
                    )
                else:
                    range_ = DEFAULT_RANGE

                diagnostics.append(
                    lsp.Diagnostic(
                        message=message,
                        severity=lsp.DiagnosticSeverity.Error,
                        range=range_,
                        source=SOURCE,
                    )
                )

            case ValidationError(absolute_path=path, message=str(message)):
                path = cast(list[str], path)
                range_ = get_diagnostic_range(tree, path) if path else DEFAULT_RANGE

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


def get_diagnostic_range(tree: Tree, diag_segments: Iterable[str]) -> lsp.Range:
    """Link the validation error to the position in the original document."""
    segments = list(diag_segments)
    if not segments:
        return DEFAULT_RANGE

    segment_idx = 0
    stack = [tree.root_node]

    while stack and segment_idx < len(segments):
        current = stack.pop()
        target = segments[segment_idx]

        if current.type in ("block_mapping_pair", "flow_pair"):
            k = current.child_by_field_name("key")
            if k and k.text and k.text.decode("utf-8").strip() == target:
                segment_idx += 1

                # Early exit if this was the last segment
                if segment_idx == len(segments):
                    return lsp.Range(
                        start=lsp.Position(k.start_point.row, k.start_point.column),
                        end=lsp.Position(k.end_point.row, k.end_point.column),
                    )

                # Then continue the descent in this value subtree
                v = current.child_by_field_name("value")
                stack = [v] if v else []

            # Wrong subtree, continue horizontally
            continue

        if current.children:
            stack.extend(reversed(current.children))

    return DEFAULT_RANGE


def get_immediate_child_pairs(node: Node) -> list[Node]:
    """Get the key value pairs that are the level just below a given node."""
    captures = query_pairs.captures(node)

    if not (pairs := captures.get("pair", [])):
        return pairs

    # Quite the hack, but I like it
    # Immediate children are left aligned on the same col, because yaml is indent based
    min_column = min(p.start_point.column for p in pairs)
    return [p for p in pairs if p.start_point.column == min_column]


def create_symbol(pair: Node) -> lsp.DocumentSymbol | None:
    """Convert a tree-sitter pair node or loose scalar key into an LSP DocumentSymbol."""
    key = pair.child_by_field_name("key")
    value = pair.child_by_field_name("value")
    end_node = pair

    if not key or not key.text:
        return None

    name = key.text.decode("utf-8").strip()
    if not name or name.startswith(("-", "---", "...")):
        return None

    symbol = lsp.DocumentSymbol(
        name=name,
        kind=lsp.SymbolKind.Key,
        range=lsp.Range(
            start=lsp.Position(key.start_point.row, key.start_point.column),
            end=lsp.Position(end_node.end_point.row, end_node.end_point.column),
        ),
        selection_range=lsp.Range(
            start=lsp.Position(key.start_point.row, key.start_point.column),
            end=lsp.Position(key.end_point.row, key.end_point.column),
        ),
    )

    if name in SPECIAL_SYMBOL_PARENTS and value:
        children_symbols = []
        for child_pair in get_immediate_child_pairs(value):
            if child_symbol := create_symbol(child_pair):
                children_symbols.append(child_symbol)

        if children_symbols:
            symbol.children = sorted(children_symbols, key=lambda s: s.range.start.line)

    return symbol


def list_symbols(tree: Tree) -> list[lsp.DocumentSymbol]:
    """List first-level keys and expands special parent to get the second level.

    Simply put, we are only interested in keys up to the second level at most, and only
    for things like parts, apps, etc.
    """
    symbols = []
    for child_pair in get_immediate_child_pairs(tree.root_node):
        child_symbol = create_symbol(child_pair)
        if child_symbol:
            symbols.append(child_symbol)

    return sorted(symbols, key=lambda s: s.range.start.line)


def get_node_path_from_token_position(
    tree: Tree, position: lsp.Position
) -> tuple[str, ...] | None:
    """Finds the innermost key path tracking down to the current cursor position.

    To do so, we actually proceed the other way around. We start from the node itself
    and navigate through its ancestors.
    """
    point = (position.line, position.character)
    leaf = tree.root_node.descendant_for_point_range(point, point)
    if not leaf:
        return None

    path: list[str] = []
    curr = leaf

    while curr:
        if curr.type in ("block_mapping_pair", "flow_pair"):
            k = curr.child_by_field_name("key")
            if k and k.text:
                k_str = k.text.decode("utf-8").strip()
                if k_str and not k_str.startswith(("-", "---", "...")):
                    path.append(k_str)

        if (parent := curr.parent) is None:
            break

        curr = parent

    return tuple(reversed(path)) if path else None


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
    parser = jq(query)
    candidates = parser.find(schema)

    if candidates:
        return str(candidates[0].value).capitalize()
    else:
        return MISSING_DESC


def get_completion_path(
    tree: Tree, document_text: str, position: lsp.Position
) -> list[str]:
    """Finds the YAML key path at the cursor position for autocompletion.

    We need to be more precise than for the hovering and know exactly if the user is typing
    a key or a value.
    """
    lines = document_text.splitlines()
    prefix = (
        lines[position.line][: position.character] if position.line < len(lines) else ""
    )
    current_indent = len(prefix) - len(prefix.lstrip())

    keys_above: list[tuple[int, str]] = []
    stack = [tree.root_node] if tree and tree.root_node else []

    while stack:
        node = stack.pop()

        if node.type in ("block_mapping_pair", "flow_pair"):
            key_node = node.child_by_field_name("key")

            # Quite intuitively, yaml is written left to right, top to bottom
            # Therefore, valid "ancestors" keys must be above the position
            if key_node and key_node.start_point.row < position.line and key_node.text:
                key_str = key_node.text.decode().strip()

                if key_str and not key_str.startswith(("-", "---", "...")):
                    col = key_node.start_point.column

                    # drop sibling keys or keys from closed blocks
                    while keys_above and keys_above[-1][0] >= col:
                        keys_above.pop()
                    keys_above.append((col, key_str))

        if node.children:
            stack.extend(reversed(node.children))

    # Remove keys that are as deep or deeper than our current indentation
    while keys_above and keys_above[-1][0] >= current_indent:
        keys_above.pop()

    path = [key_name for _, key_name in keys_above]

    # If the user is typing a value, then we add the current key to the path so
    # that we can check enums etc. in the schema.
    if ":" in prefix and (current_key := prefix.split(":")[0].strip()):
        path.append(current_key)

    return path


def get_completion_items_from_path(
    segments: Iterable[str], schema: Schema, instance: Any
) -> list[lsp.CompletionItem]:
    """Get possible values for children nodes or enum values."""
    sub_instance = instance
    sub_schema = schema
    for segment in segments:
        sub_schema = sub_schema.get("properties", {}).get(segment, {})
        sub_instance = sub_instance.get(segment, {})

    if (const_val := sub_schema.get("const")) is not None:
        return [lsp.CompletionItem(label=str(const_val))]

    all_schemas = [sub_schema] + [
        item for c in ("anyOf", "oneOf", "allOf") for item in sub_schema.get(c, [])
    ]

    properties_keys = {key for s in all_schemas for key in s.get("properties", {})}
    if properties_keys:
        return [
            lsp.CompletionItem(label=str(k))
            for k in properties_keys - set(sub_instance)
        ]

    enums = [val for s in all_schemas for val in s.get("enum", []) if val]
    if enums:
        return [lsp.CompletionItem(label=str(val)) for val in dict.fromkeys(enums)]

    return []
