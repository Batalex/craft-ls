"""Parser and document querying logic."""

from __future__ import annotations

import logging
from typing import Any

import lsprotocol.types as lsp
import tree_sitter_yaml as tsyaml
from pygls.workspace import PositionCodec, TextDocument
from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

YAML_LANGUAGE = Language(tsyaml.language())
parser = Parser(YAML_LANGUAGE)
query_errors = Query(YAML_LANGUAGE, "(ERROR) @error-node")
query_errors_cursor = QueryCursor(query_errors)

logger = logging.getLogger(__name__)


def apply_change_to_tree_and_text(
    tree: Tree,
    old_text: str,
    change: lsp.TextDocumentContentChangePartial,
    position_codec: PositionCodec,
) -> tuple[Tree, str]:
    """Apply a code change to the document tree and text.

    Adapted from https://tree-sitter.github.io/tree-sitter/using-parsers/3-advanced-parsing.html.
    """
    old_doc = TextDocument(uri="", source=old_text, position_codec=position_codec)

    start_char_offset = old_doc.offset_at_position(change.range.start)
    old_end_char_offset = old_doc.offset_at_position(change.range.end)

    start_byte = len(old_text[:start_char_offset].encode("utf-8"))
    old_end_byte = len(old_text[:old_end_char_offset].encode("utf-8"))
    new_end_byte = start_byte + len(change.text.encode("utf-8"))

    start_row = change.range.start.line
    start_line_start = old_text.rfind("\n", 0, start_char_offset) + 1
    start_byte_col = len(old_text[start_line_start:start_char_offset].encode("utf-8"))
    start_point = (start_row, start_byte_col)

    old_end_row = change.range.end.line
    old_end_line_start = old_text.rfind("\n", 0, old_end_char_offset) + 1
    old_end_byte_col = len(
        old_text[old_end_line_start:old_end_char_offset].encode("utf-8")
    )
    old_end_point = (old_end_row, old_end_byte_col)

    newlines = change.text.count("\n")
    new_end_row = start_row + newlines
    if newlines == 0:
        new_byte_col = start_byte_col + len(change.text.encode("utf-8"))
    else:
        last_line = change.text.split("\n")[-1]
        new_byte_col = len(last_line.encode("utf-8"))
    new_end_point = (new_end_row, new_byte_col)

    tree.edit(
        start_byte=start_byte,
        old_end_byte=old_end_byte,
        new_end_byte=new_end_byte,
        start_point=start_point,
        old_end_point=old_end_point,
        new_end_point=new_end_point,
    )

    new_text = (
        old_text[:start_char_offset] + change.text + old_text[old_end_char_offset:]
    )
    return tree, new_text


query_pairs = QueryCursor(
    Query(
        YAML_LANGUAGE,
        """
    (block_mapping_pair) @pair
    (flow_pair) @pair
    """,
    )
)

query_snap_base_keys = QueryCursor(
    Query(
        YAML_LANGUAGE,
        """
    (block_mapping_pair
      key: (_) @key_node
      value: (_) @value_node
      (#match? @key_node "^(base|build-base)$"))
    """,
    )
)

query_charm_type_keys = QueryCursor(
    Query(
        YAML_LANGUAGE,
        """
    (block_mapping_pair
      key: (_) @key_node
      value: (_) @value_node
      (#match? @key_node "^type$"))
    """,
    )
)


def _parse_scalar(val: str) -> Any:
    """Infers primitive types from scalar strings.

    Similar to yaml.safe_load, but we can instantiate/use it on every token.
    """
    val = val.strip()
    if not val or val in ("null", "~"):
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False

    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        # properly pass "1" as a str
        return val[1:-1]

    try:
        return float(val) if "." in val or "e" in val else int(val)
    except ValueError:
        return val


def node_to_dict(node: Node | None) -> Any:  # noqa: C901
    """Recursively transforms a raw Tree-sitter node into Python structures.

    We have 4 different types to transform:
    1. Scalars
    2. Sequences
    3. Transparent structures (not needed)
    4. Mappings, including errors

    A bit dirty, but it seems to work
    """
    if not node or node.type in ("-", "---", "...", "MISSING", ":"):
        return None

    if "scalar" in node.type:
        return _parse_scalar(node.text.decode("utf-8") if node.text else "")

    if node.type in ("block_sequence", "flow_sequence"):
        return [
            node_to_dict(c)
            for c in node.children
            if c.type not in (",", "[", "]", "ERROR")
        ]

    if node.type == "block_sequence_item":
        val_node = next(
            (c for c in node.children if c.type not in ("-", "ERROR", "MISSING")), None
        )
        return node_to_dict(val_node) if val_node else None

    # Bypass those ones
    if (
        node.type in ("stream", "document", "block_node", "flow_node")
        and node.child_count == 1
    ):
        return node_to_dict(node.children[0])

    if node.type in ("block_mapping", "flow_mapping", "ERROR"):
        res = {}
        for c in node.children:
            if "pair" in c.type:
                k = c.child_by_field_name("key")
                v = c.child_by_field_name("value")
                if k and k.text:
                    k_str = k.text.decode("utf-8").strip()
                    if k_str and not k_str.startswith(("-", "---", "...")):
                        res[k_str] = node_to_dict(v)

            elif "scalar" in c.type:
                # Captures an incomplete trailing key being typed at this specific scope layer
                k_str = (c.text.decode("utf-8") if c.text else "").strip()
                if k_str and not k_str.startswith(("-", "---", "...")):
                    res[k_str] = None

            elif c.type == "ERROR":
                # If there's a nested error block, recursively extract its fragments and merge them
                sub_error_dict = node_to_dict(c)
                if isinstance(sub_error_dict, dict):
                    res.update(sub_error_dict)
        return res

    return _parse_scalar(node.text.decode("utf-8")) if node.text else None


def yaml_tree_to_dict(tree: Tree) -> dict[str, Any]:
    """Top-level entrypoint to safely convert any parsed Tree into a clean dict."""
    res = node_to_dict(tree.root_node)
    return res if isinstance(res, dict) else {}
