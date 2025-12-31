import json
from collections import deque
from itertools import chain
from textwrap import dedent

import yaml
from hypothesis import assume, example, given
from hypothesis import strategies as st
from jsonschema.validators import validator_for
from lsprotocol import types as lsp

from craft_ls.core import (
    MISSING_DESC,
    get_description_from_path,
    get_diagnostic_range,
    get_diagnostics,
    get_exact_cursor_path,
    get_node_path_from_token_position,
    list_symbols,
    parse_tokens,
    segmentize_nodes,
)

# Adapted from json-schema.org
schema = json.loads(
    """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "object",
      "properties": {
        "amount": {
          "description": "The price amount",
          "type": "number",
          "exclusiveMinimum": 0
        },
        "currency": {
          "description": "The price currency",
          "type": "string"
        }
      }
    }
  },
  "required": ["productId", "productName", "price"],
  "additionalProperties": false
}
"""
)
validator = validator_for(schema)(schema)

document = """
# Some comment
productId: 001
productName: bar
price:
  amount: 50
  currency: euro
"""

parsed_document = parse_tokens(document)
document_segments = segmentize_nodes(parsed_document.nodes)


def test_get_description_first_level_ok() -> None:
    """Assert that we can get the description from the schema of a first-level key."""
    # Given
    item_path = ["price"]

    # When
    description = get_description_from_path(item_path, schema)

    # Then
    assert description == "The price of the product"


def test_get_description_nested_ok() -> None:
    """Assert that we can get the description from the schema of a non first-level key."""
    # Given
    item_path = ["price", "currency"]

    # When
    description = get_description_from_path(item_path, schema)

    # Then
    assert description == "The price currency"


@given(key=st.text(min_size=1))
@example("something_not_present")
def test_get_description_unknown_path(key: str) -> None:
    """Assert that we get a default message if the key is not in the schema."""
    # Given
    assume(key not in {"amount", "currency"})  # exclude the two counter examples
    item_path = ["price", key]

    # When
    description = get_description_from_path(item_path, schema)

    # Then
    assert description == MISSING_DESC


def test_get_node_path_from_position_first_level_ok() -> None:
    # Given
    # line is 2 because of initial newline after """ + comment in the document
    position = lsp.Position(2, 5)

    # When
    path = get_node_path_from_token_position(position, dict(document_segments))

    # Then
    assert path == ("productId",)


def test_get_node_path_from_position_nested_ok() -> None:
    # Given
    position = lsp.Position(5, 5)

    # When
    path = get_node_path_from_token_position(position, dict(document_segments))

    # Then
    assert path == ("price", "amount")


def test_get_node_path_from_outside_ko() -> None:
    # Given
    position = lsp.Position(1, 5)  # comment line

    # When
    path = get_node_path_from_token_position(position, dict(document_segments))

    # Then
    assert not path


def test_get_node_path_from_value_ok() -> None:
    # Given
    position = lsp.Position(4, 10)  # to the right of "price"

    # When
    path = get_node_path_from_token_position(position, dict(document_segments))

    # Then
    assert path == ("price",)


def test_get_cursor_path_from_token_ok() -> None:
    # Given
    position = lsp.Position(4, 3)  # inside "price", current path should be root

    # When
    path = get_exact_cursor_path(position, parsed_document.tokens)

    # Then
    assert path == deque([])


def test_get_cursor_path_from_nested_key_ok() -> None:
    # Given
    position = lsp.Position(5, 4)  # inside "amount", current path should be "price"

    # When
    path = get_exact_cursor_path(position, parsed_document.tokens)

    # Then
    assert path == deque(["price"])


def test_get_cursor_path_from_value_ok() -> None:
    # Given
    position = lsp.Position(3, 15)  # inside "bar", current path should be "productName"

    # When
    path = get_exact_cursor_path(position, parsed_document.tokens)

    # Then
    assert path == deque(["productName"])


def test_values_are_not_flagged() -> None:
    """Check that we flag the correct token: a key, not a value."""
    # Given
    # With the following document, an error on "productName" should
    # be matched with the third line, not the second one
    document = dedent(
        """
        # Some comment
        productId: productName
        productName: InvalidValue
        price:
          amount: 50
          currency: euro
        """
    )
    segments = dict(segmentize_nodes(parse_tokens(document).nodes))
    parsed_document.nodes

    # When
    range_ = get_diagnostic_range(segments, ["productName"])

    # Then
    assert range_.start.line == 3


def test_multiple_unexpected_keys() -> None:
    # Given
    # With the following document, we expect two diagnostics
    document = dedent(
        """
        # Some comment
        productId: 001
        productName: name
        price:
          amount: 50
          currency: euro
        foo: bar
        baz: buz
        """
    )
    segments = dict(segmentize_nodes(parse_tokens(document).nodes))

    # When
    diagnostics = get_diagnostics(validator, yaml.safe_load(document), segments)

    # Then
    assert len(diagnostics) == 2
    assert all("unexpected" in diag.message for diag in diagnostics)


def test_list_symbols_correct_levels() -> None:
    """We only expect 1st level keys and 2nd level keys for specific 1st keys.

    Using the document below, we expect 3 1st level keys and one child key.
    """
    # Given
    document = dedent(
        """
        # Some comment
        foo: bar
        parts:
            included-key:
                all: right
        unrelated:
            non-included-key: 50
        """
    )
    segments = dict(segmentize_nodes(yaml.compose(document)))

    # When
    symbols = list_symbols(yaml.safe_load(document), segments)

    # Then
    assert len(symbols) == 3
    children = list(
        chain.from_iterable([s.children for s in symbols if s.children is not None])
    )
    assert len(children) == 1
    assert children[0].name == "included-key"
