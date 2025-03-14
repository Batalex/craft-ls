import json
from collections import deque
from textwrap import dedent

from hypothesis import assume, example, given
from hypothesis import strategies as st
from jsonschema.validators import validator_for
from lsprotocol import types as lsp

from craft_ls.core import (
    MISSING_DESC,
    get_description_from_path,
    get_diagnostics,
    get_faulty_token_range,
    get_schema_path_from_token_position,
    scan_for_tokens,
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


@given(key=st.text())
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


def test_get_path_from_position_first_level_ok() -> None:
    # Given
    # line is 2 because of initial newline after """ + comment in the document
    position = lsp.Position(2, 5)

    # When
    path = get_schema_path_from_token_position(position, document)

    # Then
    assert path == deque(["productId"])


def test_get_path_from_position_nested_ok() -> None:
    # Given
    position = lsp.Position(5, 5)

    # When
    path = get_schema_path_from_token_position(position, document)

    # Then
    assert path == deque(["price", "amount"])


def test_get_path_from_comment_ko() -> None:
    # Given
    position = lsp.Position(1, 5)  # comment line

    # When
    path = get_schema_path_from_token_position(position, document)

    # Then
    assert not path


def test_get_path_from_empty_space_ko() -> None:
    # Given
    position = lsp.Position(4, 10)  # to the right of "price"

    # When
    path = get_schema_path_from_token_position(position, document)

    # Then
    assert not path


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
    scan = scan_for_tokens(document)

    # When
    range_ = get_faulty_token_range(scan.tokens, ["productName"])

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

    # When
    diagnostics = get_diagnostics(validator, document)

    # Then
    assert len(diagnostics) == 2
    assert all("unexpected" in diag.message for diag in diagnostics)
