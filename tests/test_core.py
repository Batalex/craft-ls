import json
from collections import deque

from hypothesis import assume, example, given
from hypothesis import strategies as st
from lsprotocol import types as lsp

from craft_ls.core import (
    MISSING_DESC,
    get_description_from_path,
    get_schema_path_from_token_position,
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
  "required": ["productId", "productName", "price"]
}
"""
)

document = """
# Some comment
productId: foo
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
