"""Tests for the tree-sitter tree to dict conversion."""

import pytest

from craft_ls.parser import parser, yaml_tree_to_dict


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Plain valid documents
        (b"name: foo\n", {"name": "foo"}),
        (b'version: "1.6.0"\n', {"version": "1.6.0"}),
        (b"parts:\n  p1:\n    plugin: nil\n", {"parts": {"p1": {"plugin": "nil"}}}),
        # Incomplete document: salvage what is valid
        (b"name: foo\nsummary", {"name": "foo"}),
        # Comments and document markers must not hide the mapping
        (b"# a\n# b\nname: foo\n", {"name": "foo"}),
        (b"---\nname: foo\n", {"name": "foo"}),
        (b"# only a comment\n", {}),
        # Multiple documents: only the last one
        (b"name: foo\n---\nname: bar\n", {"name": "bar"}),
        # Empty input
        (b"", {}),
    ],
)
def test_yaml_tree_to_dict(source: bytes, expected: dict) -> None:
    assert yaml_tree_to_dict(parser.parse(source)) == expected
