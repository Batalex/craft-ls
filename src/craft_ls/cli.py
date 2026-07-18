"""Use LSP capabilities directly using a command line interface.

Can be used to test the error parsing without the additional complexity of the server.
"""

import logging
import sys
from pathlib import Path

from lsprotocol import types as lsp

from craft_ls.core import (
    get_diagnostics,
    get_validator_from_tree,
)
from craft_ls.parser import parser, yaml_tree_to_dict

logging.basicConfig()


def check(file_name: str) -> None:
    """Report all violations for a file."""
    file = Path(file_name)

    with file.open("rb") as f:
        tree = parser.parse(f.read())

    validator = get_validator_from_tree(file.stem, tree)
    instance = yaml_tree_to_dict(tree)

    if not validator:
        print(f"Cannot validate '{file}'", file=sys.stderr)
        sys.exit(1)

    diagnostics: list[lsp.Diagnostic] = get_diagnostics(tree, validator, instance)

    if diagnostics:
        for diag in diagnostics:
            print(f"{diag.range.start.line}: {diag.message}", file=sys.stderr)
        sys.exit(1)
