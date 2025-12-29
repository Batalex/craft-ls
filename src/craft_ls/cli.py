"""Use LSP capabilities directly using a command line interface.

Can be used to test the error parsing without the additional complexity of the server.
"""

import logging
import sys
from pathlib import Path

from lsprotocol import types as lsp

from craft_ls.core import get_diagnostics, get_validator_and_parse, segmentize_nodes
from craft_ls.types_ import ParsedResult

logging.basicConfig()


def check(file_name: str) -> None:
    """Report all violations for a file."""
    file = Path(file_name)

    diagnostics: list[lsp.Diagnostic] = []
    match get_validator_and_parse(file.stem, file.read_text()):
        case None:
            print(f"Cannot validate '{file}'", file=sys.stderr)
            pass

        case validator, ParsedResult(instance=instance, nodes=nodes):
            segments = segmentize_nodes(nodes)
            diagnostics.extend(get_diagnostics(validator, instance, dict(segments)))

    if diagnostics:
        for diag in diagnostics:
            print(f"{diag.range.start.line}: {diag.message}", file=sys.stderr)
        sys.exit(1)
