"""Utilities."""

import re
from textwrap import shorten

from lsprotocol import types as lsp

MSG_SIZE = 79


def sanatize_key(key: str) -> str:
    """Sanatize key."""
    return re.sub(r"""['"\\*]""", "", key)


def shorten_diagnostics_messages(diagnostics: list[lsp.Diagnostic]) -> None:
    """Shorten diagnostics messages to better fit an editor view."""
    for diagnostic in diagnostics:
        diagnostic.message = shorten(diagnostic.message, MSG_SIZE)
