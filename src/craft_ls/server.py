"""Define the language server features."""

import logging
from pathlib import Path
from textwrap import shorten
from typing import cast

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import (
    get_description_from_path,
    get_diagnostics,
    get_schema_path_from_token_position,
    get_validator_and_parse,
    list_symbols,
    segmentize_nodes,
)
from craft_ls.settings import IS_DEV_MODE
from craft_ls.types_ import IndexEntry, ParsedResult, Schema

MSG_SIZE = 79

logger = logging.getLogger(__name__)


class CraftLanguageServer(LanguageServer):
    """*craft tools language server."""

    def __init__(
        self,
        name: str,
        version: str,
        text_document_sync_kind: lsp.TextDocumentSyncKind = lsp.TextDocumentSyncKind.Incremental,
        notebook_document_sync: lsp.NotebookDocumentSyncOptions | None = None,
    ) -> None:
        super().__init__(
            name,
            version,
            text_document_sync_kind,
            notebook_document_sync,
        )
        self.index: dict[Path, IndexEntry | None] = {}

    def parse_file(self, file_uri: Path, source: str) -> IndexEntry | None:
        """Parse a document into tokens, nodes and whatnot.

        The result is cached so we can access it in endpoints.
        """
        match get_validator_and_parse(file_uri.stem, source):
            case None:
                self.index[file_uri] = None

            case validator, ParsedResult(tokens, instance, nodes):
                segments_nodes = segmentize_nodes(nodes)
                self.index[file_uri] = IndexEntry(
                    validator, tokens, instance, dict(segments_nodes)
                )

        return self.index[file_uri]


server = CraftLanguageServer(
    name="craft-ls",
    version=__version__,
)


def shorten_messages(diagnostics: list[lsp.Diagnostic]) -> None:
    """Shorten diagnostics messages to better fit an editor view."""
    for diagnostic in diagnostics:
        diagnostic.message = shorten(diagnostic.message, MSG_SIZE)


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def on_opened(ls: CraftLanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is opened."""
    uri = params.text_document.uri
    version = params.text_document.version
    doc = ls.workspace.get_text_document(params.text_document.uri)
    source = doc.source
    diagnostics = (
        [
            lsp.Diagnostic(
                message=f"Running craft-ls {__version__}.",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=0),
                ),
                severity=lsp.DiagnosticSeverity.Information,
            )
        ]
        if IS_DEV_MODE
        else []
    )

    match ls.parse_file(Path(uri), source):
        case IndexEntry(validator, tokens, instance):
            diagnostics.extend(get_diagnostics(validator, tokens, instance))

        case _:
            return

    shorten_messages(diagnostics)
    if diagnostics:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(
                uri=uri, version=version, diagnostics=diagnostics
            )
        )


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(ls: CraftLanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is changed."""
    uri = params.text_document.uri
    version = params.text_document.version
    doc = ls.workspace.get_text_document(params.text_document.uri)
    source = doc.source
    diagnostics = []

    match ls.parse_file(Path(uri), source):
        case IndexEntry(validator, tokens, instance):
            diagnostics.extend(get_diagnostics(validator, tokens, instance))

        case _:
            return

    shorten_messages(diagnostics)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, version=version, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: CraftLanguageServer, params: lsp.HoverParams) -> lsp.Hover | None:
    """Get item description on hover."""
    pos = params.position
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    source = doc.source

    match ls.index.get(Path(uri)):
        case IndexEntry(validator_found):
            validator = validator_found

        case _:
            return None

    if not (
        path := get_schema_path_from_token_position(
            position=pos, instance_document=source
        )
    ):
        return None

    description = get_description_from_path(
        path=path, schema=cast(Schema, validator.schema)
    )

    return lsp.Hover(
        contents=lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown,
            value=description,
        ),
        range=lsp.Range(
            start=lsp.Position(line=pos.line, character=0),
            end=lsp.Position(line=pos.line + 1, character=0),
        ),
    )


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: CraftLanguageServer, params: lsp.DocumentSymbolParams
) -> list[lsp.DocumentSymbol]:
    """Return all the symbols defined in the given document."""
    uri = params.text_document.uri
    symbols_results: list[lsp.DocumentSymbol] = []

    match ls.index.get(Path(uri)):
        case IndexEntry(instance=instance, segments=segments):
            symbols_results = list_symbols(instance, segments)

    return symbols_results


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
