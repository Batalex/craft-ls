"""Define the language server features."""

import logging
from pathlib import Path
from textwrap import shorten
from typing import cast

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import (
    get_completion_items_from_path,
    get_description_from_path,
    get_diagnostics,
    get_exact_cursor_path,
    get_node_path_from_token_position,
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
        self.index: dict[str, IndexEntry | None] = {}

    def parse_file(self, file_uri: str) -> IndexEntry | None:
        """Parse a document into tokens, nodes and whatnot.

        The result is cached so we can access it in endpoints.
        """
        document = self.workspace.get_text_document(file_uri)
        match get_validator_and_parse(Path(file_uri).stem, document.source):
            case None:
                self.index[file_uri] = None

            case validator, ParsedResult(tokens, instance, nodes):
                segments_nodes = segmentize_nodes(nodes)
                self.index[file_uri] = IndexEntry(
                    validator, tokens, instance, dict(segments_nodes), document.version
                )

        return self.index[file_uri]

    def get_or_update_index(self, file_uri: str) -> IndexEntry | None:
        """Re-parse document if needed."""
        current_version = self.workspace.get_text_document(file_uri).version
        entry = self.index.get(file_uri)
        match entry:
            case IndexEntry(version=cached_version) as cached:
                if not cached_version or cached_version != current_version:
                    return self.parse_file(
                        file_uri,
                    )
                return cached
            case None:
                return None


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

    match ls.parse_file(uri):
        case IndexEntry(
            validator, instance=instance, segments=segments, version=version
        ):
            diagnostics.extend(get_diagnostics(validator, instance, segments))

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
    diagnostics = []

    match ls.parse_file(uri):
        case IndexEntry(
            validator, instance=instance, segments=segments, version=version
        ):
            diagnostics.extend(get_diagnostics(validator, instance, segments))

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

    match ls.get_or_update_index(uri):
        case IndexEntry(validator_found, segments=segments):
            validator = validator_found

        case _:
            return None

    if not (path := get_node_path_from_token_position(position=pos, segments=segments)):
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

    match ls.get_or_update_index(uri):
        case IndexEntry(instance=instance, segments=segments):
            symbols_results = list_symbols(instance, segments)

    return symbols_results


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION, lsp.CompletionOptions(trigger_characters=[" "])
)
def completions(
    ls: CraftLanguageServer, params: lsp.CompletionParams
) -> lsp.CompletionList | None:
    """Suggest next element based on the document structure."""
    pos = params.position
    uri = params.text_document.uri
    items = []

    match ls.get_or_update_index(uri):
        case IndexEntry(validator_found, instance=instance, tokens=tokens):
            validator = validator_found

        case _:
            return None

    path = get_exact_cursor_path(pos, tokens)
    items = get_completion_items_from_path(
        segments=path, schema=cast(Schema, validator.schema), instance=instance
    )

    return lsp.CompletionList(is_incomplete=False, items=items)


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
