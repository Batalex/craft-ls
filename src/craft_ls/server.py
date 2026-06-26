"""Define the language server features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, cast

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import (
    get_completion_items_from_path,
    get_completion_path,
    get_description_from_path,
    get_diagnostics,
    get_node_path_from_token_position,
    get_validator_from_tree,
    list_symbols,
)
from craft_ls.helpers import shorten_diagnostics_messages
from craft_ls.parser import (
    apply_change_to_tree_and_text,
    parser,
    yaml_tree_to_dict,
)
from craft_ls.settings import IS_DEV_MODE
from craft_ls.types_ import DocumentsIndex, IndexEntry, Schema, YamlDocument

logger = logging.getLogger(__name__)


class CraftLanguageServer(LanguageServer):
    """*craft tools language server."""

    def __init__(self, name: str, version: str) -> None:
        super().__init__(
            name=name,
            version=version,
            text_document_sync_kind=lsp.TextDocumentSyncKind.Incremental,
        )
        self.documents_index: DocumentsIndex = {}

    def parse_document(
        self,
        file_uri: str,
        content_changes: Iterable[lsp.TextDocumentContentChangeEvent] | None = None,
    ) -> IndexEntry:
        """Parse a document.

        The result is cached so we can access it various LS endpoints.
        """
        document = self.workspace.get_text_document(file_uri)
        cached = self.documents_index.get(file_uri)

        tree = None
        current_text = document.source

        # Update tree in-place to not re-parse the entire document
        if cached and content_changes:
            old_tree, _, _, old_text, _ = cached
            tree = old_tree

            full_reparse_required = False
            for change in content_changes:
                if getattr(change, "range", None) is None:
                    full_reparse_required = True
                    break

                tree, current_text = apply_change_to_tree_and_text(
                    tree,
                    old_text,
                    cast(lsp.TextDocumentContentChangePartial, change),
                    document.position_codec,
                )

            if full_reparse_required or tree is not None:
                tree = parser.parse(current_text.encode("utf-8"), old_tree=tree)

        if tree is None:
            # If the in-place upgrade failed for any reason, we fall back to re-parsing
            # the entire document.
            current_text = document.source
            tree = parser.parse(document.source.encode("utf-8"))

        validator = get_validator_from_tree(Path(file_uri).stem, tree)
        instance = cast(YamlDocument, yaml_tree_to_dict(tree))
        self.documents_index[file_uri] = IndexEntry(
            tree,
            validator,
            instance,
            current_text,
            document.version,
        )
        return IndexEntry(
            tree,
            validator,
            instance,
            current_text,
            document.version,
        )


server = CraftLanguageServer(
    name="craft-ls",
    version=__version__,
)


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def on_opened(ls: CraftLanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
    """Parse a document when it is first opened."""
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
    tree, validator, instance, _, version = ls.parse_document(params.text_document.uri)
    if validator:
        diagnostics.extend(get_diagnostics(tree, validator, instance))
    shorten_diagnostics_messages(diagnostics)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=uri,
            version=version,
            diagnostics=diagnostics,
        )
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(
    ls: CraftLanguageServer, params: lsp.DidChangeTextDocumentParams
) -> None:
    """Parse a document when it is edited."""
    uri = params.text_document.uri
    tree, validator, instance, _, version = ls.parse_document(uri)
    if validator:
        diagnostics = get_diagnostics(tree, validator, instance)
    else:
        diagnostics = []
    shorten_diagnostics_messages(diagnostics)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=uri,
            version=version,
            diagnostics=diagnostics,
        )
    )


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbols(
    ls: CraftLanguageServer, params: lsp.DocumentSymbolParams
) -> list[lsp.DocumentSymbol]:
    """Return all the symbols defined in the given document."""
    uri = params.text_document.uri
    tree, *_ = ls.documents_index[uri]

    return list_symbols(tree)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: CraftLanguageServer, params: lsp.HoverParams) -> lsp.Hover | None:
    """Get item description on hover."""
    uri = params.text_document.uri
    pos = params.position
    tree, validator, *_ = ls.documents_index[uri]

    if not validator or not (
        path := get_node_path_from_token_position(tree, position=pos)
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


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION, lsp.CompletionOptions(trigger_characters=[" "])
)
def completions(
    ls: CraftLanguageServer, params: lsp.CompletionParams
) -> lsp.CompletionList | None:
    """Suggest next element based on the document structure."""
    uri = params.text_document.uri
    pos = params.position
    tree, validator, instance, text, _ = ls.documents_index[uri]
    items = []

    if validator:
        path = get_completion_path(tree, text, pos)
        items = get_completion_items_from_path(
            segments=path, schema=cast(Schema, validator.schema), instance=instance
        )

    return lsp.CompletionList(is_incomplete=False, items=items)


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
