"""Define the language server features."""

import os
from pathlib import Path
from typing import Any, cast

from lsprotocol import types
from pygls.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import (
    get_description_from_path,
    get_diagnostics,
    get_schema_path_from_token_position,
    validators,
)

IS_DEV_MODE = os.environ.get("CRAFT_LS_DEV")

server = LanguageServer(
    name="craft-ls",
    version=__version__,
)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def on_opened(params: types.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is opened."""
    uri = params.text_document.uri
    version = params.text_document.version
    source = params.text_document.text

    file_stem = Path(uri).stem
    validator = validators.get(file_stem, None)
    diagnostics = (
        [
            types.Diagnostic(
                message=f"Running craft-ls {__version__}.",
                range=types.Range(
                    start=types.Position(line=0, character=0),
                    end=types.Position(line=0, character=0),
                ),
                severity=types.DiagnosticSeverity.Information,
            )
        ]
        if IS_DEV_MODE
        else []
    )
    if validator := validators.get(file_stem, None):
        diagnostics.extend(get_diagnostics(validator, source))

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(params: types.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is changed."""
    doc = server.workspace.get_text_document(params.text_document.uri)
    uri = params.text_document.uri
    version = params.text_document.version
    # source = params.text_document.text

    file_stem = Path(uri).stem
    validator = validators.get(file_stem, None)
    diagnostics = []
    if validator := validators.get(file_stem, None):
        diagnostics.extend(get_diagnostics(validator, doc.source))

    server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Get item description on hover."""
    pos = params.position
    uri = params.text_document.uri
    document_uri = params.text_document.uri
    document = ls.workspace.get_text_document(document_uri)

    file_stem = Path(uri).stem
    if not (validator := validators.get(file_stem, None)):
        return None

    if not (
        path := get_schema_path_from_token_position(
            position=pos, instance_document=document.source
        )
    ):
        return None

    description = get_description_from_path(
        path=path, schema=cast(dict[str, Any], validator.schema)
    )
    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=description,
        ),
        range=types.Range(
            start=types.Position(line=pos.line, character=0),
            end=types.Position(line=pos.line + 1, character=0),
        ),
    )


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
