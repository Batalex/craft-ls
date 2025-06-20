"""Define the language server features."""

import logging
import os
from pathlib import Path
from typing import cast

from lsprotocol import types as lsp
from pygls.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import (
    get_description_from_path,
    get_diagnostics,
    get_schema_path_from_token_position,
    get_validator_and_scan,
)
from craft_ls.types_ import IncompleteScan, Schema

IS_DEV_MODE = os.environ.get("CRAFT_LS_DEV")

logger = logging.getLogger(__name__)
server = LanguageServer(
    name="craft-ls",
    version=__version__,
)


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def on_opened(params: lsp.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is opened."""
    uri = params.text_document.uri
    version = params.text_document.version
    source = params.text_document.text

    file_stem = Path(uri).stem
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

    match get_validator_and_scan(file_stem, source):
        case None, _:
            pass

        case None, IncompleteScan():
            diagnostics.append(
                lsp.Diagnostic(
                    message="File is malformed",
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=0),
                    ),
                    severity=lsp.DiagnosticSeverity.Warning,
                )
            )
            pass

        case validator, scan_result:
            diagnostics.extend(get_diagnostics(validator, scan_result))

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(params: lsp.DidOpenTextDocumentParams) -> None:
    """Parse each document when it is changed."""
    doc = server.workspace.get_text_document(params.text_document.uri)
    uri = params.text_document.uri
    version = params.text_document.version
    # source = params.text_document.text

    file_stem = Path(uri).stem
    diagnostics = []

    match get_validator_and_scan(file_stem, doc.source):
        case None, _:
            pass

        case None, IncompleteScan():
            diagnostics.append(
                lsp.Diagnostic(
                    message="File is malformed",
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=0),
                    ),
                    severity=lsp.DiagnosticSeverity.Warning,
                )
            )
            pass

        case validator, scan_result:
            diagnostics.extend(get_diagnostics(validator, scan_result))

    server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: lsp.HoverParams) -> lsp.Hover | None:
    """Get item description on hover."""
    pos = params.position
    uri = params.text_document.uri
    document_uri = params.text_document.uri
    document = ls.workspace.get_text_document(document_uri)

    file_stem = Path(uri).stem
    validator, _ = get_validator_and_scan(file_stem, document.source)

    if validator is None:
        return None

    if not (
        path := get_schema_path_from_token_position(
            position=pos, instance_document=document.source
        )
    ):
        return None

    # This is needed to get a dereferenced schema we can walk through
    schema = next(validator.iter_errors({})).schema
    description = get_description_from_path(path=path, schema=cast(Schema, schema))
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


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
