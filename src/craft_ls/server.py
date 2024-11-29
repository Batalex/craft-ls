"""Define the language server features."""

from pathlib import Path

from lsprotocol import types
from pygls.server import LanguageServer

from craft_ls import __version__
from craft_ls.core import get_diagnostics, validators

server = LanguageServer(name="craft-ls", version=__version__)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def on_opened(params: types.DidOpenTextDocumentParams):
    """Parse each document when it is opened."""
    # doc = server.workspace.get_text_document(params.text_document.uri)
    uri = params.text_document.uri
    version = params.text_document.version
    source = params.text_document.text

    file_stem = Path(uri).stem
    validator = validators.get(file_stem, None)
    diagnostics = [
        types.Diagnostic(
            message=f"Running craft-ls {__version__}.",
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=0),
            ),
            severity=types.DiagnosticSeverity.Information,
        )
    ]
    if validator := validators.get(file_stem, None):
        diagnostics.extend(get_diagnostics(validator, source))

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(params: types.DidOpenTextDocumentParams):
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

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completion(ls: LanguageServer, params: types.CompletionParams):
    """Placeholder."""
    return [
        types.CompletionItem(label="hello"),
        types.CompletionItem(label="world"),
    ]


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
