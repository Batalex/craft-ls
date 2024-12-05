"""Define the language server features."""

from pathlib import Path

from lsprotocol import types
from pygls.server import LanguageServer

from craft_ls.core import get_diagnostics, validators

server = LanguageServer(name="craft-ls", version="0.1.0")


# TODO(prod): Bypass this feature
@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def test_completions(params: types.CompletionParams):
    """A simple completion feature to make sure the language server is running."""
    items = []
    document = server.workspace.get_text_document(params.text_document.uri)
    current_line = document.lines[params.position.line].strip()
    if current_line.endswith("hello."):
        items = [
            types.CompletionItem(label="world"),
            types.CompletionItem(label="friendss"),
        ]
    return types.CompletionList(is_incomplete=False, items=items)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def on_opened(params: types.DidOpenTextDocumentParams):
    """Parse each document when it is opened."""
    doc = server.workspace.get_text_document(params.text_document.uri)
    uri = params.text_document.uri
    version = params.text_document.version

    file_stem = Path(uri).stem
    validator = validators.get(file_stem, None)
    diagnostics = [
        # TODO(prod): remove this diag
        types.Diagnostic(
            message=f"File type: {Path(uri).stem}\n",
            severity=types.DiagnosticSeverity.Information,
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=0),
            ),
        ),
    ]
    if validator := validators.get(file_stem, None):
        diagnostics.extend(get_diagnostics(validator, doc.source))

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def on_changed(params: types.DidOpenTextDocumentParams):
    """Parse each document when it is changed."""
    doc = server.workspace.get_text_document(params.text_document.uri)
    uri = params.text_document.uri
    version = params.text_document.version

    file_stem = Path(uri).stem
    validator = validators.get(file_stem, None)
    diagnostics = [
        # TODO(prod): remove this diag
        types.Diagnostic(
            message=f"File type: {Path(uri).stem}\n",
            severity=types.DiagnosticSeverity.Information,
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=0),
            ),
        ),
    ]
    if validator := validators.get(file_stem, None):
        diagnostics.extend(get_diagnostics(validator, doc.source))

    if diagnostics:
        server.publish_diagnostics(uri=uri, version=version, diagnostics=diagnostics)


# @server.feature(
#     types.TEXT_DOCUMENT_DIAGNOSTIC,
#     types.DiagnosticOptions(
#         identifier="pull-diagnostics",
#         inter_file_dependencies=False,
#         workspace_diagnostics=True,
#     ),
# )
# def document_diagnostic(params: types.DocumentDiagnosticParams):
#     """Return diagnostics for the requested document"""
#     # logging.info("%s", params)

#     if (uri := params.text_document.uri) not in ls.diagnostics:
#         return

#     version, diagnostics = ls.diagnostics[uri]
#     result_id = f"{uri}@{version}"

#     if result_id == params.previous_result_id:
#         return types.UnchangedDocumentDiagnosticReport(result_id)

#     return types.FullDocumentDiagnosticReport(items=diagnostics, result_id=result_id)


@server.feature(types.EXIT)
def on_exit(*_) -> None:
    """Handle clean up on exit."""
    server.shutdown()


@server.feature(types.SHUTDOWN)
def on_shutdown(*_) -> None:
    """Handle clean up on shutdown."""
    server.shutdown()


def start() -> None:
    """Start the server."""
    server.start_io()


if __name__ == "__main__":
    start()
