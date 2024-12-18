import sys

import pytest
import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[sys.executable, "src/craft_ls/server.py"]
    ),
)
async def client(lsp_client: LanguageClient):
    # Setup
    params = types.InitializeParams(capabilities=types.ClientCapabilities())
    await lsp_client.initialize_session(params)

    yield

    # Teardown
    await lsp_client.shutdown_session()


@pytest.mark.asyncio
async def test_diagnostic_on_open(client: LanguageClient):
    """Ensure that the server implements capabilities correctly."""
    # Given
    test_uri = "file:///path/to/snapcraft.yaml"

    # When
    client.text_document_did_open(
        params=types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=test_uri, language_id="yaml", version=1, text=""
            )
        )
    )
    # await client.wait_for_notification(types.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)

    # Then
    assert test_uri in client.diagnostics


@pytest.mark.asyncio
async def test_completions(client: LanguageClient):
    """Ensure that the server implements completions correctly."""

    results = await client.text_document_completion_async(
        params=types.CompletionParams(
            position=types.Position(line=1, character=0),
            text_document=types.TextDocumentIdentifier(uri="file:///path/to/file.txt"),
        )
    )
    assert results is not None

    if isinstance(results, types.CompletionList):
        items = results.items
    else:
        items = results

    labels = [item.label for item in items]
    assert labels == ["hello", "world"]
