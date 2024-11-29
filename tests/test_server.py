import sys
from textwrap import dedent

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
    """Ensure that the server implements diagnostics correctly."""
    # Given
    test_uri = "file:///path/to/snapcraft.yaml"
    text_content = dedent(
        """
        name: my_snap
        parts:
            foo:
                name: name
        """
    )

    # When
    client.text_document_did_open(
        params=types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=test_uri,
                language_id="yaml",
                version=1,
                text=text_content,
            )
        )
    )
    await client.wait_for_notification(types.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)

    # Then
    assert (diagnostics := client.diagnostics.get(test_uri, []))
    assert any(
        "'plugin' is a required property" in diagnostic.message
        for diagnostic in diagnostics
    )
