import sys
from textwrap import dedent

import pytest
import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient

from craft_ls.types_ import MISSING_TYPE_MSG


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
    uri = "file:///path/to/snapcraft.yaml"
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
                uri=uri,
                language_id="yaml",
                version=1,
                text=text_content,
            )
        )
    )
    await client.wait_for_notification(types.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)

    # Then
    assert (diagnostics := client.diagnostics.get(uri, []))
    assert any(MISSING_TYPE_MSG in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.asyncio
async def test_completion_at_root_key(client: LanguageClient):
    """Verify incomplete keys successfully suggest top-level fields (like confinement).

    In addition, we check that we get some minimal level of feature even if we don't have
    fully determined the schema to use.
    """
    # Given
    uri = "file:///workspace/snapcraft.yaml"
    text_content = dedent(
        """
        confi
        """
    )

    # When
    client.text_document_did_open(
        params=types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="yaml",
                version=1,
                text=text_content,
            )
        )
    )

    # Trigger completion right at the tip of 'confi' at line 0 (+offset 1)) col 5
    response = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=1, character=5),
        )
    )

    # Then
    assert response is not None
    labels = [item.label for item in getattr(response, "items", [])]
    assert "confinement" in labels


@pytest.mark.asyncio
async def test_completion_inside_value(client: LanguageClient):
    """Verify value placements isolate sub-enums instead of root parameters."""
    # Given
    uri = "file:///workspace/snapcraft.yaml"
    text_content = dedent(
        """
        base: core24
        confinement: st
        """
    )

    # When
    client.text_document_did_open(
        params=types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="yaml",
                version=1,
                text=text_content,
            )
        )
    )

    # Trigger completion past the colon at 'st' at line 1 (+offset 1)) col 15
    response = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=2, character=15),
        )
    )

    # Then
    assert response is not None
    labels = [item.label for item in getattr(response, "items", [])]
    assert "strict" in labels
    assert "confinement" not in labels
