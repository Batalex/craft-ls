from pygls import server
from lsprotocol import types

craft_lserver = server.LanguageServer(name="craft-ls", version="0.1.0")


@craft_lserver.feature(types.TEXT_DOCUMENT_COMPLETION)
def completions(params: types.CompletionParams):
    items = []
    document = craft_lserver.workspace.get_text_document(params.text_document.uri)
    current_line = document.lines[params.position.line].strip()
    if current_line.endswith("hello."):
        items = [
            types.CompletionItem(label="world"),
            types.CompletionItem(label="friend"),
        ]
    return types.CompletionList(is_incomplete=False, items=items)


def start() -> None:
    craft_lserver.start_io()


if __name__ == "__main__":
    start()
