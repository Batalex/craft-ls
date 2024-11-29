from pygls import server

craft_lserver = server.LanguageServer(name="craft-ls", version="0.1.0")


def start() -> None:
    craft_lserver.start_io()


if __name__ == "__main__":
    start()
