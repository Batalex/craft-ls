import logging

logger = logging.getLogger(__name__)


def main() -> None:
    from craft_ls import server

    print("Starting Craft-ls")
    server.start()


if __name__ == "__main__":
    main()
