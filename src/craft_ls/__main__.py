"""Application module."""

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Application entrypoint."""
    from craft_ls import server

    logger.info("Starting Craft-ls")
    server.start()


if __name__ == "__main__":
    main()
