"""Application module."""

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Application entrypoint."""
    from craft_ls import __version__, server

    parser = argparse.ArgumentParser(prog="craft-ls")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.parse_args()

    logger.info("Starting Craft-ls")
    server.start()


if __name__ == "__main__":
    main()
