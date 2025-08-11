"""Application module."""

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Application entrypoint."""
    from craft_ls import __version__

    parser = argparse.ArgumentParser(prog="craft-ls")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers()
    parser_check = subparsers.add_parser(
        "check",
        help="Check a file and report violations",
    )
    parser_check.add_argument("file", help="File to check")
    parser_check.set_defaults(which="check")

    args = parser.parse_args()

    if not getattr(args, "which", ""):
        serve()
    elif args.which == "check":
        check(args.file)


def serve() -> None:
    """Run language server."""
    from craft_ls import server

    logger.info("Starting Craft-ls")
    server.start()


def check(file: str) -> None:
    """Run ad-hoc check."""
    from craft_ls.cli import check

    check(file)


if __name__ == "__main__":
    main()
