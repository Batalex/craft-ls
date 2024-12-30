"""Tasks definition for the nox runner."""

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"
nox.options.sessions = ["fmt", "lint"]


@nox.session()
def fmt(session: nox.Session) -> None:
    """
    Format source code.
    """
    session.run(
        "uv",
        "sync",
        "--frozen",
        "--only-group",
        "fmt",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("ruff", "format", "src", "tests")


@nox.session()
def lint(session: nox.Session) -> None:
    """
    Lint source code.
    """
    session.run(
        "uv",
        "sync",
        "--frozen",
        "--group",
        "lint",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("ruff", "check", "--fix", "src")
    session.run("ruff", "check", "--fix", "tests")
    session.run("mypy", "src")


@nox.session()
def tests(session: nox.Session) -> None:
    """
    Run the unit tests.
    """
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--group",
        "unit",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("python", "-m", "pytest", *session.posargs)
