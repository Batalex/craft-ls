"""Tasks definition for the nox runner."""

from pathlib import Path
from urllib.request import urlopen

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"
nox.options.sessions = ["fmt", "lint"]

SNAPCRAFT_URL = "https://raw.githubusercontent.com/canonical/snapcraft/refs/heads/main/schema/snapcraft.json"
ROCKCRAFT_URL = "https://raw.githubusercontent.com/canonical/rockcraft/refs/heads/main/schema/rockcraft.json"
CHARMCRAFT_URL = "https://raw.githubusercontent.com/canonical/charmcraft/refs/heads/main/schema/charmcraft.json"


@nox.session()
def fmt(session: nox.Session) -> None:
    """Format source code."""
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
    """Lint source code."""
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
    """Run the unit tests."""
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--group",
        "unit",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("python", "-m", "pytest", *session.posargs)


@nox.session()
def build_schemas(session=nox.Session) -> None:
    """Fetch the latest schemas available and apply patches."""
    tmp_build = Path(".build")
    resources_build = Path("build")
    src = Path("src")
    tmp_build.mkdir(exist_ok=True)

    # Snapcraft
    session.log("Getting latest snapcraft schema")

    with (
        (snap := tmp_build / "snapcraft_upstream.json").open(
            "w", encoding="utf-8"
        ) as f,
        urlopen(SNAPCRAFT_URL) as g,
    ):
        f.write(g.read().decode("utf-8"))

    for patch_file in sorted(resources_build.glob("snap_*"), reverse=True):
        session.run("patch", snap, "-i", patch_file, external=True)

    # Rockcraft
    session.log("Getting latest rockcraft schema")

    with (
        (rock := tmp_build / "rockcraft_upstream.json").open(
            "w", encoding="utf-8"
        ) as f,
        urlopen(ROCKCRAFT_URL) as g,
    ):
        f.write(g.read().decode("utf-8"))

    for patch_file in sorted(resources_build.glob("rock_*"), reverse=True):
        session.run("patch", rock, "-i", patch_file, external=True)

    # Charmcraft
    session.log("Getting latest charmcraft schema")

    with (
        (charm := tmp_build / "charmcraft_upstream.json").open(
            "w", encoding="utf-8"
        ) as f,
        urlopen(CHARMCRAFT_URL) as g,
    ):
        f.write(g.read().decode("utf-8"))

    for patch_file in sorted(resources_build.glob("charm_*"), reverse=True):
        session.run("patch", charm, "-i", patch_file, external=True)

    session.log("Replacing schemas")
    snap.replace(src / "craft_ls/schemas/snapcraft.json")
    rock.replace(src / "craft_ls/schemas/rockcraft.json")
    charm.replace(src / "craft_ls/schemas/charmcraft.json")
