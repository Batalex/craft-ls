# craft-ls

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/batalex/craft-ls/ci.yaml)

Get on\
[![PyPI - Version](https://img.shields.io/pypi/v/craft-ls)](https://pypi.org/project/craft-ls/)
[![FlakeHub](https://img.shields.io/badge/FlakeHub-5277C3)](https://flakehub.com/flake/Batalex/craft-ls)
[![Snap - Version](https://img.shields.io/snapcraft/v/craft-ls/latest/edge)](https://snapcraft.io/craft-ls)
[![VSCode Marketplace](https://vsmarketplacebadges.dev/version-short/abatisse.craft-ls.svg)](https://marketplace.visualstudio.com/items?itemName=abatisse.craft-ls)

`craft-ls` is a [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) implementation for *craft[^1] tools.

`craft-ls` enables editors that support the LSP to get quality of life improvements while working on *craft configuration files.

## Features

| Feature              | Snapcraft | Rockcraft | Charmcraft[^1] |
| :------------------- | :-------: | :-------: | :------------: |
| Diagnostics          |    ✅     |    ✅     |       ✅       |
| Symbol documentation |    ✅     |    ✅     |       ❌       |
| Autocompletion       |    ❌     |    ❌     |       ❌       |

https://github.com/user-attachments/assets/e4b831b5-dcac-4efd-aabb-d3040899b52b

## Usage

### Installation

Using `uv` or `pipx`

```shell
uv tool install craft-ls

pipx install craft-ls
```

### Setup

#### Helix

```toml
# languages.toml
[[language]]
name = "yaml"
language-servers = ["craft-ls"]

[language-server.craft-ls]
command = "craft-ls"
```

#### VSCode

The VSCode extension can be installed from the marketplace.
It requires a Python 3.12 interpreter.
If not automatically picked, you may configure it using the following key:

```json
"craft-ls.interpreter": [
  "/usr/bin/python3.12"
]
```

TBD: neovim

## Roadmap

Project availability:

- Python package
- Snap
- Nix flake
- VSCode extension

Features:

- Diagnostics
- Autocompletion **on typing**
- Symbol documentation

Ecosystem:

- Encourage *craft tools to refine their JSONSchemas even further

[^1]: snapcraft, rockcraft and partial support for charmcraft (all-in-one `charmcraft.yaml` only)
