# craft-ls

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/batalex/craft-ls/ci.yaml)

Get on\
[![PyPI - Version](https://img.shields.io/pypi/v/craft-ls)](https://pypi.org/project/craft-ls/)
[![FlakeHub](https://img.shields.io/badge/FlakeHub-5277C3)](https://flakehub.com/flake/Batalex/craft-ls)

`craft-ls` is a [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) implementation for *craft[^1] tools.

`craft-ls` enables editors that support the LSP to get quality of life improvements while working on *craft configuration files.

## Features

- Diagnostics

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

TBD: neovim, VSCode

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

[^1]: only snapcraft and rockcraft so far
