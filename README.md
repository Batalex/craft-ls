# craft-ls

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

```
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

- Autocompletion **on typing**
- Symbol documentation

Ecosystem:

- Encourage *craft tools to refine their JSONSchemas even further

## Contributing

TBD

```
# .envrc
use flake  # if you are using nix
source .venv/bin/activate
export CRAFT_LS_DEV=true
```

[^1]: only snapcraft and rockcraft so far
