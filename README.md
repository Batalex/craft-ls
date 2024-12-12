# craft-ls

`craft-ls` is a [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) implementation for *craft[^1] tools.

`craft-ls` enables editors that support the LSP to get quality of life improvements while working on *craft configuration files.

## Features

TBD

## Getting started

TBD

### Installation

TBD

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

## Roadmap

Project:

- Unit testing
- CI with publishing:
  - Pypi
  - Snapstore
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
use flake
source .venv/bin/activate
export CRAFT_LS_DEV=true
```


[^1]: only snapcraft so far
