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

| Feature                | Snapcraft | Rockcraft | Charmcraft[^1] |
| :--------------------- | :-------: | :-------: | :------------: |
| Diagnostics            |    ✅     |    ✅     |       ✅       |
| Documentation on hover |    ✅     |    ✅     |       ✅       |
| Symbols                |    ✅     |    ✅     |       ✅       |
| Autocompletion         |    ✅     |    ✅     |       ✅       |

https://github.com/user-attachments/assets/e4b831b5-dcac-4efd-aabb-d3040899b52b

## Usage

### Installation

#### Using the snap

```shell
sudo snap install craft-ls --edge
```

#### Using a Nix flake

```shell
nix run github:Batalex/craft-ls
```

#### Using a Python environment

Using `uv`

```shell
uv tool install craft-ls
```

Using `pipx`

```shell
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

#### Visual Studio Code

The Visual Studio Code [extension](https://marketplace.visualstudio.com/items?itemName=abatisse.craft-ls) can be installed right from the editor.
It requires a local `craft-ls` installation (see previous section).
If not automatically picked, you may configure it using the following key:

```json
{
  "craft-ls.serverPath": "/home/user/.local/bin/craft-ls"
}
```

#### Neovim

Add the following to your Neovim configuration (e.g., `~/.config/nvim/init.lua`
or a plugin file):

```lua
vim.lsp.config("craft_ls", {
  cmd = { "craft-ls" },
  filetypes = { "yaml" },
  root_markers = {
    "snapcraft.yaml",
    "rockcraft.yaml",
    "charmcraft.yaml",
    "snap",
    ".git",
  },
})
vim.lsp.enable("craft_ls")
```

[^1]: snapcraft, rockcraft and partial support for charmcraft (all-in-one `charmcraft.yaml` only)
