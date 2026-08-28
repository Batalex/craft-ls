{
  description = "craft-ls development flake";

  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    ...
  }:
    utils.lib.eachSystem [
      "x86_64-linux"
      "aarch64-linux"
      "aarch64-darwin"
    ] (system: let
      pkgs = import nixpkgs {inherit system;};
      pythonPkgs = pkgs.python314Packages;
    in {
      packages.default = pythonPkgs.buildPythonPackage {
        pname = "craft-ls";
        version = "0.5.0";
        format = "pyproject";
        src = ./.;
        build-system = [pythonPkgs.hatchling];

        dependencies = with pythonPkgs; [
          # Python dependencies
          pygls_2
          lsprotocol_2025
          jsonschema
          pyyaml
          jsonref
          referencing
          tree-sitter
          tree-sitter-yaml
        ];
      };

      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          python314Packages.nox
        ];
      };
    });
}
