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
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      pythonPkgs = pkgs.python3Packages;
    in {
      packages.default = pythonPkgs.buildPythonPackage {
        pname = "craft-ls";
        version = "0.2.1";
        format = "pyproject";
        src = ./.;
        build-system = [pythonPkgs.hatchling];

        dependencies = with pythonPkgs; [
          # Python dependencies
          pygls
          lsprotocol
          jsonschema
          pyyaml
        ];
      };

      devShells.default = pkgs.mkShell {
        # inputsFrom = [self.packages.${system}.default];
        packages = with pkgs; [
          uv
          python312Packages.nox
        ];
      };
    });
}
