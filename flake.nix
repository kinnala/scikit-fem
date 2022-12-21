{
  description = "scikit-fem: Simple finite element assemblers";

  inputs = {
    nixpkgs = { url = "github:nixos/nixpkgs/nixpkgs-unstable"; };
  };

  outputs = inputs:
    let
      pkgs = import inputs.nixpkgs { system = "x86_64-linux"; };
      testrequirements = with pkgs; (ps: with ps; [
          numpy
          scipy
          meshio
          matplotlib
          pytest
          pyamg
          mypy
          flake8
          sphinx
          autograd
      ]);
      devrequirements = with pkgs; (ps: with ps; [
          numpy
          scipy
          meshio
          matplotlib
          ipython
          pytest
          # pygmsh
          # gmsh
          pyamg
          mypy
          flake8
          # dmsh
          sphinx
          # glvis
          autograd
          pep517
          twine
          pip
      ]);
    in {
      devShells.x86_64-linux.py38 = inputs.nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python38.withPackages testrequirements) ];
      };
      devShells.x86_64-linux.py39 = inputs.nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python39.withPackages testrequirements) ];
      };
      devShells.x86_64-linux.default = inputs.nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python310.withPackages devrequirements) ];
      };
      devShells.x86_64-linux.py311 = inputs.nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python311.withPackages testrequirements) ];
      };
    };
}
