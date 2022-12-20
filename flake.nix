{
  description = "scikit-fem: Simple finite element assemblers";

  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
      requirements = with pkgs; (ps: with ps; [
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
      devShells.x86_64-linux.py38 = nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python38.withPackages requirements) ];
      };
      devShells.x86_64-linux.py39 = nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python39.withPackages requirements) ];
      };
      devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python310.withPackages requirements) ];
      };
      devShells.x86_64-linux.py311 = nixpkgs.legacyPackages.x86_64-linux.mkShell {
        buildInputs = [ (pkgs.python311.withPackages requirements) ];
      };
    };
}
