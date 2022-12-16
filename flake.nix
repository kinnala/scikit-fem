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
      devShells.x86_64-linux.default =
        (pkgs.python310.withPackages requirements).env;
    };
}
