with import <nixpkgs> {};

(let
  pacopy016 = python37.pkgs.buildPythonPackage rec {
    # not part of nixpkgs
    pname = "pacopy";
    version = "0.1.6";
    format = "pyproject";
    src = python37.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "0rq5yfmq5516giyqsflm3bjirbfhigydd1vlxc554jnn13n3wisr";
    };
    doCheck = false;
    meta = {
      homepage = "https://github.com/nschloe/pacopy/";
      description = "Numerical continuation in Python";
    };
  };
in python37.withPackages (ps: with ps; [
  numpy
  scipy
  meshio
  matplotlib
  pyamg
  ipython
  pytest
  sphinx
  sphinx_rtd_theme
  flake8
  twine
  pacopy016
])).env
