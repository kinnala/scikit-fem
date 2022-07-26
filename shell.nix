with import <nixpkgs> {};

(python38.withPackages (ps: with ps; [
  numpy
  scipy
  meshio
  matplotlib
  pyamg
  ipython
  pytest
  sphinx
  flake8
  twine
  pep517
  pip
])).env
