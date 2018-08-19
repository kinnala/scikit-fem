# scikit-fem

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)
[![Documentation Status](https://readthedocs.org/projects/scikit-fem/badge/?version=latest)](https://scikit-fem.readthedocs.io/en/latest/?badge=latest) [![Join the chat at https://gitter.im/scikit-fem/Lobby](https://badges.gitter.im/scikit-fem/Lobby.svg)](https://gitter.im/scikit-fem/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Easy to use finite element assemblers and the related tools.

## Installation

The most recent release can be installed using `pip install scikit-fem`.
For more recent developments, you can just clone this repository.

## Feature highlights

- Transforms bilinear forms into sparse matrices and linear forms into vectors
- Supports triangular, quadrilateral, tetrahedral and hexahedral meshes
- Supports various different finite element families including H1-, H2-, L2-, H(div)-, and
  H(curl)-conforming elements
- No complex dependencies: Most of the functionality require only SciPy and
  NumPy

## Usage

The following code solves the Poisson's equation in the unit square:
```python
from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
basis = InteriorBasis(m, e)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

I = m.interior_nodes()

x = 0*b
x[I] = solve(*condense(A, b, I=I))

m.plot3(x)
m.show()
```
Please see the directory *examples* for more instructions.

## Contributors

- Tom Gustafsson (Maintainer)
- [Geordie McBain](https://github.com/gdmcbain)

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.*
