# scikit-fem

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)

Easy to use finite element assemblers and the related tools.

## Installation

The most recent release can be installed using `pip install scikit-fem`.
Otherwise you can just clone this repository.

## Feature highlights

- Transforms bilinear forms into sparse matrices and linear forms into vectors
- Supports triangular, quadrilateral, tetrahedral and hexahedral meshes
- Supports a nice set of finite elements including H1-, H2-, L2-, H(div)-, and
  H(curl)-conforming elements
- No complex dependencies: Most of the functionality require only SciPy and
  NumPy
- Native Python: No need to compile anything 

## Usage

The following code solves the Poisson's equation in the unit square:
```python
from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
map = MappingAffine(m)
basis = InteriorBasis(m, e, map, 2)

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
- [Geordie McBain](https://github.com/gdmcbain) (Bug reports, examples)
