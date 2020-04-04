![scikit-fem](https://github.com/kinnala/scikit-fem/blob/master/logo.png?raw=true)

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://kinnala.github.io/scikit-fem-docs)
[![Join the chat at https://gitter.im/scikit-fem/Lobby](https://badges.gitter.im/scikit-fem/Lobby.svg)](https://gitter.im/scikit-fem/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/115345426.svg)](https://zenodo.org/badge/latestdoi/115345426)

Easy to use finite element assemblers and the related tools.

> We have recently released 1.0.0 which includes a plenty of new features and
> deprecates older features (see changelog below). Unfortunately, our
> documentation is still in a fluid state and, although the goal is to maintain
> backwards-compatibility in 95% of the cases, some examples may be broken or
> emit warnings. We hope to fix everything as soon as possible. Thanks for your
> patience!

## Features

This library fills an important gap in the spectrum of finite element codes.
The library is *lightweight* meaning that it has *minimal dependencies*.
It contains *no compiled code* meaning that it's *easy to install* and
use on all platforms that support NumPy.  Despite being fully interpreted, the
code has a reasonably *good performance*.

## Examples

In the following snippet, we create a tetrahedral mesh with over 1 million
elements and assemble a discrete Laplace operator, all in just a few seconds.

```python
from skfem import *
import numpy as np

mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),) * 3))
basis = InteriorBasis(mesh, ElementTetP1())

@BilinearForm
def laplace(u, v, w):
    from skfem.helpers import d, dot
    return dot(d(u), d(v))

A = asm(laplace, basis)
```

More examples can be found in the
[documentation](https://kinnala.github.io/scikit-fem-docs/learning.html).

## Installation

The most recent release can be installed simply by `pip install scikit-fem`.

For more cutting edge features, you can clone this repository.

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural Foundation](https://skr.fi/). The approach used in the finite element assembly has been inspired by the [work of A. Hannukainen and M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

A list of people who have directly contributed to the project:

- Tom Gustafsson (Author)
- [Geordie McBain](https://github.com/gdmcbain)

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*

## In literature

The library has been used in the preparation of the following scientific works:

- Gustafsson, T., Stenberg, R., & Videman, J. (2020). On Nitsche's method for elastic contact problems. SIAM Journal on Scientific Computing, 42(2). [arXiv:1902.09312](https://arxiv.org/abs/1902.09312).
- Gustafsson, T., Stenberg, R., & Videman, J. (2019). Nitsche's Master-Slave Method for Elastic Contact Problems. [arXiv:1912.08279](https://arxiv.org/abs/1912.08279).
- McBain, G. D., Mallinson, S. G., Brown, B. R., Gustafsson, T. (2019). Three ways to compute multiport inertance. The ANZIAM Journal, 60, C140–C155.  [Open access](https://doi.org/10.21914/anziamj.v60i0.14058).
- Gustafsson, T., Stenberg, R., & Videman, J. (2019). Error analysis of Nitsche's mortar method. Numerische Mathematik, 142(4), 973–994. [Open access](https://link.springer.com/article/10.1007/s00211-019-01039-5).
- Gustafsson, T., Stenberg, R., & Videman, J. (2019). Nitsche's method for unilateral contact problems. Port. Math. 75, 189–204. arXiv preprint [arXiv:1805.04283](https://arxiv.org/abs/1805.04283).
- Gustafsson, T., Stenberg, R. & Videman, J. (2018). A posteriori estimates for conforming Kirchhoff plate elements. SIAM Journal on Scientific Computing, 40(3), A1386–A1407. arXiv preprint [arXiv:1707.08396](https://arxiv.org/abs/1707.08396).
- Gustafsson, T., Rajagopal, K. R., Stenberg, R., & Videman, J. (2018). An adaptive finite element method for the inequality-constrained Reynolds equation. Computer Methods in Applied Mechanics and Engineering, 336, 156–170. arXiv preprint [arXiv:1711.04274](https://arxiv.org/abs/1711.04274).
- Gustafsson, T., Stenberg, R., & Videman, J. (2018). A stabilised finite element method for the plate obstacle problem. BIT Numerical Mathematics, 59(1), 97–124. arXiv preprint [arXiv:1711.04166](https://arxiv.org/abs/1711.04166).
- Gustafsson, T., Stenberg, R., & Videman, J. (2017). Nitsche’s Method for the Obstacle Problem of Clamped Kirchhoff Plates. In European Conference on Numerical Mathematics and Advanced Applications, 407–415. Springer.
- Gustafsson, T., Stenberg, R., & Videman, J. (2017). A posteriori analysis of classical plate elements. Rakenteiden Mekaniikka, 50(3), 141–145. [Open access](https://rakenteidenmekaniikka.journal.fi/article/view/65004/26450).

In case you want to cite the library, you can use the DOI provided by [Zenodo](https://zenodo.org/badge/latestdoi/115345426).

## Changelog

All notable changes to this project will be documented in this section.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [Unreleased]

### [1.0.0] - 2020-04-04

#### Added
- New-style form constructors `BilinearForm`, `LinearForm`, and `Functional`
- `skfem.io.json` for serialization of meshes to/from json-files
- `ElementLinePp`, p-th order one-dimensional elements
- `ElementQuadP`, p-th order quadrilateral elements
- `ElementQuadDG` for transforming quadrilateral H1 elements to DG elements
- `ElementQuadBFS`, Bogner-Fox-Schmit element for biharmonic problems
- `ElementTriMini`, MINI-element for Stokes problems
- `ElementComposite` for using multiple elements in one bilinear form
- `ElementQuadS2`, quadratic Serendipity element
- `ElementLineHermite`, cubic Hermite element for Euler-Bernoulli beams
- `Mesh.define_boundary` for defining named boundaries
- `Mesh.from_basis` for defining high-order meshes
- `Basis.find_dofs` for finding degree-of-freedom indices
- `Basis.split` for splitting multicomponent solutions
- `MortarMapping` for supporting mortar methods in 2D

#### Deprecated
- `Basis.get_dofs` in favor of `Basis.find_dofs`
- Old-style form constructors `bilinear_form`, `linear_form`, and `functional`.

#### Changed
- Renamed `skfem.importers` to `skfem.io`
- Renamed `skfem.models.helpers` to `skfem.helpers`
- `skfem.utils.solve` will now expand also the solutions of eigenvalue problems
- `Basis.interpolate` returns `DiscreteField` objects instead ndarray tuples
- `Basis.interpolate` works now for vectorial elements

### [0.4.1] - 2020-01-19

#### Added
- Additional keyword arguments to `skfem.utils.solve` get passed on to linear solvers

#### Fixed
- Made `skfem.visuals.matplotlib` Python 3.6 compatible

### [0.4.0] - 2020-01-03

#### Changed
- Renamed `GlobalBasis` to `Basis`
- Moved all `Mesh.plot` and `Mesh.draw` methods to `skfem.visuals` module
- Made matplotlib an optional dependency
