![scikit-fem](https://github.com/kinnala/scikit-fem/blob/master/logo.png?raw=true)

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/115345426.svg)](https://zenodo.org/badge/latestdoi/115345426)
[![status](https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e/status.svg)](https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e)


`scikit-fem` is a lightweight Python 3.6+ library for performing [finite element
assembly](https://en.wikipedia.org/wiki/Finite_element_method). Its main purpose
is the transformation of bilinear forms into sparse matrices and linear forms
into vectors.  The library supports triangular, quadrilateral, tetrahedral and
hexahedral meshes as well as one-dimensional problems.

The library fills an important gap in the spectrum of finite element codes.
The library is *lightweight* meaning that it has *minimal dependencies*.
It contains *no compiled code* meaning that it's *easy to install* and
use on all platforms that support NumPy.  Despite being fully interpreted, the
code has a reasonably *good performance*.

*The following benchmark (`docs/examples/performance.py`) demonstrates the time
spent on finite element assembly in comparison to the time spent on linear
solve.  The given numbers were calculated using a ThinkPad X1 Carbon laptop (7th
gen).  Note that the timings are only illustrative as they depend on, e.g., the
type of element used, the number of quadrature points used, the type of linear
solver, and the complexity of the forms.  This benchmark solves the Laplace
equation using linear tetrahedral elements and the default direct sparse solver
of `scipy.sparse.linalg.spsolve`.*

| Degrees-of-freedom | Time spent on assembly (s) | Time spent on linear solve (s) |
| --- | --- | --- |
| 64 | 0.001719430962111801 | 0.0008631939999759197 |
| 125 | 0.0022377190180122852 | 0.0009978360030800104 |
| 216 | 0.004175334994215518 | 0.0016233620117418468 |
| 512 | 0.006831337988842279 | 0.0018000249983742833 |
| 1000 | 0.027329800999723375 | 0.006487983977422118 |
| 1728 | 0.02035172702744603 | 0.00894418201642111 |
| 4096 | 0.04277183400699869 | 0.054088378965388983 |
| 8000 | 0.08925403101602569 | 0.39822074701078236 |
| 15625 | 0.17675437999423593 | 3.5216876140329987 |
| 32768 | 0.41025503899436444 | 24.167141190962866 |
| 64000 | 0.8814217780018225 | 138.6927448490169 |
| 125000 | 1.7890089299762622 | nan |
| 262144 | 4.044992835028097 | nan |
| 512000 | 8.037139102991205 | nan |
| 1030301 | 20.241967056994326 | nan |


## Examples

Forms are defined using an intuitive syntax:

```python
from skfem import *
from skfem.helpers import dot, grad

@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))
```

Meshes can be initialized manually, loaded from external files using
[meshio](https://github.com/nschloe/meshio), or created with the help of special
constructors:

```python
import numpy as np

mesh = MeshLine(np.array([0.0, 0.5, 1.0]))
mesh = MeshTri.load("docs/examples/square.msh")
mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),) * 3))
```

We support [many common finite
elements](https://github.com/kinnala/scikit-fem/tree/master/skfem/element).
Below the stiffness matrix is assembled using second-order tetrahedra:

```python
basis = InteriorBasis(mesh, ElementTetP2())
A = laplace.assemble(basis)  # type: scipy.sparse.csr_matrix
```

More examples can be found in the [gallery](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html).

## Documentation

The project is documented using Sphinx.  A recent version of the documentation
can be found from [Read the Docs](https://scikit-fem.readthedocs.io/en/latest/).

## Installation

The most recent release can be installed simply by `pip install scikit-fem`.
For more cutting edge features, you can clone this repository.

## Dependencies

The minimal dependencies for installing `scikit-fem` are
[numpy](https://numpy.org/), [scipy](https://www.scipy.org/) and
[meshio](https://github.com/nschloe/meshio).  In addition, many
[examples](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html) use
[matplotlib](https://matplotlib.org/) for visualization.  Some examples
demonstrate the use of other external packages; see our [CI job
definition](https://github.com/kinnala/scikit-fem/blob/master/.travis.yml) for a
full list of test dependencies.

## Licensing

The contents of `skfem/` and the PyPI package `scikit-fem` are licensed under
the 3-clause BSD license.  Some examples under `docs/examples/` have a different
license, see `LICENSE.md` for more information.

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural Foundation](https://skr.fi/). The approach used in the finite element assembly has been inspired by the [work of A. Hannukainen and M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

## Contributing

We are happy to welcome any contributions to the library.  Reasonable projects
for first timers include:

- Filing out a [bug report](https://github.com/kinnala/scikit-fem/issues).
- Writing an [example](https://github.com/kinnala/scikit-fem/tree/master/docs/examples)
- Improving the [tests](https://github.com/kinnala/scikit-fem/tree/master/tests).
- Finding typos in the documentation.

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*

## In literature

The library has been used in the preparation of the following scientific works:

- Gustafsson, T., Stenberg, R., & Videman, J. (2020). Nitsche's method for Kirchhoff plates. arXiv preprint [arXiv:2007.00403](https://arxiv.org/abs/2007.00403).
- Gustafsson, T., Stenberg, R., & Videman, J. (2020). On Nitsche's method for elastic contact problems. SIAM Journal on Scientific Computing, 42(2), B425–B446. arXiv preprint [arXiv:1902.09312](https://arxiv.org/abs/1902.09312).
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

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Unreleased

### [2.0.0] - 2020-08-21

#### Added
- Support for complex-valued forms: `BilinearForm` and `LinearForm` now take
  an optional argument `dtype` which defaults to `np.float64`
  but can be also `np.complex64`
- `Dofs.__or__` and `Dofs.__add__`, for merging degree-of-freedom sets
  (i.e. `Dofs` objects) using `|` and `+` operators
- `Dofs.drop` and `Dofs.keep`, for further filtering the degree-of-freedom sets

### Removed
- Support for old-style decorators `bilinear_form`, `linear_form`, and
  `functional` (deprecated since 1.0.0)

#### Fixed
- `FacetBasis` did not initialize with `ElementQuadP`

#### Deprecated
- `skfem.utils.project` will support functions like `lambda x: x[0]`
  instead of `lambda x, y, z: x` in the future

### [1.2.0] - 2020-07-07

#### Added
- `Mesh.__add__`, for merging meshes using `+` operator: duplicated nodes are
  joined
- `ElementHexS2`, a 20-node quadratic hexahedral serendipity element
- `ElementLineMini`, MINI-element for one-dimensional mesh

#### Fixed
- `Mesh3D.boundary_edges` was broken in case of hexahedral meshes
- `skfem.utils.project` did not work for `ElementGlobal`

#### Changed
- `MeshQuad._splitquads` aliased as `MeshQuad.to_meshtri`: should not be private

### [1.1.0] - 2020-05-18

#### Added
- `ElementTetMini`, MINI-element for tetrahedral mesh

#### Fixed
- `Mesh3D.boundary_edges` incorrectly returned all edges where both nodes are on
  the boundary

### [1.0.0] - 2020-04-22

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
- `Basis.find_dofs` for finding degree-of-freedom indices
- `Mesh.from_basis` for defining high-order meshes
- `Basis.split` for splitting multicomponent solutions
- `MortarMapping` with basic support for mortar methods in 2D
- `Basis` constructors now accept `quadrature` keyword argument for specifying
  a custom quadrature rule

#### Deprecated
- Old-style form constructors `bilinear_form`, `linear_form`, and `functional`.

#### Changed
- `Basis.interpolate` returns `DiscreteField` objects instead of ndarray tuples
- `Basis.interpolate` works now properly for vectorial and high-order elements
  by interpolating all components and higher order derivatives
- `Form.assemble` accepts now any keyword arguments (with type `DiscreteField`)
  that are passed over to the forms
- Renamed `skfem.importers` to `skfem.io`
- Renamed `skfem.models.helpers` to `skfem.helpers`
- `skfem.utils.solve` will now expand also the solutions of eigenvalue problems
  

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
