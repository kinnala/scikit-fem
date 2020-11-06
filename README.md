<p align="center">
<img src="https://user-images.githubusercontent.com/973268/93522777-ac28dd00-f93a-11ea-8733-4ca8e62ab09d.png" width="45%">
</p>

<p align="center">
<a href="https://pypi.org/project/scikit-fem/" alt="PyPI"><img src="https://img.shields.io/pypi/v/scikit-fem" /></a>
<a href="https://anaconda.org/conda-forge/scikit-fem" alt="Conda"><img src="https://img.shields.io/conda/vn/conda-forge/scikit-fem" /></a>
<a href="https://pypi.org/project/scikit-fem/" alt="PyPI - Python Version"><img src="https://img.shields.io/pypi/pyversions/scikit-fem" /></a>
<a href="https://scikit-fem.readthedocs.io/" alt="Documentation"><img src="https://readthedocs.org/projects/pip/badge/?version=stable" /></a>
<a href="https://opensource.org/licenses/BSD-3-Clause" alt="License"><img src="https://img.shields.io/badge/license-BSD%203--Clause-blue.svg" /></a>
<a href="https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e" alt="status"><img src="https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e/status.svg" /></a>
<a href="https://doi.org/10.5281/zenodo.1420510" alt="DOI"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1420510.svg" /></a>
<a href="https://github.com/kinnala/scikit-fem/actions" alt="Tests"><img src="https://github.com/kinnala/scikit-fem/workflows/tests/badge.svg" /></a>
<a href="https://github.com/kinnala/scikit-fem-release-tests/actions" alt="Release tests"><img src="https://github.com/kinnala/scikit-fem-release-tests/workflows/release%20tests/badge.svg" /></a>
</p>


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

| Degrees-of-freedom | Assembly (s) | Linear solve (s) |
| --- | --- | --- |
| 4096 | 0.04805 | 0.04241 |
| 8000 | 0.09804 | 0.16269 |
| 15625 | 0.20347 | 0.87741 |
| 32768 | 0.46399 | 5.98163 |
| 64000 | 1.00143 | 36.47855 |
| 125000 | 2.05274 | nan |
| 262144 | 4.48825 | nan |
| 512000 | 8.82814 | nan |
| 1030301 | 18.25461 | nan |

## Installation

The most recent release can be installed simply by `pip install scikit-fem`.
Optionally you can use `conda install -c conda-forge scikit-fem`.

## Examples

Forms are defined using an intuitive syntax:

```python
from skfem import BilinearForm
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
from skfem import MeshLine, MeshTri, MeshTet

mesh = MeshLine(np.array([0.0, 0.5, 1.0]))
mesh = MeshTri.load("docs/examples/square.msh")
mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),) * 3))
```

We support [many common finite
elements](https://github.com/kinnala/scikit-fem/blob/master/skfem/element/__init__.py#L51).
Below the stiffness matrix is assembled using second-order tetrahedra:

```python
from skfem import InteriorBasis, ElementTetP2

basis = InteriorBasis(mesh, ElementTetP2())
A = laplace.assemble(basis)  # type: scipy.sparse.csr_matrix
```

More examples can be found in the [gallery](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html).

## Documentation

The project is documented using Sphinx under `docs/`.  Built version of the
documentation can be found from [Read the
Docs](https://scikit-fem.readthedocs.io/en/latest/).

## Getting help

If you encounter an issue and cannot find help from the documentation,
you can use the Github issue tracker to [ask questions using the question label](https://github.com/kinnala/scikit-fem/issues?q=label%3Aquestion).
Try to provide a snippet of code which fails
and include also the version of the library you are
using.  The version can be found as follows:
```
python -c "import pkg_resources; print(pkg_resources.get_distribution('scikit-fem').version)"
```

## Dependencies

The minimal dependencies for installing `scikit-fem` are
[numpy](https://numpy.org/), [scipy](https://www.scipy.org/) and
[meshio](https://github.com/nschloe/meshio).  In addition, many
[examples](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html) use
[matplotlib](https://matplotlib.org/) for visualization.  Some examples
demonstrate the use of other external packages; see `requirements.txt` for a
list of test dependencies.

## Testing

The tests are run by Github Actions.  The `Makefile` in the repository root has
targets for running the testing container locally using `docker`.  For example,
`make test_py38` runs the tests using `py38` branch from
[kinnala/scikit-fem-docker-action](https://github.com/kinnala/scikit-fem-docker-action).
The releases are tested in
[kinnala/scikit-fem-release-tests](https://github.com/kinnala/scikit-fem-release-tests).

## Licensing

The contents of `skfem/` and the PyPI package `scikit-fem` are licensed under
the 3-clause BSD license.  Some examples under `docs/examples/` have a different
license, see `LICENSE.md` for more information.

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural
Foundation](https://skr.fi/).  Versions 2.0.0+ were prepared while working in a
project funded by the [Academy of
Finland](https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_HakKuvaus2&CLICKED_ON=&HAKNRO1=324611&UILANG=en).
The approach used in the finite element assembly has been inspired by the [work
of A. Hannukainen and
M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

## Contributing

We are happy to welcome any contributions to the library.  Reasonable projects
for first timers include:

- Filing out a [bug report](https://github.com/kinnala/scikit-fem/issues).
- Writing an [example](https://github.com/kinnala/scikit-fem/tree/master/docs/examples)
- Improving the [tests](https://github.com/kinnala/scikit-fem/tree/master/tests).
- Finding typos in the documentation.

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*

## Citing the library

You may use the following BibTeX entry:
```
@article{skfem2020,
  doi = {10.21105/joss.02369},
  url = {https://doi.org/10.21105/joss.02369},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2369},
  author = {Tom Gustafsson and G. D. McBain},
  title = {scikit-fem: A Python package for finite element assembly},
  journal = {Journal of Open Source Software}
}
```
Use the Zenodo DOIs only if you want to cite a specific version,
e.g., to ensure reproducibility.

## In literature

The library has been used in the preparation of the following scientific works:

- Gustafsson, T., Stenberg, R., & Videman, J. (2020). Nitsche's method for Kirchhoff plates. arXiv preprint [arXiv:2007.00403](https://arxiv.org/abs/2007.00403).

- Gustafsson, T., & McBain, G. D. (2020). scikit-fem: A Python package for finite element assembly. Journal of Open Source Software, 52(5). [Open access](https://doi.org/10.21105/joss.02369).
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

## Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Unreleased

- Added: `skfem.helpers.curl` now calculates the rotated gradient for
  two-dimensional elements.
- Added: `MeshTet.init_ball` for meshing a ball.

### [2.2.3] - 2020-10-16

- Fixed: Remove an unnecessary dependency.

### [2.2.2] - 2020-10-15

- Fixed: Make the preconditioner in `TestEx32` more robust.

### [2.2.1] - 2020-10-15

- Fixed: Remove `tests` from the PyPI distribution.

### [2.2.0] - 2020-10-14

- Fixed: Fix `Mesh.validate` for unsigned `Mesh.t`.
- Added: `MeshTet.element_finder` and `MeshLine.element_finder` for using
  `InteriorBasis.interpolator`.
- Added: `ElementTriCR`, the nonconforming Crouzeix-Raviart element for Stokes flow.
- Added: `ElementTetCR`, tetrahedral nonconforming Crouzeix-Raviart element.
- Added: `ElementTriHermite`, an extension of `ElementLineHermite` to triangular
  meshes.
- Deprecated: `L2_projection` will be replaced by `project`.
- Deprecated: `derivative` will be replaced by `project`.

### [2.1.1] - 2020-10-01

- Fixed: Further optimizations to `Mesh3D.boundary_edges`: tested to run on a laptop
  with over 10 million elements.

### [2.1.0] - 2020-09-30

- Fixed: `Mesh3D.boundary_edges` (and, consequently, `Basis.find_dofs`) was slow
  and used lots of memory due to an exhaustive search of all edges
- Added: `ElementHex2`, a triquadratic hexahedral element
- Added: `MeshTri.init_circle`, constructor for a circle mesh

### [2.0.0] - 2020-08-21

- Added: Support for complex-valued forms: `BilinearForm` and `LinearForm` now take
  an optional argument `dtype` which defaults to `np.float64`
  but can be also `np.complex64`
- Added: `Dofs.__or__` and `Dofs.__add__`, for merging degree-of-freedom sets
  (i.e. `Dofs` objects) using `|` and `+` operators
- Added: `Dofs.drop` and `Dofs.keep`, for further filtering the degree-of-freedom sets
- Removed: Support for old-style decorators `bilinear_form`, `linear_form`, and
  `functional` (deprecated since 1.0.0)
- Fixed: `FacetBasis` did not initialize with `ElementQuadP`
- Deprecated: `project` will only support functions like `lambda x: x[0]`
  instead of `lambda x, y, z: x` in the future

### [1.2.0] - 2020-07-07

- Added: `Mesh.__add__`, for merging meshes using `+` operator: duplicated nodes are
  joined
- Added: `ElementHexS2`, a 20-node quadratic hexahedral serendipity element
- Added: `ElementLineMini`, MINI-element for one-dimensional mesh
- Fixed: `Mesh3D.boundary_edges` was broken in case of hexahedral meshes
- Fixed: `skfem.utils.project` did not work for `ElementGlobal`
- Changed: `MeshQuad._splitquads` aliased as `MeshQuad.to_meshtri`: should not be private

### [1.1.0] - 2020-05-18

- Added: `ElementTetMini`, MINI-element for tetrahedral mesh
- Fixed: `Mesh3D.boundary_edges` incorrectly returned all edges where both nodes are on
  the boundary

### [1.0.0] - 2020-04-22

- Added: New-style form constructors `BilinearForm`, `LinearForm`, and `Functional`
- Added: `skfem.io.json` for serialization of meshes to/from json-files
- Added: `ElementLinePp`, p-th order one-dimensional elements
- Added: `ElementQuadP`, p-th order quadrilateral elements
- Added: `ElementQuadDG` for transforming quadrilateral H1 elements to DG elements
- Added: `ElementQuadBFS`, Bogner-Fox-Schmit element for biharmonic problems
- Added: `ElementTriMini`, MINI-element for Stokes problems
- Added: `ElementComposite` for using multiple elements in one bilinear form
- Added: `ElementQuadS2`, quadratic Serendipity element
- Added: `ElementLineHermite`, cubic Hermite element for Euler-Bernoulli beams
- Added: `Mesh.define_boundary` for defining named boundaries
- Added: `Basis.find_dofs` for finding degree-of-freedom indices
- Added: `Mesh.from_basis` for defining high-order meshes
- Added: `Basis.split` for splitting multicomponent solutions
- Added: `MortarMapping` with basic support for mortar methods in 2D
- Added: `Basis` constructors now accept `quadrature` keyword argument for specifying
  a custom quadrature rule
- Deprecated: Old-style form constructors `bilinear_form`, `linear_form`, and `functional`.
- Changed: `Basis.interpolate` returns `DiscreteField` objects instead of ndarray tuples
- Changed: `Basis.interpolate` works now properly for vectorial and high-order elements
  by interpolating all components and higher order derivatives
- Changed: `Form.assemble` accepts now any keyword arguments (with type `DiscreteField`)
  that are passed over to the forms
- Changed: Renamed `skfem.importers` to `skfem.io`
- Changed: Renamed `skfem.models.helpers` to `skfem.helpers`
- Changed: `skfem.utils.solve` will now expand also the solutions of eigenvalue problems
