# scikit-fem

`scikit-fem` is a pure Python 3.8+ library for performing [finite element
assembly](https://en.wikipedia.org/wiki/Finite_element_method). Its main
purpose is the transformation of bilinear forms into sparse matrices and linear
forms into vectors.

<a href="https://colab.research.google.com/github/kinnala/scikit-fem-notebooks/blob/master/ex1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://scikit-fem.readthedocs.io/" alt="Documentation"><img src="https://readthedocs.org/projects/pip/badge/?version=stable" /></a>
<a href="https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e" alt="status"><img src="https://joss.theoj.org/papers/4120aba1525403e6d0972f4270d7b61e/status.svg" /></a>
<a href="https://pypi.org/project/scikit-fem/" alt="PyPI"><img src="https://img.shields.io/pypi/v/scikit-fem" /></a>

The library

- has minimal dependencies
- contains no compiled code
- supports one-dimensional, triangular, quadrilateral, tetrahedral and hexahedral finite elements
- includes special elements such as Raviart-Thomas, Nédélec, MINI, Crouzeix-Raviart, Argyris, ...

If you use the library in your research, you can cite the following article:
```
@article{skfem2020,
  doi = {10.21105/joss.02369},
  year = {2020},
  volume = {5},
  number = {52},
  pages = {2369},
  author = {Tom Gustafsson and G. D. McBain},
  title = {scikit-fem: A {P}ython package for finite element assembly},
  journal = {Journal of Open Source Software}
}
```

## Installation

The most recent release can be installed simply by
```
pip install scikit-fem[all]
```
Remove `[all]` to not install the optional dependencies `meshio` for mesh
input/output, and `matplotlib` for creating simple visualizations.
The minimal dependencies are `numpy` and `scipy`.
You can also try the library in browser through [Google Colab](https://colab.research.google.com/github/kinnala/scikit-fem-notebooks/blob/master/ex1.ipynb).

## Examples

Solve the Poisson problem (see also [`ex01.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex01.py)):
```python
from skfem import *
from skfem.helpers import dot, grad

# create the mesh
mesh = MeshTri().refined(4)
# or, with your own points and elements:
# mesh = MeshTri(points, elements)

basis = Basis(mesh, ElementTriP1())

@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))

@LinearForm
def rhs(v, _):
    return 1. * v

A = laplace.assemble(basis)
b = rhs.assemble(basis)

# Dirichlet boundary conditions
A, b = enforce(A, b, D=mesh.boundary_nodes())

# solve the linear system
x = solve(A, b)

# plot using matplotlib
mesh.plot(x, shading='gouraud', colorbar=True).show()
# or, save to external file:
# mesh.save('output.vtk', point_data={'solution': x})
```

Meshes can be initialized manually, loaded from external files using
[meshio](https://github.com/nschloe/meshio), or created with the help of special
constructors:

```python
import numpy as np
from skfem import MeshLine, MeshTri, MeshTet

mesh = MeshLine(np.array([0., .5, 1.]))
mesh = MeshTri(
    np.array([[0., 0.],
              [1., 0.],
              [0., 1.]]).T,
    np.array([[0, 1, 2]]).T,
)
mesh = MeshTri.load("docs/examples/meshes/square.msh")  # requires meshio
mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),) * 3))
```

We support [many common finite
elements](https://github.com/kinnala/scikit-fem/blob/master/skfem/element/__init__.py#L51).
Below the stiffness matrix is assembled using second-order tetrahedra:

```python
from skfem import Basis, ElementTetP2

basis = Basis(mesh, ElementTetP2())  # quadratic tetrahedron
A = laplace.assemble(basis)  # type: scipy.sparse.csr_matrix
```

More examples can be found in the [gallery](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html).


## Benchmark

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


## Documentation

The project is documented using Sphinx under `docs/`.
Built version can be found from [Read the Docs](https://scikit-fem.readthedocs.io/en/latest/).
Here are direct links to additional resources:

- [Examples from our test suite](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html)
- [Examples from the FEniCS tutorial](https://github.com/gdmcbain/fenics-tuto-in-skfem)

## Getting help

If you encounter an issue you can use GitHub issue tracker.  If you cannot find help from the documentation,
you can use the GitHub Discussions to [ask questions](https://github.com/kinnala/scikit-fem/discussions).
Try to provide a snippet of code which fails
and include also the version of the library you are
using.  The version can be found as follows:
```python
import skfem; print(skfem.__version__)
```

## Dependencies

The minimal dependencies for installing `scikit-fem` are
[numpy](https://numpy.org/) and [scipy](https://www.scipy.org/).  In addition,
many
[examples](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html) use
[matplotlib](https://matplotlib.org/) for visualization and
[meshio](https://github.com/nschloe/meshio) for loading/saving different mesh
file formats.  Some examples demonstrate the use of other external packages;
see `requirements.txt` for a list of test dependencies.

## Testing

The tests are run by GitHub Actions.  The `Makefile` in the repository root has
targets for running the testing container locally using `docker`.  For example,
`make test_py38` runs the tests using `py38` branch from
[kinnala/scikit-fem-docker-action](https://github.com/kinnala/scikit-fem-docker-action).
The releases are tested in
[kinnala/scikit-fem-release-tests](https://github.com/kinnala/scikit-fem-release-tests).

## Licensing

The contents of `skfem/` and the PyPI package `scikit-fem` are licensed under
the 3-clause BSD license.  Some examples under `docs/examples/` or snippets
in the documentation may have a different license.

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural
Foundation](https://skr.fi/).  Versions 2.0.0+ were prepared while working in
a project funded by Academy of Finland (decisions nr. 324611 and 338341).
The approach used in the finite element assembly has been inspired by the [work
of A. Hannukainen and
M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

## Contributing

We are happy to welcome any contributions to the library.  Reasonable projects
for first timers include:

- Reporting a [bug](https://github.com/kinnala/scikit-fem/issues)
- Writing an [example](https://github.com/kinnala/scikit-fem/tree/master/docs/examples)
- Improving the [tests](https://github.com/kinnala/scikit-fem/tree/master/tests)
- Suggesting improvements in the [documentation](https://scikit-fem.readthedocs.io).

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*

## Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with respect to documented and/or tested features.

### [10.0.2] - 2024-09-03

- Fixed: `Mesh.load` returned incorrect orientation for tagged holes

### [10.0.1] - 2024-08-06

- Fixed: `Mesh.load` returned incorrect orientation for some Gmsh
  meshes with tagged interfaces

### [10.0.0] - 2024-07-25

- Fixed: `Mesh.p2e` returned incorrect incidence
- Fixed: `InteriorFacetBasis.get_dofs` did not return all edge DOFs for 3D elements
- Added: The lowest order, one point integration rule for tetrahedral elements
- Added: `asm` will now wrap functions with three arguments using `BilinearForm`,
  functions with two arguments using `LinearForm`, etc.
- Changed: Initializing `Basis` for `ElementTetP0` without specifying
  `intorder` or `quadrature` will now automatically fall back to a one
  point integration rule
- Changed: Default tags (left, right, top, ...) are no more
  added automatically during mesh initialization, as a workaround you
  can add them explicitly by calling `mesh = mesh.with_defaults()`
- Changed: All indices within the library are now using `np.int32` for
  around 10% boost in performance and the corresponding reduction in
  memory usage for larger meshes - theoretically, the largest possible
  tetrahedral tensor product mesh is now roughly 550 ** 3 = 166 M vertices
  and 993 M elements, without the facet indexing wrapping over 2 ** 32

### [9.1.1] - 2024-04-23

- Fixed: Tests now pass with `numpy==2.0rc1`
- Fixed: `MappingAffine` now uses lazy evaluation also for element
  mappings, in addition to boundary mappings
- Fixed: `MeshTet.init_tensor` uses significantly less memory for
  large meshes
- Fixed: `Mesh.load` uses less memory when loading and matching tags
- Added: `Basis` has new optional `disable_doflocs` kwarg which
  can be set to `True` to avoid computing `Basis.doflocs`, to reduce memory
  usage

### [9.0.1] - 2024-01-12

- Fixed: `ElementVector` now works also for split_bases/split_indices in case `mesh.dim() != elem.dim`

### [9.0.0] - 2023-12-24

- Removed: Python 3.7 support
- Removed: `MappingMortar` and `MortarFacetBasis` in favor of
  `skfem.supermeshing`
- Deprecated: `skfem.visuals.glvis`; current version is broken and no
  replacement is being planned
- Added: Python 3.12 support
- Added: `Mesh.load` supports new keyword arguments
  `ignore_orientation=True` and `ignore_interior_facets=True` which
  will both speed up the loading of larger three-dimensional meshes by
  ignoring facet orientation and all tags not on the boundary,
  respectively.
- Added: `skfem.supermeshing` (requires `shapely>=2`) for creating quadrature
  rules for interpolating between two 1D or 2D meshes.
- Added: `Mesh.remove_unused_nodes`
- Added: `Mesh.remove_duplicate_nodes`
- Added: `Mesh.remove_elements` now supports passing any subdomain
  reference through `Mesh.normalize_elements`; subdomains and
  boundaries are also properly preserved
- Fixed: `MeshTet` uniform refine was reindexing subdomains incorrectly
- Fixed: `MeshDG.draw` did not work; now calls `Basis.draw` which
  works for any mesh topology
- Fixed: `FacetBasis` now works with `MeshTri2`, `MeshQuad2`,
  `MeshTet2` and `MeshHex2`
- Fixed: `ElementGlobal` now uses outward normals to initialize DOFs
  on boundary facets

### [8.1.0] - 2023-06-16

- Added: `ElementTriHHJ0` and `ElementTriHHJ1` matrix finite elements
  for implementing the Hellan-Hermann-Johnson mixed method (see [`ex48.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex48.py))
- Added: `ElementHexSkeleton0`, piecewise constant element on the skeleton
  of a hexahedral mesh
- Added: `ElementHexC1`, C1-continuous hexahedral element
- Added: `Mesh.restrict` now preserves subdomains and boundaries
- Added: `skfem.utils.mpc` for setting multipoint constraints
- Added: `Basis.get_dofs` now supports fetching DOFs at specific nodes
  through kwarg `nodes`
- Added: `DofsView.sort` for sorting a set of DOFs returned by
  `Basis.get_dofs`
- Added: `CellBasis.with_elements` as a shortcut for initiating a new Basis with subset of elements
- Added: `Mesh.refined` now preserves subdomains in adaptive mesh refinement
- Fixed: `Mesh.refined` used to modify the element connectivity of the original mesh

### [8.0.0] - 2022-12-16

- Removed: The deprecated `Basis.find_dofs` method, see `Basis.get_dofs` for a
  replacement
- Added: Renamed `ElementTetN0` to `ElementTriN1` and added alias for backwards
  compatibility
- Added: `ElementQuadN1`, first order H(curl) conforming quadrilateral element
- Added: `ElementTriN1`, first order H(curl) conforming triangle element
- Added: `ElementTriN2`, second order H(curl) conforming triangle element
- Added: `ElementTetSkeletonP0`, extension of `ElementTriSkeletonP0` to
  tetrahedral meshes
- Added: `Mesh.restrict` which returns a new mesh given a subset of elements or subdomain
- Added: `Mesh.trace` which turns facets into a trace mesh
- Added: `skfem.utils.bmat`, a variant of `scipy.sparse.bmat` which adds the
  indices of the different blocks as an attribute `out.blocks`
- Added: Plane strain to plane stress mapping under `skfem.models.elasticity`
- Added: Various methods such as `Basis.interpolate` and `Basis.project`
  now support specifying `dtype` and using complex fields
- Fixed:  Python 3.11
- Fixed: `Basis.intepolate` did not work properly with `ElementComposite`
  when the basis was defined only for a subset of elements
- Fixed: `Basis.split` worked incorrectly for `ElementVector` and multiple DOFs
  of same type
- Fixed: Caching of `ElementQuadP` basis for reused quadrature points
  did not work correctly
- Deprecated: `MappingMortar` and `MortarFacetBasis` in favor of
  plain `FacetBasis` and the upcoming `skfem.experimental.supermeshing` (see [`ex04.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex04.py))

### [7.0.1] - 2022-08-03

- Fixed: Updated changelog was missing.

### [7.0.0] - 2022-08-03

- Changed: Removed the optimization of using `DiscreteField.is_zero` in the
  helpers to skip the evaluation of zero components in `ElementComposite` to
  improve type stability with respect to the size of the underlying numpy
  arrays; this is technically a backwards incompatible change and might affect
  self-created helper functions
- Deprecated: `FacetBasis.trace` in favor of `Basis.interpolator` and `Basis.project`
- Added: Output of `Basis.interpolator` supports trailing axes; can be now
  passed to `Basis.project` for (inexact) interpolation between meshes
- Added: Renamed `ElementTriRT0` to `ElementTriRT1` and added alias for
  backwards compatibility
- Added: Renamed `ElementTetRT0` to `ElementTetRT1` and added alias for
  backwards compatibility
- Added: Renamed `ElementQuadRT0` to `ElementQuadRT1` and added alias for
  backwards compatibility
- Added: `ElementTriRT2`, the second order Raviart-Thomas element
- Added: `ElementHexRT1`, the first order Raviart-Thomas element for hexahedral meshes
- Added: `Basis.project` now better supports `ElementComposite`
- Added: `solver_iter_cg`, a simple pure Python conjugate gradient solver for
  environments that do not have sparse solver libraries (e.g., Pyodide)
- Added: `ElementTriP2B` and `ElementTriP1B`, new aliases for `ElementTriMini`
  and `ElementTriCCR`
- Added: `ElementTriP1G` and `ElementTriP2G`, variants of `ElementTriP1` and
  `ElementTriP2` using `ElementGlobal` so that second derivatives are available
  (useful, e.g., for stabilized methods and the Stokes problem)
- Added: `Basis.plot3`, a wrapper to `skfem.visuals.*.plot3`
- Fixed: Calculation of size in `Basis.__repr__` was slow and incorrect
- Fixed: Subclasses of `ElementHdiv` did not work together with `FacetBasis`

### [6.0.0] - 2022-03-15

- Changed: `DiscreteField` is now a subclass of `ndarray` instead of
  `NamedTuple` and, consequently, the components of `DiscreteField` cannot no
  more be indexed inside forms like `u[1]` (use `u.grad` instead)
- Changed: Writing `w['u']` and `w.u` inside the form definition is now
  equivalent (previously `w.u == w['u'].value`)
- Changed: `Mesh.draw` now uses `matplotlib` by default, calling
  `Mesh.draw("vedo")` uses `vedo`
- Changed: `skfem.visuals.matplotlib` now uses `jet` as the default colormap
- Changed: `BoundaryFacetBasis` is now an alias of `FacetBasis` instead of
  other way around
- Deprecated: `DiscreteField.value` remains for backwards-compatibility but is
  now deprecated and can be dropped
- Added: `Mesh.plot`, a wrapper to `skfem.visuals.*.plot`
- Added: `Basis.plot`, a wrapper to `skfem.visuals.*.plot`
- Added: `Basis.refinterp` now supports vectorial fields
- Added: `skfem.visuals.matplotlib.plot` now has a basic quiver plot for vector
  fields
- Added: `Mesh.facets_around` which constructs a set of facets around a
  subdomain
- Added: `Mesh.save` and `load` now preserve the orientation of boundaries and
  interfaces
- Added: `OrientedBoundary` which is a subclass of `ndarray` for facet index
  arrays with the orientation information (0 or 1 per facet) available as
  `OrientedBoundary.ori`
- Added: `FacetBasis` will use the facet orientations (if present) to calculate
  traces and normal vectors
- Added: `skfem.visuals.matplotlib.draw` will visualize the orientations if
  `boundaries=True` is given
- Added: `Mesh.facets_satisfying` allows specifying the keyword argument
  `normal` for orienting the resulting interface
- Added: `FacetBasis` constructor now has the keyword argument `side` which
  allows changing the side of the facet used to calculate the basis function
  values and gradients
- Added: `Basis.boundary` method to quickly initialize the corresponding
  `FacetBasis`
- Fixed: Improvements to backwards compatibility in `asm`/`assemble` keyword
  arguments
- Fixed: Save format issue with meshio 5.3.0+
- Fixed: `CellBasis` did not properly support `elements` argument
- Fixed: `Basis.interpolate` did not properly interpolate all components of
  `ElementComposite`

### [5.2.0] - 2021-12-27

- Added: `Basis.project`, a more general and easy to use alternative for
  `projection`
- Added: `Basis` and `FacetBasis` kwargs `elements` and `facets` can now be a
  string refering to subdomain and boundary tags
- Added: `ElementQuadRT0`, lowest-order quadrilateral Raviart-Thomas element
- Fixed: `Functional` returned only the first component for forms with
  non-scalar output

### [5.1.0] - 2021-11-30

- Added: `skfem.helpers.mul` for matrix multiplication
- Added: `Basis.split` will now also split `ElementVector` into its components
- Fixed: `ElementDG` was not included in the wildcard import
- Fixed: Automatic visualization of `MeshTri2` and `MeshQuad2` in Jupyter
  notebooks raised exception

### [5.0.0] - 2021-11-21

- Changed: `meshio` is now an optional dependency
- Changed: `ElementComposite` uses `DiscreteField()` to represent zero
- Added: Support more argument types in `Basis.get_dofs`
- Added: Version information in `skfem.__version__`
- Added: Preserve `Mesh.boundaries` during uniform refinement of `MeshLine1`,
  `MeshTri1` and `MeshQuad1`
- Fixed: Refinement of quadratic meshes will now fall back to the refinement
  algorithm of first-order meshes instead of crashing
- Fixed: Edge cases in the adaptive refine of `MeshTet1` that failed to produce
  a valid mesh
- Deprecated: `Basis.find_dofs` in favor of `Basis.get_dofs`
- Deprecated: Merging `DofsView` objects via `+` and `|` because of surprising
  behavior in some edge cases

### [4.0.1] - 2021-10-15

- Fixed: `MappingIsoparametric` can now be pickled

### [4.0.0] - 2021-09-27

- Added: `Mesh.save`/`Mesh.load` now exports/imports `Mesh.subdomains` and
  `Mesh.boundaries`
- Added: `Mesh.load` now optionally writes any mesh data to a list passed via
  the keyword argument `out`, e.g., `out=data` where `data = ['point_data']`
- Added: `Mesh.load` (and `skfem.io.meshio.from_file`) now supports the
  additional keyword argument `force_meshio_type` for loading mesh files that
  have multiple element types written in the same file, one element type at
  a time
- Added: `asm` will now accept a list of bases, assemble the same form using
  all of the bases and sum the result (useful for jump terms and mixed meshes,
  see [`ex41.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex41.py)
- Added: `Mesh.with_boundaries` now allows the definition of internal boundaries/interfaces
  via the flag `boundaries_only=False`
- Added: `MeshTri1DG`, `MeshQuad1DG`, `MeshHex1DG`, `MeshLine1DG`; new mesh
  types for describing meshes with a discontinuous topology, e.g., periodic
  meshes (see [`ex42.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex42.py))
- Added: `ElementHexDG` for transforming hexahedral H1 elements to DG/L2 elements.
- Added: `ElementTriP1DG`, `ElementQuad1DG`, `ElementHex1DG`,
  `ElementLineP1DG`; shorthands for `ElementTriDG(ElementTriP1())` etc.
- Added: `ElementTriSkeletonP0` and `ElementTriSkeletonP1` for defining
  Lagrange multipliers on the skeleton mesh (see [`ex40.py`](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex40.py))
- Added: `TrilinearForm` for assembling a sparse 3-tensor, e.g., when dealing
  with unknown material data
- Added: `MeshTri.oriented` for CCW oriented triangular meshes which can be
  useful for debugging or interfacing to external tools
- Added: partial support for `MeshWedge1` and `ElementWedge1`, the lowest order
  wedge mesh and element
- Added: `ElementTriP3`, cubic triangular Lagrange element
- Added: `ElementTriP4`, quartic triangular Lagrange element
- Added: `ElementTri15ParamPlate`, 15-parameter nonconforming triangular element for plates
- Added: `ElementTriBDM1`, the lowest order Brezzi-Douglas-Marini element
- Added: `Mesh.draw().show()` will now visualize any mesh interactively (requires [vedo](https://vedo.embl.es/))
- Added: Adaptive refinement for `MeshTet1`
- Fixed: `MappingIsoparametric` is now about 2x faster for large meshes thanks
  to additional caching
- Fixed: `MeshHex2.save` did not work properly
- Fixed: `Mesh.load` ignores unparseable `cell_sets` inserted by `meshio` in MSH 4.1
- Changed: `Mesh` string representation is now more informative
- Changed: `Form.assemble` no more allows keyword arguments with `list` or
  `dict` type: from now on only `DiscreteField` or 1d/2d `ndarray` objects are
  allowed and 1d `ndarray` is passed automatically to `Basis.interpolate` for
  convenience
- Changed: `MeshLine` is now a function which initializes `MeshLine1`
  and not an alias to `MeshLine1`
- Changed: `FacetBasis` is now a shorthand for `BoundaryFacetBasis` and no
  longer initializes `InteriorFacetBasis` or `MortarFacetBasis` if the keyword
  argument `side` is passed to the constructor
- Removed: the deprecated `Mesh.define_boundary` method

### [3.2.0] - 2021-08-02

- Added: `ElementTriCCR` and `ElementTetCCR`, conforming Crouzeix-Raviart finite elements
- Fixed: `Mesh.mirrored` returned a wrong mesh when a point other than the origin was used
- Fixed: `MeshLine` constructor accepted only numpy arrays and not plain Python lists
- Fixed: `Mesh.element_finder` (and `CellBasis.probes`, `CellBasis.interpolator`) was not working properly for a small number of elements (<5) or a large number of input points (>1000)
- Fixed: `MeshTet` and `MeshTri.element_finder` are now more robust against degenerate elements
- Fixed: `Mesh.element_finder` (and `CellBasis.probes`, `CellBasis.interpolator`) raises exception if the query point is outside of the domain

### [3.1.0] - 2021-06-18

- Added: `Basis`, a shorthand for `CellBasis`
- Added: `CellBasis`, a new preferred name for `InteriorBasis`
- Added: `BoundaryFacetBasis`, a new preferred name for `ExteriorFacetBasis`
- Added: `skfem.utils.penalize`, an alternative to `condense` and `enforce` for
  essential boundary conditions
- Added: `InteriorBasis.point_source`, with `ex38`
- Added: `ElementTetDG`, similar to `ElementTriDG` for tetrahedral meshes
- Fixed: `MeshLine1.element_finder` 

### [3.0.0] - 2021-04-19

- Added: Completely rewritten `Mesh` base class which is "immutable" and uses
  `Element` classes to define the ordering of nodes; better support for
  high-order and other more general mesh types in the future
- Added: New quadratic mesh types: `MeshTri2`, `MeshQuad2`, `MeshTet2` and `MeshHex2`
- Added: `InteriorBasis.probes`; like `InteriorBasis.interpolator` but returns a matrix
  that operates on solution vectors to interpolate them at the given points
- Added: More overloads for `DiscreteField`, e.g., multiplication, summation
  and subtraction are now explicitly supported inside the form definitions
- Added: `MeshHex.to_meshtet` for splitting hexahedra into tetrahedra
- Added: `MeshHex.element_finder` for interpolating finite element solutions
  on hexahedral meshes via `InteriorBasis.interpolator`
- Added: `Mesh.with_boundaries`, a functional replacement to
  `Mesh.define_boundary`, i.e. defining boundaries via Boolean lambda function
- Added: `Mesh.with_subdomains` for defining subdomains via Boolean lambda function
- Added: `skfem.utils.projection`, a replacement of `skfem.utils.project`
  with a different, more intuitive order of arguments
- Added: `skfem.utils.enforce` for setting essential boundary conditions by
  changing matrix rows to zero and diagonals to one.
- Deprecated: `skfem.utils.project` in favor of `skfem.utils.projection`
- Deprecated: `Mesh.define_boundary` in favor of `Mesh.with_boundaries`
- Removed: `Mesh.{refine,scale,translate}`; the replacements are `Mesh.{refined,scaled,translated}`
- Removed: `skfem.models.helpers`; available as `skfem.helpers`
- Removed: `DiscreteField.{f,df,ddf,hod}`; available as `DiscreteField.{value,grad,hess,grad3,...}`
- Removed: Python 3.6 support
- Removed: `skfem.utils.L2_projection`
- Removed: `skfem.utils.derivative`
- Changed: `Mesh.refined` no more attempts to fix the indexing of `Mesh.boundaries` after refine
- Changed: `skfem.utils.solve` now uses `scipy.sparse.eigs` instead of `scipy.sparse.eigsh` by default;
  the old behavior can be retained by explicitly passing `solver=solver_scipy_eigs_sym()`
- Fixed: High memory usage in `skfem.visuals.matplotlib` related to 1D plotting

### [2.5.0] - 2021-02-13

- Deprecated: `side` keyword argument to `FacetBasis` in favor of the more
  explicit `InteriorFacetBasis` and `MortarFacetBasis`.
- Added: `InteriorFacetBasis` for integrating over the interior facets, e.g.,
  evaluating error estimators with jumps and implementing DG methods.
- Added: `MortarFacetBasis` for integrating over the mortar mesh.
- Added: `InteriorBasis.with_element` for reinitializing an equivalent basis
  that uses a different element.
- Added: `Form.partial` for applying `functools.partial` to the form function
  wrapped by `Form`.
- Fixed: Include explicit Python 3.9 support.

### [2.4.0] - 2021-01-20

- Deprecated: List and tuple keyword argument types to `asm`.
- Deprecated: `Mesh2D.mirror` in favor of the more general `Mesh.mirrored`.
- Deprecated: `Mesh.refine`, `Mesh.scale` and `Mesh.translate` in favor of
  `Mesh.refined`, `Mesh.scaled` and `Mesh.translated`.
- Added: `Mesh.refined`, `Mesh.scaled`, and `Mesh.translated`. The new methods
  return a copy instead of modifying `self`.
- Added: `Mesh.mirrored` for mirroring a mesh using a normal and a point.
- Added: `Functional` now supports forms that evaluate to vectors or other
  tensors.
- Added: `ElementHex0`, piecewise constant element for hexahedral meshes.
- Added: `FacetBasis.trace` for restricting existing solutions to lower
  dimensional meshes on boundaries or interfaces.
- Fixed: `MeshLine.refined` now correctly performs adaptive refinement of
  one-dimensional meshes.

### [2.3.0] - 2020-11-24

- Added: `ElementLineP0`, one-dimensional piecewise constant element.
- Added: `skfem.helpers.curl` now calculates the rotated gradient for
  two-dimensional elements.
- Added: `MeshTet.init_ball` for meshing a ball.
- Fixed: `ElementQuad0` was not compatible with `FacetBasis`.

### [2.2.3] - 2020-10-16

- Fixed: Remove an unnecessary dependency.

### [2.2.2] - 2020-10-15

- Fixed: Make the preconditioner in `TestEx32` more robust.

### [2.2.1] - 2020-10-15

- Fixed: Remove `tests` from the PyPI distribution.

### [2.2.0] - 2020-10-14

- Deprecated: `L2_projection` will be replaced by `project`.
- Deprecated: `derivative` will be replaced by `project`.
- Added: `MeshTet.element_finder` and `MeshLine.element_finder` for using
  `InteriorBasis.interpolator`.
- Added: `ElementTriCR`, the nonconforming Crouzeix-Raviart element for Stokes flow.
- Added: `ElementTetCR`, tetrahedral nonconforming Crouzeix-Raviart element.
- Added: `ElementTriHermite`, an extension of `ElementLineHermite` to triangular
  meshes.
- Fixed: Fix `Mesh.validate` for unsigned `Mesh.t`.

### [2.1.1] - 2020-10-01

- Fixed: Further optimizations to `Mesh3D.boundary_edges`: tested to run on a laptop
  with over 10 million elements.

### [2.1.0] - 2020-09-30

- Added: `ElementHex2`, a triquadratic hexahedral element.
- Added: `MeshTri.init_circle`, constructor for a circle mesh.
- Fixed: `Mesh3D.boundary_edges` (and, consequently, `Basis.find_dofs`) was slow
  and used lots of memory due to an exhaustive search of all edges.

### [2.0.0] - 2020-08-21

- Deprecated: `project` will only support functions like `lambda x: x[0]`
  instead of `lambda x, y, z: x` in the future.
- Added: Support for complex-valued forms: `BilinearForm` and `LinearForm` now take
  an optional argument `dtype` which defaults to `np.float64`
  but can be also `np.complex64`.
- Added: `Dofs.__or__` and `Dofs.__add__`, for merging degree-of-freedom sets
  (i.e. `Dofs` objects) using `|` and `+` operators.
- Added: `Dofs.drop` and `Dofs.keep`, for further filtering the degree-of-freedom sets
- Removed: Support for old-style decorators `bilinear_form`, `linear_form`, and
  `functional` (deprecated since 1.0.0).
- Fixed: `FacetBasis` did not initialize with `ElementQuadP`.

### [1.2.0] - 2020-07-07

- Added: `MeshQuad._splitquads` aliased as `MeshQuad.to_meshtri`.
- Added: `Mesh.__add__`, for merging meshes using `+` operator: duplicated nodes are
  joined.
- Added: `ElementHexS2`, a 20-node quadratic hexahedral serendipity element.
- Added: `ElementLineMini`, MINI-element for one-dimensional mesh.
- Fixed: `Mesh3D.boundary_edges` was broken in case of hexahedral meshes.
- Fixed: `skfem.utils.project` did not work for `ElementGlobal`.

### [1.1.0] - 2020-05-18

- Added: `ElementTetMini`, MINI-element for tetrahedral mesh.
- Fixed: `Mesh3D.boundary_edges` incorrectly returned all edges where both nodes are on
  the boundary.

### [1.0.0] - 2020-04-22

- Deprecated: Old-style form constructors `bilinear_form`, `linear_form`, and `functional`.
- Changed: `Basis.interpolate` returns `DiscreteField` objects instead of ndarray tuples.
- Changed: `Basis.interpolate` works now properly for vectorial and high-order elements
  by interpolating all components and higher order derivatives.
- Changed: `Form.assemble` accepts now any keyword arguments (with type `DiscreteField`)
  that are passed over to the forms.
- Changed: Renamed `skfem.importers` to `skfem.io`.
- Changed: Renamed `skfem.models.helpers` to `skfem.helpers`.
- Changed: `skfem.utils.solve` will now expand also the solutions of eigenvalue problems.
- Added: New-style form constructors `BilinearForm`, `LinearForm`, and `Functional`.
- Added: `skfem.io.json` for serialization of meshes to/from json-files.
- Added: `ElementLinePp`, p-th order one-dimensional elements.
- Added: `ElementQuadP`, p-th order quadrilateral elements.
- Added: `ElementQuadDG` for transforming quadrilateral H1 elements to DG elements.
- Added: `ElementQuadBFS`, Bogner-Fox-Schmit element for biharmonic problems.
- Added: `ElementTriMini`, MINI-element for Stokes problems.
- Added: `ElementComposite` for using multiple elements in one bilinear form.
- Added: `ElementQuadS2`, quadratic Serendipity element.
- Added: `ElementLineHermite`, cubic Hermite element for Euler-Bernoulli beams.
- Added: `Mesh.define_boundary` for defining named boundaries.
- Added: `Basis.find_dofs` for finding degree-of-freedom indices.
- Added: `Mesh.from_basis` for defining high-order meshes.
- Added: `Basis.split` for splitting multicomponent solutions.
- Added: `MortarMapping` with basic support for mortar methods in 2D.
- Added: `Basis` constructors now accept `quadrature` keyword argument for specifying
  a custom quadrature rule.
