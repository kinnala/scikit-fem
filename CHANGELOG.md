# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New-style form constructors `BilinearForm`, `LinearForm`, and `Functional`
- `MortarMapping` for supporting mortar methods in 2D
- `skfem.io.json` for serialization of meshes to/from json-files
- `ElementLinePp`, p-th order one-dimensional elements
- `ElementQuadP`, p-th order quadrilateral elements
- `ElementQuadDG` for transforming H1 elements to DG elements
- `ElementTriMini`, MINI-element for Stokes problems
- `ElementComposite` for using many elements in one bilinear form
- `ElementQuadS2`, quadratic Serendipity element
- `Basis.find_dofs` for finding degree-of-freedom indices
- `Mesh.define_boundary` for defining named boundaries
- `Mesh.from_basis` for defining high-order meshes
- `Basis.split` for splitting multicomponent solutions

### Deprecated
- `Basis.get_dofs` in favor of `Basis.find_dofs`

### Changed
- Renamed `skfem.importers` to `skfem.io`
- Renamed `skfem.models.helpers` to `skfem.helpers`
- `skfem.utils.solve` will now expand the solutions of eigenvalue problems
- `Basis.interpolate` returns `DiscreteField` objects instead ndarray tuples
- `Basis.interpolate` works now for vectorial elements

## [0.4.1] - 2020-01-19

### Added
- Additional keyword arguments to `skfem.utils.solve` get passed on to linear solvers

### Fixed
- Made `skfem.visuals.matplotlib` Python 3.6 compatible

## [0.4.0] - 2020-01-03

### Changed
- Renamed `GlobalBasis` to `Basis`
- Moved all `Mesh.plot` and `Mesh.draw` methods to `skfem.visuals` module
- Made matplotlib an optional dependency
