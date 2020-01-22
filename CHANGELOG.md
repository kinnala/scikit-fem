# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Renamed skfem.importers to skfem.io.
- Basis.interpolate and Element.gbasis return now DiscreteField objects instead
  of plain tuples of ndarrays.

### Added
- MortarBasis and MortarPair that support mortar methods in 2D.
- Serialization of meshes to json-files via skfem.io.json.
- ElementQuadDG for transforming H1 elements to DG elements.

## [0.4.1] - 2020-01-19

### Added
- Additional keyword arguments to skfem.utils.solve get passed on to solvers.

### Fixed
- Made skfem.visuals.matplotlib Python 3.6 compatible.

## [0.4.0] - 2020-01-03

### Changed
- Renamed GlobalBasis to Basis.
- Moved all Mesh.plot and Mesh.draw methods to skfem.visuals module.
- Made matplotlib an optional dependency.
