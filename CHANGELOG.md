# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Additional keyword arguments to skfem.utils.solve get passed on to solvers.
- Add MortarBasis and MortarPair for mortaring in 2D.

### Changed
- Renamed skfem.importers to skfem.io.

## [0.4.0] - 2020-01-03

### Changed
- Renamed GlobalBasis to Basis.
- Moved all Mesh.plot and Mesh.draw methods to skfem.visuals module.
- Made matplotlib an optional dependency.
