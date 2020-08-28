---
title: 'scikit-fem: A Python package for finite element assembly'
tags:
  - Python
  - numerics
  - finite element method
authors:
  - name: Tom Gustafsson
    orcid: 0000-0003-1611-5032
    affiliation: 1
  - name: G. D. McBain
    orcid: 0000-0002-1904-122X
    affiliation: 2
affiliations:
 - name: Department of Mathematics and Systems Analysis, Aalto University
   index: 1
 - name: Memjet North Ryde Pty Ltd, Macquarie Park, NSW, Australia
   index: 2
date: 4 June 2020
bibliography: paper.bib
---

# Summary

Partial differential equations (PDEs)—such as the Navier–Stokes equations in
fluid mechanics, the Maxwell equations in electromagnetism, and the Schrödinger
equation in quantum mechanics—are the basic building blocks of modern physics
and engineering.  The finite element method (FEM) is a flexible computational
technique for the discretization and solution of PDEs, especially in the case
of complex spatial domains.

Conceptually, the FEM transforms a time-independent (or temporally discretized)
PDE into a system of linear equations $Ax=b$.  `scikit-fem` is a lightweight
Python library for the creation, or *assembly*, of the finite element matrix $A$
and vector $b$.  The user loads a computational mesh, picks suitable basis
functions, and provides the PDE's weak formulation [@fenicsbook].  This results
in sparse matrices and vectors compatible with the SciPy [@scipy] ecosystem.

# Purpose and prior art

There exist several open source packages and frameworks that implement the
finite element method.  `scikit-fem` was developed as a simple and lightweight
alternative to the existing Python packages with a focus on computational
experimentation and custom PDE-based model development.  We rely on pure
interpreted Python code on top of the NumPy–SciPy base which makes `scikit-fem`
easy to install and portable across multiple operating systems.  The reliance on
plain NumPy arrays and SciPy sparse matrices enables interoperability with
various packages in the Python ecosystem such as meshio [@meshio], pacopy
[@pacopy], and pyamg [@pyamg].

In contrast to NGSolve [@ngsolve], FEniCS [@fenics], Firedrake [@firedrake],
SfePy [@sfepy], and GetFEM [@getfem], `scikit-fem` contains no compiled code
making the installation quick and straightforward.  We specifically target
finite element assembly instead of encapsulating the entire finite element
analysis from pre- to postprocessing into a single framework.  As a consequence,
we cannot provide an end-to-end experience when it comes to, e.g., specific
physical models or distributed computing.  Our aim is to be generic in terms of
PDEs and, hence, support a variety of finite element schemes.  Currently
`scikit-fem` includes basic support for $H^1$-, $H(\mathrm{div})$-,
$H(\mathrm{curl})$-, and $H^2$-conforming problems as well as various
nonconforming schemes.

`scikit-fem` accepts weak forms that depend on the values and the derivatives of
the trial and the test functions, their high-order derivatives, the local mesh
parameter, nonuniform material or coefficient fields defined at the quadrature
points, or any existing finite element solutions.  Iterations related to, e.g.,
nonlinear problems (Newton's method and the variants, parameter continuation) or
adaptive mesh refinement (evaluation of functionals and the marking strategy)
should be implemented by the user although we provide basic tools such as
interpolation routines and conforming mesh refinement, and examples by using
them.  The same applies to boundary conditions: the linear system $(A, b)$ is
provided as such and eliminating or penalizing the correct degrees-of-freedom,
implementing inhomogeneous or periodic boundary conditions should be done
separately either by using the various helper routines of `scikit-fem` or by
other means.  `scikit-fem` has no explicit support for distributed computing
although it could be used as a building block in parallel computations such as
parameter sweeps or domain decomposition techniques.

# Examples and enabled work

The documentation of `scikit-fem` contains over 30 examples that demonstrate the
library and its use. The results of some of the examples are highlighted in
\autoref{fig:examples}.  Several publications already utilize computational
results from `scikit-fem`, e.g., @mcbain2018, @gustafsson2019, and
@gustafsson2020.  In addition, `scikit-fem` is used in a recently published
Python package for battery modelling [@pybamm].

![(Top left.) A combination of triangular and quadrilateral elements is used to solve the linear elastic contact problem. (Top right.) The lowest order tetrahedral Nédélec element is used to solve the $H(\mathrm{curl})$-conforming model problem $\nabla \times \nabla \times E + E = f$. The color corresponds to the magnitude of the vector field $E$. (Bottom.) The Taylor–Hood element is used to solve the Navier–Stokes flow over a backward-facing step for different Reynolds numbers.  The first and last two-dimensional figures were generated using `scikit-fem`'s wrapping of matplotlib [@hunter2007]; three-dimensional postprocessing is more specialized and usually left to export of, e.g., VTK or XDMF formats, via meshio [@meshio] for subsequent rendering in, e.g., ParaView [@ahrens2005] as done in the second figure.\label{fig:examples}](examples.png)

# Acknowledgements

Tom Gustafsson has received external funding from the Finnish Cultural
Foundation and the Academy of Finland (decision nr. 324611) while working on the
project.

# References
