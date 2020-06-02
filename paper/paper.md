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
  - name: Geordie McBain
    orcid: 0000-0002-1904-122X
    affiliation: 2
affiliations:
 - name: Department of Mathematics and Systems Analysis, Aalto University
   index: 1
 - name: Affiliation 2
   index: 2
date: 26 February 2020
bibliography: paper.bib
---

# Summary

Partial differential equations (PDE's) – such as Navier-Stokes equations in
fluid mechanics, Maxwell's equations in electromagnetism, and Schrödinger
equation in quantum mechanics – are the basic building blocks of modern physics
and engineering.  Finite element method (FEM) is a flexible computational
technique for the discretization and solution of PDE's, especially in the case
of complex spatial domains.

Conceptually, FEM transforms a time-independent (or temporally discretized) PDE
into a system of linear equations $Ax=b$.  `scikit-fem` is a lightweight Python
library for the creation, or *assembly*, of the finite element matrix $A$ and
vector $b$.  The user loads a computational mesh, picks suitable basis functions
from the collection, and provides the PDE's weak formulation.  This results in
sparse matrices and vectors compatible with the SciPy [@scipy] ecosystem.

# Purpose and prior art

There exist several open source frameworks – in Python or with a Python
interface – that implement the finite element method.  `scikit-fem` was
developed as a simple and lightweight alternative to the existing Python
packages with a focus on computational experimentation and custom PDE-based
model development.  We rely on pure interpreted Python code on top of the
NumPy-SciPy base which makes `scikit-fem` easy to install and portable across
multiple operating systems.  The reliance on NumPy arrays and SciPy sparse
matrices enables interoperability with various useful packages in the wider
Python ecosystem such as meshio [@meshio], pacopy, pyamg [@pyamg], and
scikit-sparse.

In contrast to FEniCS [@fenics], Firedrake [@firedrake], SfePy [@sfepy], GetFEM
[@getfem] and NGSolve [@ngsolve], `scikit-fem` incorporates no compiled code
making the installation quick and straightforward.  We specifically target
finite element assembly instead of encapsulating the entire finite element
analysis from pre- to postprocessing into a single framework.  As a consequence,
we will not provide an end-to-end experience when it comes to, e.g., specific
physical models or large scale distributed computing.

# Examples and enabled work

The source code distribution [@skfem] ships with over 30 examples that
demonstrate the library and its use.  Some examples are highlighted in
\autoref{fig:nitsche}.  Several publications already utilize computational
results from `scikit-fem`, e.g., @mcbain2018, @gustafsson2019, and
@gustafsson2020.  In addition, `scikit-fem` is a dependency in a recently
published Python package for battery modelling [@pybamm].

![A combination of quadratic triangular and biquadratic quadrilateral elements is used in solving the linear elastic contact problem using the Nitsche mortaring.\label{fig:nitsche}](ex_nitsche.png =200x)

# Acknowledgements

# References
