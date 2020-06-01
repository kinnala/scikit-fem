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
library for the creation, or *assembly*, of the finite element matrices $A$ and
vectors $b$.  The user loads a computational mesh, picks suitable basis
functions from the library's collection, and provides the PDE's weak
formulation.  This results in matrices and vectors compatible with the SciPy
ecosystem.

# Features

`scikit-fem` supports the bgb 

# Prior art

There exist several open source frameworks – written in Python or with a Python
interface – that implement the finite element method.  In contrast to Fenics,
Firedrake, SfePy, GetFEM++ and Netgen-NG, `scikit-fem` does not have any
compiled code making the installation quick and straightforward.  It
specifically targets the finite element assembly instead of being an end-to-end
finite element solver.


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

# References
