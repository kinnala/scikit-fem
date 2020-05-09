---
title: 'scikit-fem: A Python package for assembling finite element matrices'
tags:
  - Python
  - numerics
  - finite element method
authors:
  - name: Tom Gustafsson
    orcid: XXXX-YYYY-ZZZZ-WWWW
    affiliation: 1
  - name: Geordie McBain
    orcid: XXXX-YYYY-ZZZZ-WWWW
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

Various models in modern physics and engineering - such as Navier-Stokes
equations in fluid mechanics, Maxwell's equations in electromagnetism, and
Schrödinger equation in quantum mechanics - are based on partial differential
equations (PDE's).  Finite element method (FEM) is a flexible computational
technique for the discretisation and solution of PDE's, especially in the case
of complex spatial domains.

Conceptually FEM transforms PDE's into systems of linear equations that can be
solved using linear solvers. ``scikit-fem`` is a lightweight Python library for
the creation, or /assembly/, of finite element matrices.

Several open source libraries implement the finite element method; cf. Fenics,
Firedrake, SfePy, GetFEM++, and nutils.  A majority of the libraries target
solving large scale problems with millions of unknowns (or more) or are
specialised to a set of predefined PDE's. As a consequence, accessing the
underlying finite element matrices may require additional work and many
implementational details are abstracted from the user.

``scikit-fem`` is a lightweight Python library for finite element matrix
assembly.  The philosophy is to provide a 

In other words, the user inputs a mesh of the computational domain together with
a weak formulation of the differential operator, chooses basis functions from a
list of supported elements, and receives the corresponding finite element
matrix.  This philosophy and workflow allows rapid prototyping of customised
FEM-based numerical solvers.

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
