==================
 List of examples
==================

This page contains an overview of the examples included in the source code
distribution.

Example 1: Poisson equation with unit load
==========================================

This example solves the Poisson problem :math:`-\Delta u = 1` with the Dirichlet
boundary condition :math:`u = 0` in the unit square using piecewise linear
triangular elements.

.. figure:: https://user-images.githubusercontent.com/973268/87638021-c3d1c280-c74b-11ea-9859-dd82555747f5.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex01.py>`_
           
Example 2: Kirchhoff plate bending problem
==========================================

This example applies the nonconforming Morley element to the solution of the
biharmonic Kirchhoff plate bending problem :math:`D \Delta^2 u = f` in the unit
square with a constant loading :math:`f`, bending stiffness :math:`D` and a
combination of clamped, simply supported and free boundary conditions.

.. figure:: https://user-images.githubusercontent.com/973268/87659951-f50bbc00-c766-11ea-8c0e-7de0e9e83714.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex02.py>`_

Example 3: Linear elastic eigenvalue problem
============================================

This example solves the linear elastic eigenvalue problem
:math:`\mathrm{div}\,\sigma(u)= \lambda u` with
the displacement fixed on the left hand side boundary.
The following figure depicts the fifth eigenmode
of the cantilever beam.

.. figure:: https://user-images.githubusercontent.com/973268/87661134-cbec2b00-c768-11ea-81bc-f5455df7cc33.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex03.py>`_

Example 4: Linearized contact problem
=====================================

.. figure:: https://user-images.githubusercontent.com/973268/87661313-1372b700-c769-11ea-89ee-db144986a25a.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex04.py>`_

Example 5: Integral constraint
==============================

.. figure:: https://user-images.githubusercontent.com/973268/87661575-7ebc8900-c769-11ea-8e51-07aeb49797f9.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex05.py>`_

Example 6: High-order plotting
==============================

.. figure:: https://user-images.githubusercontent.com/973268/87661665-aa3f7380-c769-11ea-9677-8ec0ff27184c.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex06.py>`_

Example 7: Interior penalty method
==================================

This example solves the Poisson problem :math:`-\Delta u = 1` with :math:`u=0`
on the boundary using an interior penalty discontinuous Galerkin method.
The finite element basis is piecewise linear but discontinuous over
the element edges.

.. figure:: https://user-images.githubusercontent.com/973268/87662192-80d31780-c76a-11ea-9291-2d11920bc098.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex07.py>`_

Example 8: Argyris basis functions
==================================

This example visualizes the :math:`C^1`-continuous fifth degree Argyris basis
functions on a simple triangular mesh.
This element can be used in the conforming discretization of biharmonic problems.

.. figure:: https://user-images.githubusercontent.com/973268/87662432-e0c9be00-c76a-11ea-85b9-711c6b34791e.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex08.py>`_

Example 9: Three-dimensional Poisson equation
=============================================

This example solves :math:`-\Delta u = 1`
with :math:`u=0` on the boundary using tetrahedral elements and a preconditioned
conjugate gradient method.  The figure was created using `Paraview
<https://www.paraview.org/>`_.

.. note::

   This example will make use of the external packages `PyAMG <https://pypi.org/project/pyamg/>`_ or `pyamgcl <https://pypi.org/project/pyamgcl/>`_, if installed.

.. figure:: https://user-images.githubusercontent.com/973268/87681574-7a06cd80-c787-11ea-8cfd-6ff5079e752c.png
   :width: 500px

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex09.py>`_

Example 10: Nonlinear minimal surface problem
=============================================

This example solves the nonlinear minimal surface problem :math:`\nabla \cdot
\left(\frac{1}{\sqrt{1 + \|u\|^2}} \nabla u \right)= 0` with :math:`u=g`
prescribed on the boundary of the square domain.  The nonlinear problem is
linearized using the Newton's method with an analytical Jacobian calculated by
hand.

.. figure:: https://user-images.githubusercontent.com/973268/87663902-1c658780-c76d-11ea-9e00-324a18769ad2.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex10.py>`_

Example 11: Three-dimensional linear elasticity
===============================================

This example solves the three-dimensional linear elasticity equations
:math:`\mathrm{div}\,\sigma(u)=0` using trilinear hexahedral elements.
Dirichlet conditions are set on the opposing faces of a cube: one face remains
fixed and the other is displaced slightly outwards.
The figure was created using `Paraview <https://www.paraview.org/>`_.

.. figure:: https://user-images.githubusercontent.com/973268/87685532-31054800-c78c-11ea-9b89-bc41dc0cb80c.png
   :width: 500px

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex11.py>`_

Example 12: Mesh generation and postprocessing
==============================================

This example demonstrates mesh generation using an external package and
postprocessing the value of a functional, Boussinesq k-factor.

.. note::

   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_.

.. figure:: https://user-images.githubusercontent.com/973268/87686059-bee13300-c78c-11ea-9693-727f0baf0433.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex12.py>`_
