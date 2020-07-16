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

This example solves

.. figure:: https://user-images.githubusercontent.com/973268/87662192-80d31780-c76a-11ea-9291-2d11920bc098.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex07.py>`_

Example 8: Argyris basis functions
==================================

This example visualizes the :math:`C^1`-continuous fifth degree Argyris basis
functions on a simple triangular mesh.

.. figure:: https://user-images.githubusercontent.com/973268/87662432-e0c9be00-c76a-11ea-85b9-711c6b34791e.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex08.py>`_

Example 9: Three-dimensional Poisson equation
=============================================

.. note::
   This example will make use of the external packages `PyAMG <https://pypi.org/project/pyamg/>`_ or `pyamgcl <https://pypi.org/project/pyamgcl/>`_, if installed.

This example solves a three-dimensional Poisson equation using tetrahedral
elements and a preconditioned conjugate gradient method.
The figure was created using `Paraview <https://www.paraview.org/>`_.

.. figure:: https://user-images.githubusercontent.com/973268/87681574-7a06cd80-c787-11ea-8cfd-6ff5079e752c.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex09.py>`_

Example 10: Nonlinear minimal surface problem
=============================================

.. figure:: https://user-images.githubusercontent.com/973268/87663902-1c658780-c76d-11ea-9e00-324a18769ad2.png

`Source code <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex10.py>`_

Example 11: --
==============

Example 12: --
==============
