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

Example 13: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775065-226f6d00-c82e-11ea-950c-fe9a10901133.png


Example 14: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775119-3dda7800-c82e-11ea-8576-2219fcf31814.png

Example 15: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775166-52b70b80-c82e-11ea-9009-c9fa0a9e28e8.png

Example 16: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775206-65c9db80-c82e-11ea-8c49-bf191915602a.png

Example 17: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775309-8db93f00-c82e-11ea-9015-add2226ad01e.png

Example 18: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87775390-b04b5800-c82e-11ea-8999-e22305e909c1.png

Example 19: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87778846-7b420400-c834-11ea-8ff6-c439699b2802.gif

Example 20: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87778910-9745a580-c834-11ea-8277-62d58a7fe7b8.png

Example 21: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87779087-ebe92080-c834-11ea-9acc-d455b6124ad7.png

Example 22: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87779195-15a24780-c835-11ea-9a18-767092ae9467.png

Example 23: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87779278-38ccf700-c835-11ea-955a-b77a0336b791.png

Example 24: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87858848-92b6e500-c939-11ea-81f9-cc51f254d19e.png

Example 25: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87858907-f8a36c80-c939-11ea-87a2-7357d5f073b1.png

Example 26: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87858933-3902ea80-c93a-11ea-9d54-464235ab6325.png

Example 27: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87858972-97c86400-c93a-11ea-86e4-66f870b03e48.png

Example 28: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859005-c0505e00-c93a-11ea-9a78-72603edc242a.png

Example 29: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859022-e0801d00-c93a-11ea-978f-b1930627010b.png

Example 30: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859044-06a5bd00-c93b-11ea-84c2-9fbb9fc6e832.png

Example 31: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859068-32c13e00-c93b-11ea-984d-684e1e4c5066.png

Example 32: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859195-fcd08980-c93b-11ea-930e-ddcd26aabdb4.png
   :width: 500px

Example 33: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859239-47520600-c93c-11ea-8241-d62fdfd2a9a2.png
   :width: 500px

Example 34: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859267-749eb400-c93c-11ea-82cd-2d488fda39d4.png

Example 35: -
=============

.. figure:: https://user-images.githubusercontent.com/973268/87859275-85e7c080-c93c-11ea-9e62-3a9a8ee86070.png
