=====================
 Gallery of examples
=====================

This page contains an overview of the examples contained in the `source code
repository <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/>`_.

Example 1: Poisson equation with unit load
==========================================

This example solves the Poisson problem :math:`-\Delta u = 1` with the Dirichlet
boundary condition :math:`u = 0` in the unit square using piecewise linear
triangular elements.

.. figure:: https://user-images.githubusercontent.com/973268/87638021-c3d1c280-c74b-11ea-9859-dd82555747f5.png

   The solution of Example 1.

See the `source code of Example 1 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex01.py>`_ for more information.
           
Example 2: Kirchhoff plate bending problem
==========================================

This example solves the biharmonic Kirchhoff plate bending problem :math:`D
\Delta^2 u = f` in the unit square with a constant loading :math:`f`, bending
stiffness :math:`D` and a combination of clamped, simply supported and free
boundary conditions.

.. figure:: https://user-images.githubusercontent.com/973268/87659951-f50bbc00-c766-11ea-8c0e-7de0e9e83714.png

   The solution of Example 2.

See the `source code of Example 2 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex02.py>`_ for more information.

Example 3: Linear elastic eigenvalue problem
============================================

This example solves the linear elastic eigenvalue problem
:math:`\mathrm{div}\,\sigma(u)= \lambda u` with
the displacement fixed on the left hand side boundary.

.. figure:: https://user-images.githubusercontent.com/973268/87661134-cbec2b00-c768-11ea-81bc-f5455df7cc33.png

   The fifth eigenmode of Example 3.

See the `source code of Example 3 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex03.py>`_ for more information.

Example 4: Linearized contact problem
=====================================

This example solves a single interation of the contact problem
between two elastic bodies using the Nitsche's method.
Triangular and quadrilateral second-order elements are used
in the discretization of the two elastic bodies.

.. note::

   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87661313-1372b700-c769-11ea-89ee-db144986a25a.png

   The displaced meshes and the von Mises stress of Example 4.

See the `source code of Example 4 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex04.py>`_ for more information.

Example 7: Interior penalty method
==================================

This example solves the Poisson problem :math:`-\Delta u = 1` with :math:`u=0`
on the boundary using interior penalty discontinuous Galerkin method.
The finite element basis is piecewise linear but discontinuous over
the element edges.

.. figure:: https://user-images.githubusercontent.com/973268/87662192-80d31780-c76a-11ea-9291-2d11920bc098.png

   The solution of Example 7.

See the `source code of Example 7 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex07.py>`_ for more information.

Example 8: Argyris basis functions
==================================

This example visualizes the :math:`C^1`-continuous fifth degree Argyris basis
functions on a simple triangular mesh.
This element can be used in the conforming discretization of biharmonic problems.

.. figure:: https://user-images.githubusercontent.com/973268/87662432-e0c9be00-c76a-11ea-85b9-711c6b34791e.png

   The Argyris basis functions of Example 8 corresponding to the middle node and
   the edges connected to it.

See the `source code of Example 8 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex08.py>`_ for more information.

Example 9: Three-dimensional Poisson equation
=============================================

This example solves :math:`-\Delta u = 1`
with :math:`u=0` on the boundary using tetrahedral elements and a preconditioned
conjugate gradient method.

.. note::

   This example will make use of the external packages `PyAMG
   <https://pypi.org/project/pyamg/>`__ or `pyamgcl
   <https://pypi.org/project/pyamgcl/>`__, if installed.

.. figure:: https://user-images.githubusercontent.com/973268/87681574-7a06cd80-c787-11ea-8cfd-6ff5079e752c.png

   The solution of Example 9 on a cross-section of the tetrahedral mesh.  The
   figure was created using `ParaView <https://www.paraview.org/>`__.

See the `source code of Example 9 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex09.py>`_ for more information.

Example 10: Nonlinear minimal surface problem
=============================================

This example solves the nonlinear minimal surface problem :math:`\nabla \cdot
\left(\frac{1}{\sqrt{1 + \|u\|^2}} \nabla u \right)= 0` with :math:`u=g`
prescribed on the boundary of the square domain.  The nonlinear problem is
linearized using the Newton's method with an analytical Jacobian calculated by
hand.

.. figure:: https://user-images.githubusercontent.com/973268/87663902-1c658780-c76d-11ea-9e00-324a18769ad2.png

   The solution of Example 10.

See the `source code of Example 10 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex10.py>`_ for more information.

Example 11: Three-dimensional linear elasticity
===============================================

This example solves the three-dimensional linear elasticity equations
:math:`\mathrm{div}\,\sigma(u)=0` using trilinear hexahedral elements.
Dirichlet conditions are set on the opposing faces of a cube: one face remains
fixed and the other is displaced slightly outwards.

.. figure:: https://user-images.githubusercontent.com/973268/87685532-31054800-c78c-11ea-9b89-bc41dc0cb80c.png

   The displaced mesh of Example 11.  The figure was created using `ParaView
   <https://www.paraview.org/>`__.

See the `source code of Example 11 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex11.py>`_ for more information.

Example 12: Mesh generation and postprocessing
==============================================

This example demonstrates mesh generation using an external package and
postprocessing the value of a functional, Boussinesq k-factor.

.. figure:: https://user-images.githubusercontent.com/973268/87686059-bee13300-c78c-11ea-9693-727f0baf0433.png

   The solution of Example 12.

See the `source code of Example 12 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex12.py>`_ for more information.

Example 13: Laplace with mixed boundary conditions
==================================================

This example solves :math:`\Delta u = 0` in
:math:`\Omega=\{(x,y):1<x^2+y^2<4,~0<\theta<\pi/2\}`, where :math:`\tan \theta =
y/x`, with :math:`u = 0` on :math:`y = 0`, :math:`u = 1` on :math:`x =
0`, and :math:`\frac{\partial u}{\partial n} = 0` on the rest of the
boundary.

.. note::

   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87775065-226f6d00-c82e-11ea-950c-fe9a10901133.png

   The solution of Example 13.

See the `source code of Example 13 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex13.py>`_ for more information.

.. _ex14:

Example 14: Laplace with inhomogeneous boundary conditions
==========================================================

This example demonstrates how to impose coordinate-dependent Dirichlet
conditions for the Laplace equation :math:`\Delta u = 0`. The solution will
satisfy :math:`u=x^2 - y^2` on the boundary of the square domain.

.. figure:: https://user-images.githubusercontent.com/973268/87775119-3dda7800-c82e-11ea-8576-2219fcf31814.png

   The solution of Example 14.

See the `source code of Example 14 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex14.py>`_ for more information.

Example 15: One-dimensional Poisson equation
============================================

This example solves :math:`-u'' = 1` in :math:`(0,1)` with the boundary
condition :math:`u(0)=u(1)=0`.

.. figure:: https://user-images.githubusercontent.com/973268/87775166-52b70b80-c82e-11ea-9009-c9fa0a9e28e8.png

   The solution of Example 15.

See the `source code of Example 15 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex15.py>`_ for more information.

Example 16: Legendre's equation
===============================

This example solves the eigenvalue problem :math:`((1 - x^2) u')' + k u = 0` in
:math:`(-1,1)`.

.. figure:: https://user-images.githubusercontent.com/973268/87775206-65c9db80-c82e-11ea-8c49-bf191915602a.png

   The six first eigenmodes of Example 16.

See the `source code of Example 16 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex16.py>`_ for more information.

Example 17: Insulated wire
==========================

This example solves the steady heat conduction
with generation in an insulated wire. In radial
coordinates, the governing equations read: find :math:`T`
satisfying :math:`\nabla \cdot (k_0 \nabla T) + A = 0,~0<r<a`,
and
:math:`\nabla \cdot (k_1 \nabla T) = 0,~a<r<b`,
with the boundary condition
:math:`k_1 \frac{\partial T}{\partial r} + h T = 0` on :math:`r=b`.

.. figure:: https://user-images.githubusercontent.com/973268/87775309-8db93f00-c82e-11ea-9015-add2226ad01e.png

   The solution of Example 17.

See the `source code of Example 17 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex17.py>`_ for more information.

Example 18: Stokes equations
============================

This example solves for the creeping flow problem in the primitive variables,
i.e. velocity and pressure instead of the stream-function.  These are governed
by the Stokes momentum :math:`- \nu\Delta\boldsymbol{u} + \rho^{-1}\nabla p = \boldsymbol{f}` and the continuity equation :math:`\nabla\cdot\boldsymbol{u} = 0`.

.. figure:: https://user-images.githubusercontent.com/973268/87775390-b04b5800-c82e-11ea-8999-e22305e909c1.png

   The streamlines of Example 18.

See the `source code of Example 18 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex18.py>`_ for more information.

Example 19: Heat equation
=========================

This example solves the heat equation :math:`\frac{\partial T}{\partial t} = \kappa\Delta T` in the domain :math:`|x|<w_0` and :math:`|y|<w_1` with the initial value :math:`T_0(x,y) = \cos\frac{\pi x}{2w_0}\cos\frac{\pi y}{2w_1}` using the generalized trapezoidal
rule ("theta method") and fast time-stepping by factorizing the evolution matrix once and for all.

.. figure:: https://user-images.githubusercontent.com/973268/87778846-7b420400-c834-11ea-8ff6-c439699b2802.gif

   The solution of Example 19.

See the `source code of Example 19 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex19.py>`_ for more information.

Example 20: Creeping flow via stream-function
=============================================

This example solves the creeping flow problem via the stream-function
formulation.
The stream-function :math:`\psi` for two-dimensional creeping flow is
governed by the biharmonic equation :math:`\nu \Delta^2\psi = \mathrm{rot}\,\boldsymbol{f}` where :math:`\nu` is the kinematic viscosity (assumed constant),
:math:`\boldsymbol{f}` the volumetric body-force, and :math:`\mathrm{rot}\,\boldsymbol{f} =
\partial f_y/\partial x - \partial f_x/\partial y`.  The boundary
conditions at a wall are that :math:`\psi` is constant (the wall is
impermeable) and that the normal component of its gradient vanishes (no
slip)

.. figure:: https://user-images.githubusercontent.com/973268/87778910-9745a580-c834-11ea-8277-62d58a7fe7b8.png

   The velocity field of Example 20.

See the `source code of Example 20 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex20.py>`_ for more information.

Example 21: Structural vibration
================================

This example demonstrates the solution of a three-dimensional vector-valued
eigenvalue problem by considering the vibration of an elastic structure.

.. figure:: https://user-images.githubusercontent.com/973268/87779087-ebe92080-c834-11ea-9acc-d455b6124ad7.png

   An eigenmode of Example 21.

See the `source code of Example 21 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex21.py>`_ for more information.

Example 22: Adaptive Poisson equation
=====================================

This example solves Example 1 adaptively in an L-shaped domain.
Using linear elements, the error indicators read :math:`\eta_K^2 = h_K^2 \|f\|_{0,K}^2` and :math:`\eta_E^2 = h_E \| [[\nabla u_h \cdot n ]] \|_{0,E}^2`   
for each element :math:`K` and
edge :math:`E`.

.. figure:: https://user-images.githubusercontent.com/973268/87779195-15a24780-c835-11ea-9a18-767092ae9467.png

   The adaptively refined mesh of Example 22.

See the `source code of Example 22 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex22.py>`_ for more information.

Example 23: Bratu-Gelfand
=========================

This example solves the Bratu-Gelfand two-point boundary value problem :math:`u'' + \lambda \mathrm e^u = 0`, :math:`0 < x < 1`,
with :math:`u(0)=u(1)=0` and where :math:`\lambda > 0` is a parameter.

.. note::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87779278-38ccf700-c835-11ea-955a-b77a0336b791.png

   The results of Example 23.

See the `source code of Example 23 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex23.py>`_ for more information.

Example 24: Stokes flow with inhomogeneous boundary conditions
==============================================================

This example solves the Stokes flow over a backward-facing step
with a parabolic velocity profile at the inlet.

.. figure:: https://user-images.githubusercontent.com/973268/87858848-92b6e500-c939-11ea-81f9-cc51f254d19e.png

   The streamlines of Example 24.

See the `source code of Example 24 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex24.py>`_ for more information.

Example 25: Forced convection
=============================

This example solves the plane Graetz problem with the governing
advection-diffusion equation :math:`\mathrm{Pe} \;u\frac{\partial T}{\partial x}
= \nabla^2 T` where the velocity profile is :math:`u (y) = 6 y (1 - y)` and the
Péclet number :math:`\mathrm{Pe}` is the mean velocity times the width divided
by the thermal diffusivity.

.. figure:: https://user-images.githubusercontent.com/973268/87858907-f8a36c80-c939-11ea-87a2-7357d5f073b1.png

   The solution of Example 25.

See the `source code of Example 25 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex25.py>`_ for more information.

Example 26: Restricting problem to a subdomain
==============================================

This example extends Example 17 by restricting the solution to a subdomain.

.. figure:: https://user-images.githubusercontent.com/973268/87858933-3902ea80-c93a-11ea-9d54-464235ab6325.png

   The solution of Example 26.

See the `source code of Example 26 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex26.py>`_ for more information.

Example 27: Backward-facing step
================================

This example uses `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__ to extend
the Stokes equations over a backward-facing step (Example 24) to finite Reynolds
number; this means defining a residual for the nonlinear problem and its
derivatives with respect to the solution and to the Reynolds number.

.. note::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87858972-97c86400-c93a-11ea-86e4-66f870b03e48.png

   The streamlines of Example 27 for :math:`\mathrm{Re}=750`.

See the `source code of Example 27 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex27.py>`_ for more information.

Example 28: Conjugate heat transfer
===================================

This example extends Example 25 to conjugate heat transfer by giving a finite
thickness and thermal conductivity to one of the walls.  The example is modified
to a configuration for which there exists a fully developed solution which can be
found in closed form: given a uniform heat flux over each of the walls, the
temperature field asymptotically is the superposition of a uniform longitudinal
gradient and a transverse profile.

.. note::
   This example requires the external package
   `pygmsh <https://pypi.org/project/pygmsh/>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87859005-c0505e00-c93a-11ea-9a78-72603edc242a.png

   The solution of Example 28.

See the `source code of Example 28 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex28.py>`_ for more information.

Example 29: Linear hydrodynamic stability
=========================================

The linear stability of one-dimensional solutions of the Navier-Stokes equations
is governed by the `Orr-Sommerfeld equation <https://en.wikipedia.org/wiki/Orr%E2%80%93Sommerfeld_equation>`_.  This is expressed in terms of the stream-function
:math:`\phi` of the perturbation, giving a two-point boundary value problem      
:math:`\alpha\phi(\pm 1) = \phi'(\pm 1) = 0`
for a complex fourth-order ordinary differential equation,

.. math::
   \left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)^2\phi
   = (\mathrm j\alpha R)\left\{
     (c - U)\left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)\phi
     - U''\phi,
   \right\}
   
where :math:`U(z)` is the base velocity profile, :math:`c` and :math:`\alpha`
are the wavespeed and wavenumber of the disturbance, and :math:`R` is the
Reynolds number.

.. figure:: https://user-images.githubusercontent.com/973268/87859022-e0801d00-c93a-11ea-978f-b1930627010b.png

   The results of Example 29.

See the `source code of Example 29 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex29.py>`_ for more information.

Example 30: Krylov-Uzawa method for the Stokes equation
=======================================================

This example solves the Stokes equation iteratively in a square domain.

.. figure:: https://user-images.githubusercontent.com/973268/87859044-06a5bd00-c93b-11ea-84c2-9fbb9fc6e832.png

   The pressure field of Example 30.

See the `source code of Example 30 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex30.py>`_ for more information.

Example 31: Curved elements
===========================

This example solves the eigenvalue problem :math:`-\Delta u = \lambda u`
with the boundary condition :math:`u|_{\partial \Omega} = 0` using isoparametric
mapping via biquadratic basis and finite element approximation using fifth-order
quadrilaterals.

.. figure:: https://user-images.githubusercontent.com/973268/87859068-32c13e00-c93b-11ea-984d-684e1e4c5066.png

   An eigenmode of Example 31 in a curved mesh.

See the `source code of Example 31 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex31.py>`_ for more information.

Example 32: Block diagonally preconditioned Stokes solver
=========================================================

This example solves the Stokes problem in three dimensions, with an
algorithm that scales to reasonably fine meshes (a million tetrahedra in a few
minutes).

.. note::
   This examples requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`__ and an implementation of algebraic multigrid (either `pyamgcl    <https://pypi.org/project/pyamgcl>`_ or `pyamg <https://pypi.org/project/pyamg/>`_).

.. figure:: https://user-images.githubusercontent.com/973268/87859195-fcd08980-c93b-11ea-930e-ddcd26aabdb4.png

   The pressure field of Example 32.
   The figure was created using `ParaView <https://www.paraview.org/>`_.

See the `source code of Example 32 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex32.py>`_ for more information.

Example 33: H(curl) conforming model problem
============================================


This example solves the vector-valued problem :math:`\nabla \times \nabla \times
E + E = f` in domain :math:`\Omega = [-1, 1]^3` with the boundary condition
:math:`E \times n|_{\partial \Omega} = 0` using the lowest order Nédélec edge
element.

.. figure:: https://user-images.githubusercontent.com/973268/87859239-47520600-c93c-11ea-8241-d62fdfd2a9a2.png

   The solution of Example 33 with the colors given by the magnitude
   of the vector field.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

See the `source code of Example 33 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex33.py>`_ for more information.

Example 34: Euler-Bernoulli beam
================================


This example solves the Euler-Bernoulli beam equation
:math:`(EI u'')'' = 1`
with the boundary conditions
:math:`u(0)=u'(0) = 0` and using cubic Hermite elements.
The exact solution at :math:`x=1` is :math:`u(1)=1/8`.

.. figure:: https://user-images.githubusercontent.com/973268/87859267-749eb400-c93c-11ea-82cd-2d488fda39d4.png

   The solution of Example 34.

See the `source code of Example 34 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex34.py>`_ for more information.

Example 35: Characteristic impedance and velocity factor
========================================================

This example solves the series inductance (per meter) and parallel capacitance
(per meter) of RG316 coaxial cable. These values are then used to compute the
characteristic impedance and velocity factor of the cable.

.. note::
   This example requires the external package
   `pygmsh <https://pypi.org/project/pygmsh/>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87859275-85e7c080-c93c-11ea-9e62-3a9a8ee86070.png

   The results of Example 35.

See the `source code of Example 35 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex35.py>`_ for more information.
