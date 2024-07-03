=====================
 Gallery of examples
=====================

This page contains an overview of the examples contained in the source code
repository.

Poisson equation
================

Example 1: Poisson equation with unit load
------------------------------------------

This example solves the Poisson problem :math:`-\Delta u = 1` with the Dirichlet
boundary condition :math:`u = 0` in the unit square using piecewise-linear
triangular elements.

.. plot::
   :caption: The solution of Example 1.

   from docs.examples.ex01 import visualize
   visualize()

See the source code of :exlink:`01` for more information.

.. _ex07:

Example 7: Discontinuous Galerkin method
----------------------------------------

This example solves the Poisson problem :math:`-\Delta u = 1` with :math:`u=0`
on the boundary using discontinuous Galerkin method.  The finite element basis
is piecewise-quartic but discontinuous over the element edges.

.. plot::
   :caption: The solution of Example 7.

   from docs.examples.ex07 import visualize
   visualize()

See the source code of :exlink:`07` for more information.

Example 12: Postprocessing
--------------------------

This example demonstrates postprocessing the value of a functional, Boussinesq's k-factor.

.. plot::
   :caption: The solution of Example 12.

   from docs.examples.ex12 import visualize
   visualize()

See the source code of :exlink:`12` for more information.

Example 13: Laplace with mixed boundary conditions
--------------------------------------------------

This example solves :math:`\Delta u = 0` in
:math:`\Omega=\{(x,y):1<x^2+y^2<4,~0<\theta<\pi/2\}`, where :math:`\tan \theta =
y/x`, with :math:`u = 0` on :math:`y = 0`, :math:`u = 1` on :math:`x =
0`, and :math:`\frac{\partial u}{\partial n} = 0` on the rest of the
boundary.

.. plot::
   :caption: The solution of Example 13.

   from docs.examples.ex13 import visualize
   visualize()

See the source code of :exlink:`13` for more information.

.. _ex14:

Example 14: Laplace with inhomogeneous boundary conditions
----------------------------------------------------------

This example demonstrates how to impose coordinate-dependent Dirichlet
conditions for the Laplace equation :math:`\Delta u = 0`. The solution will
satisfy :math:`u=x^2 - y^2` on the boundary of the square domain.

.. plot::
   :caption: The solution of Example 14.

   from docs.examples.ex14 import visualize
   visualize()

See the source code of :exlink:`14` for more information.

Example 15: One-dimensional Poisson equation
--------------------------------------------

This example solves :math:`-u'' = 1` in :math:`(0,1)` with the boundary
condition :math:`u(0)=u(1)=0`.

.. figure:: https://user-images.githubusercontent.com/973268/87775166-52b70b80-c82e-11ea-9009-c9fa0a9e28e8.png

   The solution of Example 15.

See the source code of :exlink:`15` for more information.


Example 9: Three-dimensional Poisson equation
---------------------------------------------

This example solves :math:`-\Delta u = 1` with :math:`u=0` on the boundary
using linear tetrahedral elements and a preconditioned conjugate gradient
method.

.. note::

   This example will make use of the external packages `PyAMG
   <https://pypi.org/project/pyamg/>`__ or `pyamgcl
   <https://pypi.org/project/pyamgcl/>`__, if installed.

.. figure:: https://user-images.githubusercontent.com/973268/93183072-33abfb80-f743-11ea-9076-1324cbf28531.png

   The solution of Example 9 on a cross-section of the tetrahedral mesh.  The
   figure was created using `ParaView <https://www.paraview.org/>`__.

See the source code of :exlink:`09` for more information.

Example 22: Adaptive Poisson equation
-------------------------------------

This example solves Example 1 adaptively in an L-shaped domain.
Using linear elements, the error indicators read :math:`\eta_K^2 = h_K^2 \|f\|_{0,K}^2` and :math:`\eta_E^2 = h_E \| [[\nabla u_h \cdot n ]] \|_{0,E}^2`   
for each element :math:`K` and
edge :math:`E`.

.. plot::
   :caption: The final solution of Example 22.

   from docs.examples.ex22 import visualize
   visualize()

See the source code of :exlink:`22` for more information.

Example 37: Mixed Poisson equation
----------------------------------

This example solves the mixed formulation of the Poisson equation
using the lowest order Raviart-Thomas elements.

.. figure:: https://user-images.githubusercontent.com/973268/182335108-7f29b086-f9b2-42fb-ae9f-dd707d3808f8.png

   The piecewise constant solution field.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

See the source code of :exlink:`37` for more information.

Example 38: Point source
------------------------

Point sources require different assembly to other linear forms.

This example computes the Green's function for a disk; i.e. the solution of the
Dirichlet problem for the Poisson equation with the source term concentrated at
a single interior point :math:`\boldsymbol{s}`, :math:`-\Delta u = \delta
(\boldsymbol{x} - \boldsymbol{s})`.

.. plot::
   :caption: The scalar potential in the disk with point source at (0.3, 0.2).

   from docs.examples.ex38 import visualize
   visualize()

See the source code of :exlink:`38`
for more information.

Example 40: Hybridizable discontinuous Galerkin method
------------------------------------------------------

This examples solves the Poisson equation with unit load using a technique
where the finite element basis is first discontinous across element edges and
then the continuity is recovered with the help of Lagrange multipliers defined
on the mesh skeleton (i.e. a "skeleton mesh" consisting only of the edges of
the original mesh).

.. plot::
   :caption: The solution of Example 40 on the skeleton mesh.

   from docs.examples.ex40 import visualize
   visualize()

See the source code of :exlink:`40`
for more information.

Example 41: Mixed meshes
------------------------

This example solves the Poisson equation with unit load on a mesh consisting
of both triangles and quadrilaterals.  The support for mixed meshes is
preliminary and works only for elements with nodal or internal
degrees-of-freedom (sharing face and edge DOFs between mesh types is
work-in-progress).

.. plot::
   :caption: The solution of Example 41 on the mesh with both
             triangles and quadrilaterals.

   from docs.examples.ex41 import visualize
   visualize()

See the source code of :exlink:`41`
for more information.

Solid mechanics
===============

Example 2: Kirchhoff plate bending problem
------------------------------------------

This example solves the biharmonic Kirchhoff plate bending problem :math:`D
\Delta^2 u = f` in the unit square with a constant loading :math:`f`, bending
stiffness :math:`D` and a combination of clamped, simply supported and free
boundary conditions.

.. plot::
   :caption: The solution of Example 2.

   from docs.examples.ex02 import visualize
   visualize()

See the source code of :exlink:`02` for more information.

Example 3: Linear elastic eigenvalue problem
--------------------------------------------

This example solves the linear elastic eigenvalue problem
:math:`\mathrm{div}\,\sigma(u)= \lambda u` with
the displacement fixed on the left boundary.

.. plot::
   :caption: The fifth eigenmode of Example 3.

   from docs.examples.ex03 import visualize
   visualize()

See the source code of :exlink:`03` for more information.

Example 4: Linearized contact problem
-------------------------------------

This example solves a single interation of the contact problem
between two elastic bodies using the Nitsche's method.
Triangular and quadrilateral second-order elements are used
in the discretization of the two elastic bodies.

.. plot::
   :caption: The displaced meshes and the von Mises stress of Example 4.

   from docs.examples.ex04 import visualize
   visualize()

See the source code of :exlink:`04` for more information.


Example 8: Argyris basis functions
----------------------------------

This example visualizes the :math:`C^1`-continuous fifth degree Argyris basis
functions on a simple triangular mesh.
This element can be used in the conforming discretization of biharmonic problems.

.. plot::
   :caption: The Argyris basis functions of Example 8 corresponding to the
             middle node and the edges connected to it.

   from docs.examples.ex08 import visualize
   visualize()

See the source code of :exlink:`08` for more information.

Example 11: Three-dimensional linear elasticity
-----------------------------------------------

This example solves the three-dimensional linear elasticity equations
:math:`\mathrm{div}\,\sigma(u)=0` using trilinear hexahedral elements.
Dirichlet conditions are set on the opposing faces of a cube: one face remains
fixed and the other is displaced slightly outwards.

.. figure:: https://user-images.githubusercontent.com/973268/87685532-31054800-c78c-11ea-9b89-bc41dc0cb80c.png

   The displaced mesh of Example 11.  The figure was created using `ParaView
   <https://www.paraview.org/>`__.

See the source code of :exlink:`11` for more information.

Example 21: Structural vibration
--------------------------------

This example demonstrates the solution of a three-dimensional vector-valued
eigenvalue problem by considering the vibration of an elastic structure.

.. figure:: https://user-images.githubusercontent.com/973268/147790554-4b768d43-25fa-49cd-ab19-b16a199a6459.png

   The first eigenmode of Example 21.

See the source code of :exlink:`21` for more information.

Example 34: Euler-Bernoulli beam
--------------------------------

This example solves the Euler-Bernoulli beam equation
:math:`(EI u'')'' = 1`
with the boundary conditions
:math:`u(0)=u'(0) = 0` and using cubic Hermite elements.
The exact solution at :math:`x=1` is :math:`u(1)=1/8`.

.. figure:: https://user-images.githubusercontent.com/973268/87859267-749eb400-c93c-11ea-82cd-2d488fda39d4.png

   The solution of Example 34.

See the source code of :exlink:`34` for more information.

Example 36: Nearly incompressible hyperelasticity
-------------------------------------------------

This example demonstrates the implementation of a two field mixed formulation
for nearly incompressible Neo-Hookean solids.

.. figure:: https://user-images.githubusercontent.com/22624037/91212007-4055aa80-e6d5-11ea-8572-f27986887331.png

   The displacement contour of Example 36.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

See the source code of :exlink:`36` for more information.


Example 43: Hyperelasticity
---------------------------

This example demonstrates Newton's method applied to the classical formulation
of a hyperelastic Neo-Hookean solid.

.. figure:: https://user-images.githubusercontent.com/973268/147790182-64f4abf4-3909-4ec0-89ac-2add304b133d.png

   The deformed mesh of Example 43.
   The figure was created using `vedo <https://github.com/marcomusy/vedo>`__.

See the source code of :exlink:`43` for more information.

Fluid mechanics
===============

Example 18: Stokes equations
----------------------------

This example solves for the creeping flow problem in the primitive variables,
i.e. velocity and pressure instead of the stream-function.  These are governed
by the Stokes momentum :math:`- \nu\Delta\boldsymbol{u} + \rho^{-1}\nabla p = \boldsymbol{f}` and the continuity equation :math:`\nabla\cdot\boldsymbol{u} = 0`.

.. figure:: https://user-images.githubusercontent.com/1588947/93292002-d6d64100-f827-11ea-9a0a-c64d5d2979b7.png

   The streamlines of Example 18.

See the source code of :exlink:`18` for more information.

Example 20: Creeping flow via stream-function
---------------------------------------------

This example solves the creeping flow problem via the stream-function
formulation.
The stream-function :math:`\psi` for two-dimensional creeping flow is
governed by the biharmonic equation :math:`\nu \Delta^2\psi = \mathrm{rot}\,\boldsymbol{f}` where :math:`\nu` is the kinematic viscosity (assumed constant),
:math:`\boldsymbol{f}` the volumetric body-force, and :math:`\mathrm{rot}\,\boldsymbol{f} =
\partial f_y/\partial x - \partial f_x/\partial y`.  The boundary
conditions at a wall are that :math:`\psi` is constant (the wall is
impermeable) and that the normal component of its gradient vanishes (no
slip)

.. figure:: https://user-images.githubusercontent.com/1588947/93291998-d50c7d80-f827-11ea-861b-f24ed27072d0.png

   The velocity field of Example 20.

See the source code of :exlink:`20` for more information.

Example 24: Stokes flow with inhomogeneous boundary conditions
--------------------------------------------------------------

This example solves the Stokes flow over a backward-facing step
with a parabolic velocity profile at the inlet.

.. figure:: https://user-images.githubusercontent.com/973268/87858848-92b6e500-c939-11ea-81f9-cc51f254d19e.png

   The streamlines of Example 24.

See the source code of :exlink:`24` for more information.

Example 29: Linear hydrodynamic stability
-----------------------------------------

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

See the source code of :exlink:`29` for more information.

Example 30: Krylov-Uzawa method for the Stokes equation
-------------------------------------------------------

This example solves the Stokes equation iteratively in a square domain.

.. figure:: https://user-images.githubusercontent.com/973268/87859044-06a5bd00-c93b-11ea-84c2-9fbb9fc6e832.png

   The pressure field of Example 30.

See the source code of :exlink:`30` for more information.

Example 32: Block diagonally preconditioned Stokes solver
---------------------------------------------------------

This example solves the Stokes problem in three dimensions, with an
algorithm that scales to reasonably fine meshes (a million tetrahedra in a few
minutes).

.. note::
   This examples requires an implementation of algebraic multigrid (either `pyamgcl <https://pypi.org/project/pyamgcl>`_ or `pyamg <https://pypi.org/project/pyamg/>`_).

.. figure:: https://user-images.githubusercontent.com/1588947/96520786-8a18d680-12bb-11eb-981a-c3388f2c8e35.png

   The velocity and pressure fields of Example 32, clipped in the plane of spanwise symmetry, *z* = 0.
   The figure was created using `ParaView <https://www.paraview.org/>`_ 5.8.1.

See the source code of :exlink:`32` for more information.

Example 42: Periodic meshes
---------------------------

This example solves the advection equation on a periodic square mesh.

.. figure:: https://user-images.githubusercontent.com/973268/133767233-a5d78ec4-ffe7-4d49-bc93-9d9a0faae5a1.png

   The solution of Example 42 on a periodic mesh.

See the source code of :exlink:`42`
for more information.

Example 50: Advection-diffusion in non-uniform flow
---------------------------------------------------

This example solves the advection-diffusion problem
for the temperature distribution around a cold sphere
in a warm liquid. The liquid flow is modeled using
the Stokes flow field. The thermal diffusivity to
advection ratio is controlled by the Peclet number.
The problem is solved in a cylindrical coordinate
system (hence a different definition of the Laplacian).

.. figure:: https://github.com/turczyneq/fem_advection_diffusion/assets/51670923/271b5b86-71f1-49fb-b32e-3c9d012a29ee

   Temperature distribution of Example 50. Liquid behind the sphere is colder than the incoming one.

See the source code of :exlink:`50` for more information.


Heat transfer
=============

Example 17: Insulated wire
--------------------------

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

See the source code of :exlink:`17` for more information.

Example 19: Heat equation
-------------------------

This example solves the heat equation :math:`\frac{\partial T}{\partial t} = \kappa\Delta T` in the domain :math:`|x|<w_0` and :math:`|y|<w_1` with the initial value :math:`T_0(x,y) = \cos\frac{\pi x}{2w_0}\cos\frac{\pi y}{2w_1}` using the generalized trapezoidal
rule ("theta method") and fast time-stepping by factorizing the evolution matrix once and for all.

.. figure:: https://user-images.githubusercontent.com/973268/152611674-984c18fb-2e81-4253-9abb-b4a95289dee5.gif

   The solution of Example 19.

See the source code of :exlink:`19` for more information.

Example 25: Forced convection
-----------------------------

This example solves the plane Graetz problem with the governing
advection-diffusion equation :math:`\mathrm{Pe} \;u\frac{\partial T}{\partial x}
= \nabla^2 T` where the velocity profile is :math:`u (y) = 6 y (1 - y)` and the
Péclet number :math:`\mathrm{Pe}` is the mean velocity times the width divided
by the thermal diffusivity.

.. figure:: https://user-images.githubusercontent.com/973268/87858907-f8a36c80-c939-11ea-87a2-7357d5f073b1.png

   The solution of Example 25.

See the source code of :exlink:`25` for more information.

Example 26: Restricting problem to a subdomain
----------------------------------------------

This example extends Example 17 by restricting the solution to a subdomain.

.. figure:: https://user-images.githubusercontent.com/973268/87858933-3902ea80-c93a-11ea-9d54-464235ab6325.png

   The solution of Example 26.

See the source code of :exlink:`26` for more information.

Example 28: Conjugate heat transfer
-----------------------------------

This example extends Example 25 to conjugate heat transfer by giving a finite
thickness and thermal conductivity to one of the walls.  The example is modified
to a configuration for which there exists a fully developed solution which can be
found in closed form: given a uniform heat flux over each of the walls, the
temperature field asymptotically is the superposition of a uniform longitudinal
gradient and a transverse profile.

.. figure:: https://user-images.githubusercontent.com/973268/142778186-99d8e02e-d02e-4b54-ac09-53bda0591dac.png

   A comparison of inlet and outlet temperature profiles in Example 28.

See the source code of :exlink:`28` for more information.

Example 39: One-dimensional heat equation
-----------------------------------------

This examples reduces the two-dimensional heat equation of Example 19 to
demonstrate the special post-processing required.

.. figure:: https://user-images.githubusercontent.com/973268/152611200-a427dcc1-9425-40af-b6fd-23f66ed18088.gif

   The solution of Example 39.

See the source code of :exlink:`39`
for more information.

Electromagnetism


Miscellaneous
=============

Example 10: Nonlinear minimal surface problem
---------------------------------------------

This example solves the nonlinear minimal surface problem :math:`\nabla \cdot
\left(\frac{1}{\sqrt{1 + \|u\|^2}} \nabla u \right)= 0` with :math:`u=g`
prescribed on the boundary of the square domain.  The nonlinear problem is
linearized using the Newton's method with an analytical Jacobian calculated by
hand.

.. figure:: https://user-images.githubusercontent.com/973268/87663902-1c658780-c76d-11ea-9e00-324a18769ad2.png

   The solution of Example 10.

See the source code of :exlink:`10` for more information.

Example 16: Legendre's equation
-------------------------------

This example solves the eigenvalue problem :math:`((1 - x^2) u')' + k u = 0` in
:math:`(-1,1)`.

.. figure:: https://user-images.githubusercontent.com/973268/87775206-65c9db80-c82e-11ea-8c49-bf191915602a.png

   The six first eigenmodes of Example 16.

See the source code of :exlink:`16` for more information.

Example 31: Curved elements
---------------------------

This example solves the eigenvalue problem :math:`-\Delta u = \lambda u`
with the boundary condition :math:`u|_{\partial \Omega} = 0` using isoparametric
mapping via biquadratic basis and finite element approximation using fifth-order
quadrilaterals.

.. figure:: https://user-images.githubusercontent.com/973268/87859068-32c13e00-c93b-11ea-984d-684e1e4c5066.png

   An eigenmode of Example 31 in a curved mesh.

See the source code of :exlink:`31` for more information.

Example 33: H(curl) conforming model problem
--------------------------------------------

This example solves the vector-valued problem :math:`\nabla \times \nabla \times
E + E = f` in domain :math:`\Omega = [-1, 1]^3` with the boundary condition
:math:`E \times n|_{\partial \Omega} = 0` using the lowest order Nédélec edge
element.

.. figure:: https://user-images.githubusercontent.com/973268/87859239-47520600-c93c-11ea-8241-d62fdfd2a9a2.png

   The solution of Example 33 with the colors given by the magnitude
   of the vector field.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

See the source code of :exlink:`33` for more information.

Example 35: Characteristic impedance and velocity factor
--------------------------------------------------------

This example solves the series inductance (per meter) and parallel capacitance
(per meter) of RG316 coaxial cable. These values are then used to compute the
characteristic impedance and velocity factor of the cable.

.. figure:: https://user-images.githubusercontent.com/973268/87859275-85e7c080-c93c-11ea-9e62-3a9a8ee86070.png

   The results of Example 35.

See the source code of :exlink:`35` for more information.

Example 44: Wave equation
-------------------------

This example solves the one-dimensional wave equation :math:`u_{tt} = c^2
u_{xx}` by reducing it to a first order system.

.. figure:: https://user-images.githubusercontent.com/973268/152610782-8c7534ba-3a47-4bfd-bf59-c7a4f187fac9.gif

   The results of Example 44.

See the source code of :exlink:`44` for more information.
