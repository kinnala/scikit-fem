Forced convection
-----------------

We begin the study of forced convection with the plane Graetz problem; viz. the steady distribution of temperature in a plane channel with unit inlet temperature and zero temperature on the walls and a steady laminar unidirectional parabolic plane-Poiseuille flow.

The governing advection–diffusion equation is

.. math::

   \mathrm{Pe} u\cdot\frac{\partial T}{\partial x} = \nabla^2 T

where the velocity profile is

.. math::

   u (y) = 6 y (1 - y), \qquad (0 < y < 1)

The equations here have been nondimensionalized by the width of the channel and the volumetric flow-rate.  The governing parameter is the Péclet number, being the mean velocity times the width divided by the thermal diffusivity.

Because the problem is symmetric about :math:`y = \frac{1}{2}`, only half is solved here, with natural boundary conditions along the centreline.
