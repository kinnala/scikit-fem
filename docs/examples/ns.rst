.. _navierstokes:

Navier–Stokes equations
-----------------------

.. note::
   This example requires the external packages `pygmsh <https://pypi.org/project/pygmsh/>`_ and `pacopy <https://pypi.org/project/pacopy/>`_.

   In this example, `pacopy <https://pypi.org/project/pacopy/>`_ is used to extend the example :ref:`backwardfacingstep0` from creeping flow over a backward-facing step to finite Reynolds number; as in the example on the bifurcation of the Bratu–Gelfand problem :ref:`bratu`, this means defining a residual for the nonlinear problem and its derivatives with respect to the solution and to the parameter (here Reynolds number).

   Compared to the Stokes equations of example :ref:`backwardfacingstep0`, the Navier–Stokes equation has one additional term.  If the problem is nondimensionalized using a characteristic length (here the height of the step) and velocity (the average over the inlet), this term appears multiplied by the Reynolds number, which thus serves as a convenient parameter for numerical continuation.

.. math::

   -\nabla^2 \mathbf u + \nabla p - \mathrm{Re} \mathbf u \cdot\nabla\mathbf u = 0

   \nabla\cdot\mathbf u = 0
