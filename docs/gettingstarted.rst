=================
 Getting started
=================

This tutorial introduces you to scikit-fem.
It assumes that you are familiar with the basics of the finite element method
and how the method relates to partial differential equations.

.. note::

   An overview of the theory of finite element methods is given in "Background".

Step 0: Installing scikit-fem
=============================

If you have a supported Python installation on your computer, you can run

.. code-block:: bash

   pip install scikit-fem

If you do not want to install anything locally, you can try `Google Colab
<https://colab.research.google.com/>`_ in your web browser and install scikit-fem
by executing

.. code-block:: bash

   !pip install scikit-fem

Step 1: Problem statement
=========================

It is benefical to know the weak formulation
before writing any code.  In this tutorial we solve the system

.. math::
   \begin{aligned}
        -\Delta u &= f \quad && \text{in $\Omega$,} \\
        u &= 0 \quad && \text{on $\partial \Omega$,}
   \end{aligned}

where :math:`\Omega = [0, 1]^2` is a square domain
and :math:`f(x,y)=\sin \pi x \sin \pi y`.
The corresponding weak formulation reads:
find :math:`u \in V` satisfying

.. math::
   \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x = \int_\Omega fv\,\mathrm{d}x \quad \forall v \in V,

where :math:`V = \{ v : \int_\Omega (\nabla v)^2 \,\mathrm{d}x < \infty, v|_{\partial \Omega} = 0 \}`.

Step 2: Forms as code
=====================

Next step is to write the forms

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x \quad \text{and} \quad L(v) = \int_\Omega f v \,\mathrm{d}x

as source code.  We write each form as a function and
decorate it accordingly:

.. doctest::

   >>> import skfem as fem
   >>> from skfem.helpers import dot, grad  # helpers make forms look nice
   >>> @fem.BilinearForm
   ... def a(u, v, w):
   ...     return dot(grad(u), grad(v))

.. doctest::

   >>> import numpy as np
   >>> @fem.LinearForm
   ... def L(v, w):
   ...     x, y = w['x']  # global coordinates
   ...     f = np.sin(np.pi * x) * np.sin(np.pi * y)
   ...     return f * v

Step 3: Create a mesh
=====================

By default, all :class:`~skfem.mesh.Mesh` classes initialize a mesh for the
unit square if no parameters are provided to the constructor.

.. doctest::

   >>> fem.MeshQuad()
   Quadrilateral mesh with 4 vertices and 1 elements.
   >>> mesh = fem.MeshQuad().refined(3)  # refine thrice
   >>> mesh
   Quadrilateral mesh with 81 vertices and 64 elements.


Step 4: Define a basis
======================

There are plenty of finite elements supported in scikit-fem.
Here we choose the classical bilinear quadrilateral basis:

.. doctest::

   >>> Vh = fem.InteriorBasis(mesh, ElementQuad1())
