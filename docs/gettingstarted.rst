.. _gettingstarted:

=================
 Getting started
=================

If you have a supported Python installation on your computer, you can
install the package via

.. code-block:: bash

   pip install scikit-fem[all]

You can also try `Google Colab <https://colab.research.google.com/>`_ in your
web browser and install scikit-fem by executing

.. code-block:: bash

   !pip install scikit-fem[all]

Step 1: Clarify the problem
===========================

In this tutorial we solve the Poisson problem

.. math::
   \begin{aligned}
        -\Delta u &= f \quad && \text{in $\Omega$,} \\
        u &= 0 \quad && \text{on $\partial \Omega$,}
   \end{aligned}

where :math:`\Omega = (0, 1)^2` is a square domain
and :math:`f(x,y)=\sin \pi x \sin \pi y`.
The weak formulation reads:
find :math:`u \in H^1_0(\Omega)` satisfying

.. math::
   \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x = \int_\Omega fv\,\mathrm{d}x \quad \forall v \in H^1_0(\Omega).

.. note::

   Above :math:`H^1_0(\Omega)` is the space of functions that are zero
   on the boundary :math:`\partial \Omega` and the square integral of the
   first derivative is finite.  This is a common function space
   to use for second-order boundary value problems because 
   it often corresponds to the space of functions with finite energy.

Step 2: Express the forms as code
=================================

Next we write the forms

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x \quad \text{and} \quad L(v) = \int_\Omega f v \,\mathrm{d}x

as source code.  Each form is written as a function and
decorated as follows:

.. doctest::

   >>> import skfem as fem
   >>> from skfem.helpers import dot, grad  # helpers make forms look nice
   >>> @fem.BilinearForm
   ... def a(u, v, _):
   ...     return dot(grad(u), grad(v))

.. doctest::

   >>> import numpy as np
   >>> @fem.LinearForm
   ... def L(v, w):
   ...     x, y = w.x  # global coordinates
   ...     f = np.sin(np.pi * x) * np.sin(np.pi * y)
   ...     return f * v

Step 3: Create a mesh
=====================

The default constructors of :class:`~skfem.mesh.Mesh` initialize a
unit square:

.. doctest::

   >>> mesh = fem.MeshTri().refined(3)  # refine thrice
   >>> mesh
   <skfem MeshTri1 object>
     Number of elements: 128
     Number of vertices: 81
     Number of nodes: 81
     Named boundaries [# facets]: left [8], bottom [8], right [8], top [8]


Step 4: Define a basis
======================

The mesh is combined with a finite element to form a global
basis.
Here we choose the piecewise-linear basis:

.. doctest::

   >>> Vh = fem.Basis(mesh, fem.ElementTriP1())

Step 5: Assemble the linear system
==================================

Now everything is in place for the finite element assembly.
The resulting matrix has the type ``scipy.sparse.csr_matrix``
and the load vector has the type ``ndarray``.

.. doctest::

   >>> A = a.assemble(Vh)
   >>> l = L.assemble(Vh)
   >>> A.shape
   (81, 81)
   >>> l.shape
   (81,)

Step 6: Find boundary DOFs
==========================

Setting boundary conditions requires finding the degrees-of-freedom (DOFs) on
the boundary.  Empty call to
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs` matches all boundary DOFs.

.. doctest::

   >>> D = Vh.get_dofs()
   >>> D.flatten()
   array([ 0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 25,
          26, 27, 29, 30, 32, 33, 34, 35, 36, 39, 40, 49, 50, 53, 54])

Step 7: Eliminate boundary DOFs and solve
=========================================

The boundary DOFs must be eliminated from the linear system :math:`Ax=l`
to set :math:`u=0` on the boundary.
This can be done using :func:`~skfem.utils.condense`.
The output can be passed to :func:`~skfem.utils.solve`
which is a simple wrapper to ``scipy`` sparse solver:

.. doctest::

   >>> x = fem.solve(*fem.condense(A, l, D=D))
   >>> x.shape
   (81,)

Step 8: Calculate error
=======================

The exact solution is known to be

.. math::

   u(x, y) = \frac{1}{2 \pi^2} \sin \pi x \sin \pi y.

Thus, it makes sense to verify that the error is small.

.. doctest::

   >>> @fem.Functional
   ... def error(w):
   ...     x, y = w.x
   ...     uh = w['uh']
   ...     u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2. * np.pi ** 2)
   ...     return (uh - u) ** 2
   >>> round(error.assemble(Vh, uh=Vh.interpolate(x)), 9)
   1.069e-06
