.. _gettingstarted:

=================
 Getting started
=================

If you have a supported Python installation on your computer, you can
install the package via

.. code-block:: bash

   pip install scikit-fem[all]

Specifying ``[all]`` includes ``meshio`` for mesh input/output, and
``matplotlib`` for simple visualizations.  The minimal dependencies are
``numpy`` and ``scipy``.  You can also install scikit-fem in Jupyter Notebook
or in `Google Colab <https://colab.research.google.com/>`_ by executing

.. code-block:: bash

   !pip install scikit-fem

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

   :math:`H^1_0(\Omega)` is the space of functions that are zero on the
   boundary :math:`\partial \Omega` and have finite energy: the square integral
   of the first derivative is finite.

Step 2: Express the forms as code
=================================

Next we write the forms

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x \quad \text{and} \quad l(v) = \int_\Omega f v \,\mathrm{d}x

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
   ... def l(v, w):
   ...     x, y = w.x  # global coordinates
   ...     f = np.sin(np.pi * x) * np.sin(np.pi * y)
   ...     return f * v

For more information see :ref:`forms`.

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


.. plot::

   from skfem import *
   MeshTri().refined(3).draw()


Step 4: Define a basis
======================

The mesh is combined with a finite element to form a global
basis.
Here we choose the piecewise-linear basis:

.. doctest::

   >>> Vh = fem.Basis(mesh, fem.ElementTriP1())
   >>> Vh
   <skfem CellBasis(MeshTri1, ElementTriP1) object>
     Number of elements: 128
     Number of DOFs: 81
     Size: 27648 B

Step 5: Assemble the linear system
==================================

Now everything is in place for the finite element assembly.
The resulting matrix has the type ``scipy.sparse.csr_matrix``
and the load vector has the type ``ndarray``.

.. doctest::

   >>> A = a.assemble(Vh)
   >>> b = l.assemble(Vh)
   >>> A.shape
   (81, 81)
   >>> b.shape
   (81,)

Step 6: Find boundary DOFs
==========================

Setting boundary conditions requires finding the degrees-of-freedom (DOFs) on
the boundary.  Empty call to
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs` matches all boundary DOFs.

.. doctest::

   >>> D = Vh.get_dofs()
   >>> D
   <skfem DofsView(MeshTri1, ElementTriP1) object>
     Number of nodal DOFs: 32 ['u']

:ref:`finddofs` explains how to match other subsets of DOFs.

Step 7: Eliminate boundary DOFs and solve
=========================================

The boundary DOFs must be eliminated from the linear system :math:`Ax=b`
to set :math:`u=0` on the boundary.
This can be done using :func:`~skfem.utils.condense`
which can be useful also for inhomogeneous Dirichlet conditions.
The output can be passed to :func:`~skfem.utils.solve`
which is a simple wrapper to ``scipy`` sparse solver:

.. doctest::

   >>> x = fem.solve(*fem.condense(A, b, D=D))
   >>> x.shape
   (81,)

.. plot::

   from skfem import *
   from skfem.visuals.matplotlib import *
   from skfem.helpers import dot, grad
   import numpy as np
   basis = Basis(MeshTri().refined(3), ElementTriP1())
   a = BilinearForm(lambda u, v, _: dot(grad(u), grad(v)))
   b = LinearForm(lambda v, w: np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v)
   y = solve(*condense(a.assemble(basis), b.assemble(basis), D=basis.get_dofs()))
   ax = draw(basis)
   plot(basis, y, ax=ax, nrefs=2, colorbar=True, shading='gouraud')

:ref:`visualizing` has some guidelines for visualization
and various other examples can be found in :ref:`gallery`.
   
Step 8: Calculate error
=======================

The exact solution is known to be

.. math::

   u(x, y) = \frac{1}{2 \pi^2} \sin \pi x \sin \pi y.

Thus, it makes sense to verify that the error is small
by calculating the error in the :math:`L^2` norm:

.. doctest::

   >>> @fem.Functional
   ... def error(w):
   ...     x, y = w.x
   ...     uh = w['uh']
   ...     u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2. * np.pi ** 2)
   ...     return (uh - u) ** 2
   >>> str(round(error.assemble(Vh, uh=Vh.interpolate(x)), 9))
   '1.069e-06'

:ref:`postprocessing` covers some ideas behind the use of :class:`~skfem.assembly.form.functional.Functional` wrapper.
