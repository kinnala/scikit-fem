=============
How-to guides
=============

This section contains goal-oriented guides on the features of scikit-fem.

.. _finddofs:

Finding degrees-of-freedom
==========================

Often the goal is to constrain DOFs on a specific part of
the boundary.  The DOF numbers have one-to-one correspondence with
the rows and the columns of the finite element system matrix,
and the DOFs are typically passed to functions
such as :func:`~skfem.utils.condense` or :func:`~skfem.utils.enforce`
that modify the linear system to implement the boundary conditions.

Currently the main tool for finding DOFs is
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs`.
Let us look at the use of this method through an example
with a triangular mesh
and the quadratic Lagrange element:

.. doctest::

   >>> from skfem import MeshTri, Basis, ElementTriP2
   >>> m = MeshTri().refined(2).with_defaults()
   >>> basis = Basis(m, ElementTriP2())

Normally, a list of facet indices is provided
for :meth:`~skfem.assembly.basis.AbstractBasis.get_dofs`
and it will find the matching DOF indices.
However, for convenience,
we can provide an indicator function to
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs` and it will find the
DOFs on the matching facets (via their midpoints):

.. doctest::

   >>> dofs = basis.get_dofs(lambda x: x[0] == 0.)
   >>> dofs.nodal
   {'u': array([ 0,  2,  5, 10, 14], dtype=int32)}
   >>> dofs.facet
   {'u': array([26, 30, 39, 40], dtype=int32)}

Here is a visualization of the nodal DOFs:

.. plot::

   from skfem import *
   from skfem.visuals.matplotlib import *
   m = MeshTri().refined(2)
   basis = Basis(m, ElementTriP2())
   ax = draw(m)
   for dof in basis.nodal_dofs.flatten():
       ax.text(*basis.doflocs[:, dof], str(dof))

This quadratic Lagrange element has one DOF per node and one DOF per
facet.  Globally, the facets have their own numbering scheme starting
from zero, while the corresponding DOFs are offset by the total
number of nodal DOFs:

.. doctest::

   >>> dofs.facet['u']
   array([26, 30, 39, 40], dtype=int32)

Here is a visualization of the facet DOFs:

.. plot::

   from skfem import *
   from skfem.visuals.matplotlib import *
   m = MeshTri().refined(2)
   basis = Basis(m, ElementTriP2())
   ax = draw(m)
   for dof in basis.facet_dofs.flatten():
       ax.text(*basis.doflocs[:, dof], str(dof))

The keys in the above dictionaries indicate the type of the DOF according to
the following table:

+-----------+---------------------------------------------------------------+
| Key       | Description                                                   |
+===========+===============================================================+
| ``u``     | Point value                                                   |
+-----------+---------------------------------------------------------------+
| ``u_n``   | Normal derivative                                             |
+-----------+---------------------------------------------------------------+
| ``u_x``   | Partial derivative w.r.t. :math:`x`                           |
+-----------+---------------------------------------------------------------+
| ``u_xx``  | Second partial derivative w.r.t :math:`x`                     |
+-----------+---------------------------------------------------------------+
| ``u^n``   | Normal component of a vector field (e.g., Raviart-Thomas)     |
+-----------+---------------------------------------------------------------+
| ``u^t``   | Tangential component of a vector field (e.g., Nédélec)        |
+-----------+---------------------------------------------------------------+
| ``u^1``   | First component of a vector field                             |
+-----------+---------------------------------------------------------------+
| ``u^1_x`` | Partial derivative of the first component w.r.t. :math:`x`    |
+-----------+---------------------------------------------------------------+
| ``u^1^1`` | First component of the first component in a composite field   |
+-----------+---------------------------------------------------------------+
| ``NA``    | Description not available (e.g., hierarchical or bubble DOFs) |
+-----------+---------------------------------------------------------------+

Most of the time we just want an array of all DOFs with the key ``u``.
This can be obtained as follows:

.. doctest::

   >>> dofs.all(['u'])
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40], dtype=int32)
   >>> dofs.flatten()  # all DOFs, no matter which key
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40], dtype=int32)

If a set of facets is tagged, the name of the tag can be passed
also to :meth:`~skfem.assembly.basis.AbstractBasis.get_dofs`:

.. doctest::

   >>> dofs = basis.get_dofs('left')
   >>> dofs.flatten()
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40], dtype=int32)
   
Many DOF types have a well-defined location.  These DOF locations can be found
as follows:

.. doctest::

   >>> basis.doflocs[:, dofs.flatten()]
   array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
          [0.   , 1.   , 0.5  , 0.25 , 0.75 , 0.125, 0.875, 0.375, 0.625]])

.. plot::

   from skfem import *
   from skfem.visuals.matplotlib import *
   m = MeshTri().refined(2).with_defaults()
   basis = Basis(m, ElementTriP2())
   dofs = basis.get_dofs('left')
   ax = draw(m)
   for dof in dofs.flatten():
       ax.plot(*basis.doflocs[:, dof], 'ro')
       ax.text(*basis.doflocs[:, dof], str(dof))

See :ref:`dofindexing` for more details.

.. _l2proj:

Performing projections
======================

A common issue in finite element analysis is that you have either
an analytical function with a given expression or a finite element
function defined on one basis, while what you would like to have
instead is the same function defined on another finite element basis.

We can use :math:`L^2` projection to find discrete counterparts of functions or
transform from one finite element basis to another.  For example,
suppose we have
:math:`u_0(x,y) = x^3 y^3` defined on the boundary of the domain and want to
find the corresponding discrete function which is extended by zero in the
interior of the domain.  Technically, you could explicitly assemble and solve the linear
system corresponding to: find :math:`\widetilde{u_0} \in V_h` satisfying

.. math::

   \int_{\partial \Omega} \widetilde{u_0} v\,\mathrm{d}s = \int_{\partial \Omega} u_0 v\,\mathrm{d}s\quad \forall v \in V_h.

However, this is so common that we have a shortcut command
:meth:`~skfem.assembly.basis.AbstractBasis.project`:

.. doctest::

   >>> import numpy as np
   >>> from skfem import *
   >>> m = MeshQuad().refined(2)
   >>> basis = FacetBasis(m, ElementQuad1())
   >>> u0 = lambda x: x[0] ** 3 * x[1] ** 3
   >>> u0t = basis.project(u0)
   >>> np.abs(np.round(u0t, 5))
   array([1.0000e-05, 8.9000e-04, 9.7054e-01, 8.9000e-04, 6.0000e-05,
          6.0000e-05, 1.0944e-01, 1.0944e-01, 0.0000e+00, 2.0000e-05,
          2.0000e-05, 2.4000e-04, 8.0200e-03, 3.9797e-01, 3.9797e-01,
          2.4000e-04, 8.0200e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])

.. plot::

   import skfem as fem
   m = fem.MeshQuad().refined(2)
   basis = fem.FacetBasis(m, fem.ElementQuad1())
   u0 = lambda x: x[0] ** 3 * x[1] ** 3
   u0t = basis.project(u0)
   ibasis = fem.InteriorBasis(m, fem.ElementQuad1())
   from skfem.visuals.matplotlib import plot, draw
   ax = draw(ibasis)
   plot(ibasis, u0t, nrefs=3, ax=ax, colorbar=True, shading='gouraud')

As another example, we can also project over the entire domain:

.. doctest::

   >>> basis = Basis(m, ElementQuad1())
   >>> f = lambda x: np.sin(2. * np.pi * x[0]) + 1.
   >>> fh = basis.project(f)
   >>> np.abs(np.round(fh, 5))
   array([1.09848, 0.90152, 0.90152, 1.09848, 1.     , 1.09848, 0.90152,
          1.     , 1.     , 2.19118, 1.09848, 0.19118, 0.90152, 0.90152,
          0.19118, 1.09848, 2.19118, 1.     , 2.19118, 0.19118, 1.     ,
          2.19118, 0.19118, 0.19118, 2.19118])

.. plot::

   import skfem as fem
   m = fem.MeshQuad().refined(2)
   basis = fem.CellBasis(m, fem.ElementQuad1())
   f = lambda x: np.sin(2. * np.pi * x[0]) + 1.
   fh = basis.project(f)
   from skfem.visuals.matplotlib import plot, draw
   ax = draw(basis)
   plot(basis, fh, nrefs=3, ax=ax, colorbar=True, shading='gouraud')

Or alternatively, we can use the same command to
project from one finite element basis to another:

.. doctest::

   >>> basis0 = basis.with_element(ElementQuad0())
   >>> fh = basis0.project(basis.interpolate(fh))
   >>> np.abs(np.round(fh, 5))
   array([1.64483, 0.40441, 0.40441, 1.64483, 1.59559, 0.35517, 0.35517,
          1.59559, 1.59559, 0.35517, 0.35517, 1.59559, 1.64483, 0.40441,
          0.40441, 1.64483])

.. plot::

   from skfem import *
   m = MeshQuad().refined(2)
   basis = CellBasis(m, ElementQuad1())
   basis0 = basis.with_element(ElementQuad0())
   f = lambda x: np.sin(2. * np.pi * x[0]) + 1.
   fh = basis.project(f)
   fh = basis0.project(basis.interpolate(fh))
   from skfem.visuals.matplotlib import plot, draw
   ax = draw(basis)
   plot(basis0, fh, nrefs=3, ax=ax, colorbar=True, shading='gouraud')

We can also interpolate the gradient at quadrature points and then project:

.. doctest::

   >>> f = lambda x: np.sin(2. * np.pi * x[0]) + 1.
   >>> fh = basis.project(f)  # P1
   >>> fh = basis.project(basis.interpolate(fh).grad[0])  # df/dx
   >>> np.abs(np.round(fh, 5))
   array([6.6547 , 6.6547 , 6.6547 , 6.6547 , 7.04862, 6.6547 , 6.6547 ,
          7.04862, 7.04862, 0.19696, 6.6547 , 0.19696, 6.6547 , 6.6547 ,
          0.19696, 6.6547 , 0.19696, 7.04862, 0.19696, 0.19696, 7.04862,
          0.19696, 0.19696, 0.19696, 0.19696])

.. plot::

   from skfem import *
   m = MeshQuad().refined(2)
   basis = CellBasis(m, ElementQuad1())
   basis0 = basis.with_element(ElementQuad0())
   f = lambda x: np.sin(2. * np.pi * x[0]) + 1.
   fh = basis.project(f)
   fh = basis.project(basis.interpolate(fh).grad[0])
   from skfem.visuals.matplotlib import plot, draw
   ax = draw(basis)
   plot(basis, fh, nrefs=3, ax=ax, colorbar=True, shading='gouraud')

.. _predefined:

Discrete functions in the forms
===============================

It is a common pattern to reuse an existing finite element function in the forms.
Everything within the form is expressed at the quadrature points and
the finite element functions must be interpolated
from nodes to the
quadrature points through :meth:`~skfem.assembly.basis.AbstractBasis.interpolate`.

For example, consider a fixed-point iteration for the
nonlinear problem

.. math::

   \begin{aligned}
      -\nabla \cdot ((u + \tfrac{1}{10})\nabla u) &= 1 \quad \text{in $\Omega$}, \\
      u &= 0 \quad \text{on $\partial \Omega$}.
   \end{aligned}

We can repeatedly find :math:`u_{k+1} \in H^1_0(\Omega)` which satisfies

.. math::

   \int_\Omega (u_{k} + \tfrac{1}{10}) \nabla u_{k+1} \cdot \nabla v \,\mathrm{d}x = \int_\Omega v\,\mathrm{d}x \quad \forall v \in H^1_0(\Omega).

The bilinear form depends on the previous solution :math:`u_k`
which can be defined as follows:

.. doctest::

   >>> import skfem as fem
   >>> from skfem.models.poisson import unit_load
   >>> from skfem.helpers import grad, dot
   >>> @fem.BilinearForm
   ... def bilinf(u, v, w):
   ...     return (w.u_k + .1) * dot(grad(u), grad(v))

The previous solution :math:`u_k` is interpolated at quadrature points using
:meth:`~skfem.assembly.CellBasis.interpolate` and then provided to
:meth:`~skfem.assembly.BilinearForm.assemble` as a keyword argument:

.. doctest::

   >>> m = fem.MeshTri().refined(3)
   >>> basis = fem.Basis(m, fem.ElementTriP1())
   >>> b = unit_load.assemble(basis)
   >>> x = 0. * b.copy()
   >>> for itr in range(20):  # fixed point iteration
   ...     A = bilinf.assemble(basis, u_k=basis.interpolate(x))
   ...     x = fem.solve(*fem.condense(A, b, I=m.interior_nodes()))
   ...     print(round(x.max(), 10))
   0.7278262868
   0.1956340215
   0.3527261363
   0.2745541843
   0.3065381711
   0.2921831118
   0.298384264
   0.2956587119
   0.2968478347
   0.2963273314
   0.2965548428
   0.2964553357
   0.2964988455
   0.2964798184
   0.2964881386
   0.2964845003
   0.2964860913
   0.2964853955
   0.2964856998
   0.2964855667

.. plot::

   import skfem as fem
   from skfem.models.poisson import unit_load
   from skfem.helpers import grad, dot
   @fem.BilinearForm
   def bilinf(u, v, w):
       return (w.u_k + .1) * dot(grad(u), grad(v))
   m = fem.MeshTri().refined(4)
   basis = fem.Basis(m, fem.ElementTriP1())
   b = unit_load.assemble(basis)
   x = 0. * b.copy()
   for itr in range(20):  # fixed point iteration
       A = bilinf.assemble(basis, u_k=basis.interpolate(x))
       x = fem.solve(*fem.condense(A, b, I=m.interior_nodes()))
   from skfem.visuals.matplotlib import *
   plot(basis, x, colorbar=True, nrefs=3, shading='gouraud')

.. note::

    Inside the form definition, ``w`` is a dictionary of user provided
    arguments and additional default keys.  By default, ``w['x']`` (accessible
    also as ``w.x``) corresponds to the global coordinates and ``w['h']``
    (accessible also as ``w.h``) corresponds to the local mesh parameter.

.. _postprocessing:

Postprocessing the solution
===========================

After solving the finite element system :math:`Ax=b`, it is common to
calculate derived quantities.
The most common techniques are:

* Calculating gradient fields using the technique described at the end of :ref:`l2proj`.
* Using :class:`~skfem.assembly.form.functional.Functional` wrapper to calculate integrals
  over the finite element solution.

The latter consists of writing the integrand as a function
and decorating it using :class:`~skfem.assembly.form.functional.Functional`.
This is similar to the use of :class:`~skfem.assembly.form.bilinear_form.BilinearForm`
and :class:`~skfem.assembly.form.linear_form.LinearForm` wrappers
expect that the function wrapped by :class:`~skfem.assembly.form.functional.Functional`
should accept only a single argument ``w``.

The parameter ``w`` is a dictionary containing all the default keys
(e.g., ``w['h']`` for mesh parameter and ``w['x']`` for global
coordinates) and any user provided arguments that can
be arbitrary finite element functions interpolated at the quadrature points
using :meth:`~skfem.assembly.basis.AbstractBasis.interpolate`.
As a simple example, we calculate the integral of the finite element
solution to the Poisson problem with a unit load:

.. doctest::

   >>> from skfem import *
   >>> from skfem.models.poisson import laplace, unit_load
   >>> mesh = MeshTri().refined(2).with_defaults()
   >>> basis = Basis(mesh, ElementTriP2())
   >>> A = laplace.assemble(basis)
   >>> b = unit_load.assemble(basis)
   >>> x = solve(*condense(A, b, D=basis.get_dofs('left')))
   >>> @Functional
   ... def integral(w):
   ...    return w['uh']  # grad, dot, etc. can be used here
   >>> float(round(integral.assemble(basis, uh=basis.interpolate(x)), 5))
   0.33333

.. plot::

   from skfem import *
   from skfem.models.poisson import laplace, unit_load
   basis = Basis(MeshTri().refined(2).with_defaults(), ElementTriP2())
   A = laplace.assemble(basis)
   b = unit_load.assemble(basis)
   x = solve(*condense(A, b, D=basis.get_dofs('left')))
   from skfem.visuals.matplotlib import plot
   plot(basis, x, nrefs=3, shading='gouraud', colorbar=True)

Similarly we can calculate the integral of its derivative:

   >>> @Functional
   ... def diffintegral(w):
   ...    return w['uh'].grad[0]  # derivative wrt x
   >>> float(round(diffintegral.assemble(basis, uh=basis.interpolate(x)), 5))
   0.5

We can also calculate integrals over the boundary
using :class:`~skfem.assembly.basis.facet_basis.FacetBasis`:

   >>> fbasis = basis.boundary('left')
   >>> @Functional
   ... def bndintegral(w):
   ...    return w['uh'].grad[1]  # derivative wrt y
   >>> float(round(bndintegral.assemble(fbasis, uh=fbasis.interpolate(x)), 5))
   0.0

.. _visualizing:

Visualizing the solution
========================

After solving the finite element system :math:`Ax=b`, it is common to
visualize the solution or other
derived fields.  The library includes some basic 2D visualization
routines implemented using matplotlib.  For more complex
visualizations, we suggest that the solution is saved to VTK and a
visualization is created using Paraview.

The main routine for creating 2D visualizations using matplotlib is
:func:`skfem.visuals.matplotlib.plot` and it accepts the basis
object and the solution vector as its arguments:

.. doctest::

   >>> from skfem import *
   >>> from skfem.models.poisson import laplace, unit_load
   >>> mesh = MeshTri().refined(2)
   >>> basis = Basis(mesh, ElementTriP2())
   >>> A = laplace.assemble(basis)
   >>> b = unit_load.assemble(basis)
   >>> x = solve(*condense(A, b, D=basis.get_dofs()))
   >>> from skfem.visuals.matplotlib import plot
   >>> plot(basis, x)
   <Axes: >

.. plot::

   from skfem import *
   from skfem.models.poisson import laplace, unit_load
   basis = Basis(MeshTri().refined(2), ElementTriP2())
   A = laplace.assemble(basis)
   b = unit_load.assemble(basis)
   x = solve(*condense(A, b, D=basis.get_dofs()))
   from skfem.visuals.matplotlib import plot
   plot(basis, x)

It accepts various optional arguments to make the plots
nicer, some of which have been demonstrated in :ref:`gallery`.
For example, here is the same solution as above with different settings:

.. doctest::

   >>> plot(basis, x, shading='gouraud', colorbar={'orientation': 'horizontal'}, nrefs=3)
   <Axes: >

.. plot::

   from skfem import *
   from skfem.models.poisson import laplace, unit_load
   basis = Basis(MeshTri().refined(2), ElementTriP2())
   A = laplace.assemble(basis)
   b = unit_load.assemble(basis)
   x = solve(*condense(A, b, D=basis.get_dofs()))
   from skfem.visuals.matplotlib import plot
   plot(basis, x, shading='gouraud', colorbar={'orientation': 'horizontal'}, nrefs=3)

The routine is based on `matplotlib.pyplot.tripcolor <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tripcolor.html>`_ command and shares
its limitations.
For more control we suggest that the solution is saved to a VTK file for
visualization in Paraview.  Saving of the solution is done through the
mesh object and it requires giving one number per node of the mesh.
Thus, for other than piecewise linear finite element bases,
the mesh must be refined and the solution interpolated for visualization:

.. doctest::

   >>> rmesh, rx = basis.refinterp(x, nrefs=3)  # refine and interpolate
   >>> rmesh.save('sol.vtk', point_data={'x': rx})

Another option would be to first project the solution
onto a piecewise linear finite element basis
as described in :ref:`l2proj`.
Please see :ref:`gallery` for more examples of visualization.
