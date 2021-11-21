=============
How-to guides
=============

This section contains goal-oriented guides on the features of scikit-fem.

.. _finddofs:

Finding degrees-of-freedom
==========================

Often the goal is to constrain DOFs on a specific part of
the boundary.  Currently the main tool for finding DOFs is
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs`.

.. doctest::

   >>> from skfem import MeshTri, Basis, ElementTriP2
   >>> m = MeshTri().refined(2)
   >>> basis = Basis(m, ElementTriP2())

We can provide an indicator function to
:meth:`~skfem.assembly.basis.AbstractBasis.get_dofs` and it will call
:meth:`~skfem.mesh.Mesh.facets_satisfying` and find the corresponding DOFs:

.. doctest::

   >>> dofs = basis.get_dofs(lambda x: x[0] == 0.)
   >>> dofs.nodal
   {'u': array([ 0,  2,  5, 10, 14])}
   >>> dofs.facet
   {'u': array([26, 30, 39, 40])}

The keys in the above dictionaries indicate the type of the
DOF according to the following table:

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
| ``u^n``   | Normal component of a vector field (e.g. Raviart-Thomas)      |
+-----------+---------------------------------------------------------------+
| ``u^t``   | Tangential component of a vector field (e.g. Nédélec)         |
+-----------+---------------------------------------------------------------+
| ``u^1``   | First component of a vector field                             |
+-----------+---------------------------------------------------------------+
| ``u^1_x`` | Partial derivative of the first component w.r.t. :math:`x`    |
+-----------+---------------------------------------------------------------+
| ``u^1^1`` | First component of the first component in a composite field   |
+-----------+---------------------------------------------------------------+
| ``NA``    | Description not available (e.g. hierarchical or bubble DOF's) |
+-----------+---------------------------------------------------------------+

An array of all DOFs belonging to the left boundary with the key ``u`` can be
obtained as follows:

.. doctest::

   >>> dofs.all(['u'])
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40])
   >>> dofs.flatten()  # all DOFs, no matter which key
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40])

If a name is associated with the set of facets it can be passed
to :meth:`~skfem.assembly.basis.AbstractBasis.get_dofs`:

.. doctest::

   >>> dofs = basis.get_dofs('left')
   >>> dofs.flatten()
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40])
   
Many DOF types are associated with a specific global coordinate.  These
so-called DOF locations can be found as follows:

.. doctest::

   >>> basis.doflocs[:, dofs.flatten()]
   array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
          [0.   , 1.   , 0.5  , 0.25 , 0.75 , 0.125, 0.875, 0.375, 0.625]])

See :ref:`dofindexing` for more detailed information.

Creating discrete functions via projection
==========================================

It is possible to perform an :math:`L^2` projection of the boundary data
:math:`u_0` onto the finite element space :math:`V_h` by solving for the
function :math:`\widetilde{u_0} \in V_h` which satisfies

.. math::

   \int_{\partial \Omega} \widetilde{u_0} v\,\mathrm{d}s = \int_{\partial \Omega} u_0 v\,\mathrm{d}s\quad \forall v \in V_h.

Below we solve explicitly the above variational problem:

.. doctest::

   >>> import numpy as np
   >>> import skfem as fem
   >>> m = fem.MeshQuad()
   >>> basis = fem.FacetBasis(m, fem.ElementQuadP(3))
   >>> u_0 = lambda x: (x[0] * x[1]) ** 3
   >>> M = fem.BilinearForm(lambda u, v, w: u * v).assemble(basis)
   >>> f = fem.LinearForm(lambda v, w: u_0(w.x) * v).assemble(basis)
   >>> x = fem.solve(*fem.condense(M, f, I=basis.get_dofs()))
   >>> np.abs(np.round(x, 5))
   array([0.     , 0.     , 1.     , 0.     , 0.     , 0.     , 0.     ,
          0.     , 0.61237, 0.15811, 0.61237, 0.15811, 0.     , 0.     ,
          0.     , 0.     ])

Alternatively, you can use :func:`skfem.utils.projection` which does exactly
the same thing:

.. doctest::

   >>> y = fem.projection(u_0, basis, I=basis.get_dofs(), expand=True)
   >>> np.abs(np.round(y, 5))
   array([0.     , 0.     , 1.     , 0.     , 0.     , 0.     , 0.     ,
          0.     , 0.61237, 0.15811, 0.61237, 0.15811, 0.     , 0.     ,
          0.     , 0.     ])

Assembling jump terms
=====================

The shorthand :func:`~skfem.assembly.asm`
supports special syntax for assembling the same form over a list or lists of
bases and summing the result.  Consider the form

.. math::

   b(u,v) = \sum_{E \in \mathcal{E}_h} \int_{E} [u][v]\,\mathrm{d}s

where :math:`\mathcal{E}_h` is the set of interior facets of a mesh
and :math:`[u]` is the jump in the value of :math:`u` over the facet
:math:`E`.
We have
:math:`[u] = u_1 - u_2` and :math:`[v] = v_1 - v_2`
where the subscript denotes the value of the function restricted to one of the
elements sharing a facet.  The form can be split as

.. math::

   b(u,v) = \sum_{E \in \mathcal{E}_h} \left(\int_{E} u_1 v_1\,\mathrm{d}s - \int_{E} u_1 v_2\,\mathrm{d}s - \int_{E} u_2 v_1\,\mathrm{d}s + \int_{E} u_2 v_2\,\mathrm{d}s\right)

and normally we would assemble all four forms separately.

We can instead provide a list of bases during a call to :func:`skfem.assembly.asm`:

.. doctest::

   >>> import skfem as fem
   >>> m = fem.MeshTri()
   >>> e = fem.ElementTriP0()
   >>> bases = [fem.InteriorFacetBasis(m, e, side=k) for k in [0, 1]]
   >>> jumpform = fem.BilinearForm(lambda u, v, p: (-1) ** sum(p.idx) * u * v)
   >>> fem.asm(jumpform, bases, bases).toarray()
   array([[ 1.41421356, -1.41421356],
          [-1.41421356,  1.41421356]])

.. _predefined:

Using discrete functions in forms
=================================

Often we use a previous solution vector in the form
definition, e.g., when solving nonlinear problems or
when evaluating functionals.
A simple fixed-point iteration for

.. math::

   \begin{aligned}
      -\nabla \cdot ((u + 1)\nabla u) &= 1 \quad \text{in $\Omega$}, \\
      u &= 0 \quad \text{on $\partial \Omega$},
   \end{aligned}

corresponds to repeatedly
finding :math:`u_{k+1} \in H^1_0(\Omega)` which satisfies

.. math::

   \int_\Omega (u_{k} + 1) \nabla u_{k+1} \cdot \nabla v \,\mathrm{d}x = \int_\Omega v\,\mathrm{d}x

for every :math:`v \in H^1_0(\Omega)`.
The argument ``w`` is used to define such forms:

.. doctest::

   >>> import skfem as fem
   >>> from skfem.models.poisson import unit_load
   >>> from skfem.helpers import grad, dot
   >>> @fem.BilinearForm
   ... def bilinf(u, v, w):
   ...     return (w.u_k + 1.) * dot(grad(u), grad(v))

The previous solution :math:`u_k` must be provided to
:meth:`~skfem.assembly.BilinearForm.assemble` as a keyword argument:

.. doctest::

   >>> m = fem.MeshTri().refined(3)
   >>> basis = fem.Basis(m, fem.ElementTriP1())
   >>> b = unit_load.assemble(basis)
   >>> x = 0. * b.copy()
   >>> for itr in range(10):  # fixed point iteration
   ...     A = bilinf.assemble(basis, u_k=basis.interpolate(x))
   ...     x = fem.solve(*fem.condense(A, b, I=m.interior_nodes()))
   ...     print(round(x.max(), 10))
   0.0727826287
   0.0703043369
   0.0703604546
   0.070359403
   0.0703594207
   0.0703594204
   0.0703594204
   0.0703594204
   0.0703594204
   0.0703594204

Inside the form definition, ``w`` is a dictionary of user provided arguments and
additional default keys.
By default, ``w['x']`` (accessible also as ``w.x``) corresponds to the global
coordinates and ``w['h']`` (accessible also as ``w.h``) corresponds to the local
mesh parameter.
