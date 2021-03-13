=============
How-to guides
=============

This section contains goal-oriented guides on the features of scikit-fem.

Choosing a finite element
=========================

Here are some instructions for choosing an
:class:`~skfem.element.Element` class.  Firstly, the naming of the element
classes reflects their compatibility with the mesh types:

>>> from skfem.element import ElementTriP1
>>> ElementTriP1.refdom
<class 'skfem.refdom.RefTri'>

Secondly, the chosen finite element should be compatible with the approximated
partial differential equation.  Here are some general guidelines:

* Use subclasses of :class:`~skfem.element.ElementH1`, e.g.,
  :class:`~skfem.element.ElementTriP1`, :class:`~skfem.element.ElementTriP2`,
  :class:`~skfem.element.ElementQuad2`, :class:`~skfem.element.ElementTetP2` or
  :class:`~skfem.element.ElementHex2`, for standard second-order problems.
* Discretize vectorial problems either by manually building the block matrices
  (e.g., using ``scipy.sparse.bmat``) with scalar elements, or by using
  :class:`~skfem.element.ElementVector` or
  :class:`~skfem.element.ElementComposite` that abstract out the creation of
  the block matrices.
* Pay special attention to constrained problems, e.g., the Stokes system which
  may require the use of special elements such as :class:`~skfem.element.ElementTriMini`.
* Use subclasses of :class:`~skfem.element.ElementHdiv` or
  :class:`~skfem.element.ElementHcurl`, e.g.,
  :class:`~skfem.element.ElementTriRT0` or :class:`~skfem.element.ElementTetN0`,
  for mixed problems with less regular solutions.
* Use subclasses of :class:`ElementGlobal`, e.g., :class:`ElementTriMorley` or
  :class:`ElementTriArgyris`, for fourth-order problems or if there are
  postprocessing requirements, e.g., the need for high-order derivatives.

Thirdly, the finite element should use degrees-of-freedom that are relevant
for the essential boundary conditions that you want to impose.
See :ref:`finddofs` for more information.


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
   >>> basis = fem.InteriorBasis(m, fem.ElementTriP1())
   >>> b = unit_load.assemble(basis)
   >>> x = 0. * b.copy()
   >>> for itr in range(10):  # fixed point iteration
   ...     A = bilinf.assemble(basis, u_k=basis.interpolate(x))
   ...     x = fem.solve(*fem.condense(A, b, I=m.interior_nodes()))
   ...     print(x.max())
   0.07278262867647059
   0.07030433694174187
   0.07036045457157739
   0.07035940302769318
   0.07035942072395032
   0.07035942044353624
   0.07035942044783286
   0.07035942044776827
   0.07035942044776916
   0.07035942044776922

Inside the form definition, ``w`` is a dictionary of user provided arguments and
additional default keys.
By default, ``w['x']`` (accessible also as ``w.x``) corresponds to the global
coordinates and ``w['h']`` (accessible also as ``w.h``) corresponds to the local
mesh parameter.


.. _finddofs:

Finding degrees-of-freedom
==========================

Often the goal is to constrain DOFs on a specific part of
the boundary.  Currently the main tools for finding DOFs are
:meth:`skfem.assembly.Basis.find_dofs` and
:meth:`skfem.assembly.Basis.get_dofs`.  Let us demonstrate
the latter with an example.

.. doctest::

   >>> from skfem import MeshTri, InteriorBasis, ElementTriP2
   >>> m = MeshTri().refined(2)
   >>> basis = InteriorBasis(m, ElementTriP2())

We first find the set of facets belonging to the left boundary.

.. doctest::

   >>> m.facets_satisfying(lambda x: x[0] == 0.)
   array([ 1,  5, 14, 15])

Next we supply the array of facet indices to
:meth:`skfem.assembly.Basis.get_dofs`

.. doctest::

   >>> dofs = basis.get_dofs(m.facets_satisfying(lambda x: x[0] == 0.))
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

The list of all DOFs (belonging to the left boundary) can be obtained as
follows:

.. doctest::

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

Defining the values of the boundary DOFs is not always easy, e.g., when the DOF
does not represent a point value or another intuitive quantity.  Then it is
possible to perform an :math:`L^2` projection of the boundary data :math:`u_0`
onto the finite element space :math:`V_h` by solving for the function
:math:`\widetilde{u_0} \in V_h` which satisfies

.. math::

   \int_{\partial \Omega} \widetilde{u_0} v\,\mathrm{d}s = \int_{\partial \Omega} u_0 v\,\mathrm{d}s\quad \forall v \in V_h,

and which is zero in all DOFs inside the domain.
Below we solve explicitly the above variational problem:

.. doctest::

   >>> import skfem as fem
   >>> m = fem.MeshQuad()
   >>> basis = fem.FacetBasis(m, fem.ElementQuadP(3))
   >>> u_0 = lambda x, y: (x * y) ** 3
   >>> M = fem.BilinearForm(lambda u, v, w: u * v).assemble(basis)
   >>> f = fem.LinearForm(lambda v, w: u_0(*w.x) * v).assemble(basis)
   >>> x = fem.solve(*fem.condense(M, f, I=basis.get_dofs()))
   >>> x
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])

Alternatively, you can use :func:`skfem.utils.projection` which does exactly
the same thing:

.. doctest::

   >>> fem.projection(u_0, basis_to=basis, I=basis.get_dofs(), expand=True)
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
