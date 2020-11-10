=============================
 Setting boundary conditions
=============================

Using the techniques described in :ref:`overview` and :ref:`forms`, one obtains
the linear system

.. math::
   Ax = b

where :math:`A` corresponds to a bilinear form and :math:`b` corresponds to a
linear form.
Many times this system has no unique solution unless the degrees-of-freedom
(DOFs) :math:`x` are further constrained by imposing boundary conditions.

Essential boundary conditions
=============================

It is possible to eliminate DOFs from the resulting system if some
of the values are known `a priori`.  Suppose that the vector :math:`x` can be
split as

.. math::

   x = \begin{bmatrix}
       x_I\\
       x_D
   \end{bmatrix}

where :math:`x_D` are known and :math:`x_I` are unknown.  This allows splitting
the linear system as

.. math::

   \begin{bmatrix}
       A_{II} & A_{ID}\\
       A_{DI} & A_{DD}
   \end{bmatrix}
   \begin{bmatrix}
       x_I\\
       x_D
   \end{bmatrix}
   =
   \begin{bmatrix}
       b_I\\
       b_D
   \end{bmatrix}

which leads to the condensed system

.. math::

   A_{II} x_I = b_I - A_{ID} x_D.


As an example, let us assemble the matrix :math:`A` and the vector :math:`b`
corresponding to the Poisson equation :math:`-\Delta u = 1`.

.. doctest::

   >>> from skfem import *
   >>> from skfem.models.poisson import laplace, unit_load
   >>> m = MeshTri()
   >>> m.refine(2)
   >>> basis = InteriorBasis(m, ElementTriP1())
   >>> A = laplace.assemble(basis)
   >>> b = unit_load.assemble(basis)

The condensed system is obtained with :func:`skfem.utils.condense`.  Below
we provide the DOFs to eliminate via the keyword argument
``D``.

.. doctest::

   >>> AII, bI, xI, I = condense(A, b, D=m.boundary_nodes())
   >>> AII.todense()
   matrix([[ 4.,  0.,  0.,  0., -1., -1., -1., -1.,  0.],
           [ 0.,  4.,  0.,  0., -1.,  0., -1.,  0.,  0.],
           [ 0.,  0.,  4.,  0.,  0., -1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  4., -1., -1.,  0.,  0.,  0.],
           [-1., -1.,  0., -1.,  4.,  0.,  0.,  0.,  0.],
           [-1.,  0., -1., -1.,  0.,  4.,  0.,  0.,  0.],
           [-1., -1.,  0.,  0.,  0.,  0.,  4.,  0., -1.],
           [-1.,  0., -1.,  0.,  0.,  0.,  0.,  4., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  4.]])
    >>> bI
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
           0.0625])

By default, the eliminated DOFs are set to zero.
Different values can be provided through the keyword argument ``x``;
see :ref:`ex14`.

.. _finddofs:

Finding degrees-of-freedom
==========================

Often the goal is to constrain DOFs on a specific part of
the boundary.  Currently the main tools for finding DOFs are
:meth:`skfem.assembly.Basis.find_dofs` and
:meth:`skfem.assembly.Basis.get_dofs`.  Let us demonstrate
the latter with an example.

.. doctest::

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m.refine(2)
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

Indexing of the degrees-of-freedom
==================================

.. warning::

   This section contains lower level details on the order of the DOFs.
   Read this only if you did not find an answer in the previous section.

The DOFs :math:`x` are ordered automatically based on the mesh and the element
type.  It is possible to investigate manually how the DOFs match the different
topological entities (`nodes`, `facets`, `edges`, `elements`) of the mesh.

.. note::

   **Nomenclature:** In scikit-fem, `edges` exist only for three-dimensional
   meshes so that `facets` are something always shared between two elements of
   the mesh.  In particular, we refer to the edges of triangular and
   quadrilateral meshes as `facets`.

For example, consider the quadratic Lagrange triangle and the default two
element mesh of the unit square:

.. doctest::

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m
   Triangular mesh with 4 vertices and 2 elements.
   >>> basis = InteriorBasis(m, ElementTriP2())

The DOFs corresponding to the nodes (or vertices) of the mesh are

.. doctest::

   >>> basis.nodal_dofs
   array([[0, 1, 2, 3]])

The first column above corresponds to the first column in the corresponding mesh
data structure:

.. doctest::

   >>> m.p
   array([[0., 1., 0., 1.],
          [0., 0., 1., 1.]])

In particular, the node at :math:`(0,0)` corresponds to the first element of the
vector :math:`x`, the node at :math:`(1,0)` corresponds to the second element,
and so on.

Similarly, the DOFs corresponding to the facets of the mesh are

.. doctest::

   >>> basis.facet_dofs
   array([[4, 5, 6, 7, 8]])

The corresponding facets can be found in the mesh data structure:

.. doctest::

   >>> m.facets
   array([[0, 0, 1, 1, 2],
          [1, 2, 2, 3, 3]])
   >>> .5 * m.p[:, m.facets].sum(axis=0)  # midpoints of the facets
   array([[0. , 0. , 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 1. , 1. ]])
   
Each DOF is associated either with a node (``nodal_dofs``), a facet
(``facet_dofs``), an edge (``edge_dofs``), or an element (``interior_dofs``).

Setting the degrees-of-freedom via a projection
===============================================

Defining the values of the boundary DOFs is not always easy, e.g., when the DOF
does not represent a point value or another intuitive quantity.  Then it is
possible to perform an :math:`L^2` projection of the boundary data :math:`u_0`
onto the finite element space :math:`V_h` by solving for the function
:math:`\widetilde{u_0} \in V_h` which satisfies

.. math::

   \int_{\partial \Omega} \widetilde{u_0} v\,\mathrm{d}s = \int_{\partial \Omega} u_0 v\,\mathrm{d}s\quad \forall v \in V_h,

and which is zero in all DOFs inside the domain.
In the following snippet we solve explicitly the above variational problem:

.. doctest::

   >>> from skfem import *
   >>> m = MeshQuad()
   >>> basis = FacetBasis(m, ElementQuadP(3))
   >>> u_0 = lambda x, y: (x * y) ** 3
   >>> M = BilinearForm(lambda u, v, w: u * v).assemble(basis)
   >>> f = LinearForm(lambda v, w: u_0(*w.x) * v).assemble(basis)
   >>> x = solve(*condense(M, f, I=basis.get_dofs()))
   >>> x
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])

Alternatively, you can use :func:`skfem.utils.project` which does exactly the
same thing:

.. doctest::

   >>> project(u_0, basis_to=basis, I=basis.get_dofs(), expand=True)
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
