=============================
 Setting boundary conditions
=============================

Using the techniques described in :ref:`overview`, one obtains the linear system

.. math::
   Ax = b

where :math:`A` corresponds to a bilinear form and :math:`b` corresponds to a
linear form.
Many times this system has no unique solution unless the degrees-of-freedom
(DOF's) :math:`x` are further constrained by imposing boundary conditions.

Essential boundary conditions
=============================

It is possible to eliminate DOF's from the resulting system if some
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

.. code-block:: python

   >>> from skfem import *
   >>> from skfem.models.poisson import laplace, unit_load
   >>> m = MeshTri()
   >>> m.refine(2)
   >>> basis = InteriorBasis(m, ElementTriP1())
   >>> A = laplace.assemble(basis)
   >>> b = unit_load.assemble(basis)

The condensed system is obtained with :func:`skfem.utils.condense`.  Below
we provide the DOF's to eliminate via the keyword argument
``D``.

.. code-block:: python

   >>> condense(A, b, D=m.boundary_nodes())
   (<9x9 sparse matrix of type '<class 'numpy.float64'>'
           with 33 stored elements in Compressed Sparse Row format>,
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
           0.0625]),
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.]),
    array([ 6, 12, 15, 19, 20, 21, 22, 23, 24]))

By default, the eliminated DOF's are set to zero.
Different values can be provided through the keyword argument ``x``;
see :ref:`ex14`.

Finding degrees-of-freedom
==========================

Often the goal is to constrain DOF's on a specific part of
the boundary.  Currently the main tools for finding DOF's are
:meth:`skfem.assembly.Basis.find_dofs` and
:meth:`skfem.assembly.Basis.get_dofs`.  Let us demonstrate
the latter with an example.

.. code-block:: python

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m.refine(2)
   >>> basis = InteriorBasis(m, ElementTriP2())

We first find the set of facets belonging to the left boundary.

.. code-block:: python

   >>> m.facets_satisfying(lambda x: x[0] == 0.)
   array([ 1,  5, 14, 15])

Next we supply the array of facet indices to
:meth:`skfem.assembly.Basis.get_dofs`

.. code-block:: python

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

The list of all DOF's belonging to the left boundary can be obtained as follows:

.. code-block:: python

   >>> dofs.flatten()
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40])
   

Indexing of the degrees-of-freedom
==================================

.. warning::

   This section contains lower level details on the order of the DOF's.
   Read this only if you did not find an answer in the above sections.

The degrees-of-freedom :math:`x` are ordered automatically based on the mesh and
the element type.  It is possible to investigate manually how the
degrees-of-freedom match the different topological entities (`nodes`, `facets`,
`edges`, `elements`) of the mesh.

.. note::

   In scikit-fem, `edges` exist only for three-dimensional meshes so that
   `facets` are something always shared between two elements of the mesh.  In
   particular, we refer to the edges of triangular and quadrilateral meshes as
   `facets`.

For example, consider the quadratic Lagrange triangle and the default two
element mesh of the unit square:

.. code-block:: python

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m
   Triangular mesh with 4 vertices and 2 elements.
   >>> basis = InteriorBasis(m, ElementTriP2())

The degrees-of-freedom corresponding to the nodes (or vertices) of the mesh are

.. code-block:: python

   >>> basis.nodal_dofs
   array([[0, 1, 2, 3]])

The first column above corresponds to the first column in the corresponding mesh
data structure:

.. code-block:: python

   >>> m.p
   array([[0., 1., 0., 1.],
          [0., 0., 1., 1.]])

In particular, the node at :math:`(0,0)` corresponds to the first element of the
vector :math:`x`, the node at :math:`(1,0)` corresponds to the second element,
and so on.

Similarly, the degrees-of-freedom corresponding to the facets of the mesh are

.. code-block:: python

   >>> basis.facet_dofs
   array([[4, 5, 6, 7, 8]])

The corresponding facets are also present in the mesh data structure:

.. code-block:: python

   >>> m.facets
   array([[0, 0, 1, 1, 2],
          [1, 2, 2, 3, 3]])
   >>> .5 * m.p[:, m.facets].sum(axis=0)  # midpoints of the facets
   array([[0. , 0. , 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 1. , 1. ]])
