=============================
 Setting boundary conditions
=============================

Using the techniques described in :ref:`overview`, one obtains the linear system

.. math::
   Ax = b

where :math:`A` corresponds to a bilinear form and :math:`b` corresponds to a
linear form.
Many times this system has no unique solution unless the degrees-of-freedom
:math:`x` are further constrained by imposing boundary conditions.

Essential boundary conditions
=============================

It is possible to eliminate degrees-of-freedom from the resulting system if some
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

The condensed system can be obtained with :func:`skfem.utils.condense`.  Below
we provide a list of degrees-of-freedom to eliminate via the keyword argument
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

By default, the eliminated degrees-of-freedom are assumed to be zero.
Other values can be provided through the keyword argument ``x``,
cf. :ref:`ex14`.

Finding degrees-of-freedom
==========================

Often we wish to constrain degrees-of-freedom (DOF's) on a specific part of the
domain.  Currently the main tools for finding DOF's are
:meth:`skfem.assembly.Basis.find_dofs` and
:meth:`skfem.assembly.Basis.get_dofs`.

.. code-block:: python

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m.refine(2)
   >>> basis = InteriorBasis(m, ElementTriP2())

Finding DOF's corresponding to the left boundary:


Indexing of the degrees-of-freedom
==================================

.. warning::

   This section contains low-level details on the order of the
   degrees-of-freedom in :math:`x`.

The degrees-of-freedom :math:`x` are ordered automatically based on the mesh and
the element type.  It is possible to investigate how the degrees-of-freedom
match the different topological entities (`nodes`, `facets`, `edges`,
`elements`) of the mesh.

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

In particular, the node at :math:`(0,0)` corresponds to the first value of the
vector :math:`x`, the node at :math:`(1,0)` corresponds to the second value, and
so on.

Similarly, the degrees-of-freedom corresponding to the facets of the mesh are

.. code-block:: python

   >>> basis.facet_dofs
   array([[4, 5, 6, 7, 8]])

The corresponding facets can be also found in the mesh data structure:

.. code-block:: python

   >>> m.facets
   array([[0, 0, 1, 1, 2],
          [1, 2, 2, 3, 3]])
   >>> .5 * m.p[:, m.facets].sum(axis=0)  # midpoints of the facets
   array([[0. , 0. , 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 1. , 1. ]])
