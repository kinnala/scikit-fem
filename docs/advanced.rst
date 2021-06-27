===============
Advanced topics
===============

This section contains advanced discussions around the features of scikit-fem
with an aim to develop a more detailed understanding of the library.

.. _forms:

Anatomy of forms
================

We consider forms as the basic building blocks of finite element assembly.
Thus, it is useful to understand how forms are used in scikit-fem and how to
express them correctly.

Let us begin with examples.  The bilinear form corresponding to the Laplace
operator :math:`-\Delta` is

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x.

In order to express this in scikit-fem, we write the integrand as a Python
function:

.. doctest::

   >>> from skfem import BilinearForm
   >>> from skfem.helpers import grad, dot
   >>> @BilinearForm
   ... def integrand(u, v, w):
   ...    return dot(grad(u), grad(v))

A typical load vector is given by the :math:`L^2` inner product of a user-given
function and the test function :math:`v`, e.g.,

.. math::

   b(v) = \int_\Omega \sin(\pi x) \sin(\pi y) v \,\mathrm{d}x.

This can be written as

.. doctest::

   >>> import numpy as np
   >>> from skfem import LinearForm
   >>> @LinearForm
   ... def loading(v, w):
   ...    return np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v

In addition, forms can depend on the local mesh parameter ``w.h`` or other
finite element functions (see :ref:`predefined`).
Moreover, boundary forms can depend on the normal vector ``w.n``.
One example is the form

.. math::

   l(\boldsymbol{v}) = \int_{\partial \Omega} \boldsymbol{v} \cdot \boldsymbol{n} \,\mathrm{d}s

which can be written as

.. doctest::

   >>> from skfem import LinearForm
   >>> from skfem.helpers import dot
   >>> @LinearForm
   ... def loading(v, w):
   ...    return dot(w.n, v)

The helper functions such as ``dot`` are discussed further in :ref:`helpers`.

.. _formsreturn:

Forms return NumPy arrays
-------------------------

The form definition always returns a two-dimensional NumPy array.  This can be
verified using the Python debugger:

.. code-block:: python

   from skfem import *
   from skfem.helpers import grad, dot
   @BilinearForm
   def integrand(u, v, w):
       import pdb; pdb.set_trace()  # breakpoint
       return dot(grad(u), grad(v))

Saving the above snippet as ``test.py`` and running it via ``python -i test.py``
allows experimenting:

.. code-block:: none

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, Basis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) dot(grad(u), grad(v))
   array([[2., 2., 2.],
          [1., 1., 1.]])

Notice how ``dot(grad(u), grad(v))`` is a NumPy array with the shape `number of
elements` x `number of quadrature points per element`.  The return value should
always have such shape no matter which mesh or element type is used.

.. _helpers:

Helpers are useful but not necessary
------------------------------------

The module :mod:`skfem.helpers` contains functions that make the forms more
readable.  An alternative way to write the above form is

.. doctest:: python

   >>> from skfem import BilinearForm
   >>> @BilinearForm
   ... def integrand(u, v, w):
   ...     return u[1][0] * v[1][0] + u[1][1] * v[1][1]

.. note::

    In fact, ``u`` and ``v`` are simply tuples of NumPy arrays with the values
    of the function at ``u[0]`` and the values of the gradient at ``u[1]`` (and
    some additional magic such as implementing ``__array__`` and ``__mul__`` so
    that expressions such as ``u * v`` work as expected).

Notice how the shape of ``u[0]`` is what we expect also from the return value as discussed in :ref:`formsreturn`:

.. code-block:: none

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, Basis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !u[0]
   array([[0.66666667, 0.16666667, 0.16666667],
          [0.66666667, 0.16666667, 0.16666667]])


.. _dofindexing:

Indexing of the degrees-of-freedom
==================================

.. warning::

   This section contains lower level details on the order of the DOFs.
   Read this only if you did not find an answer in :ref:`finddofs`.

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
   >>> basis = Basis(m, ElementTriP2())

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
