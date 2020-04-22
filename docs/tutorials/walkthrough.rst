.. _basic-features:

Overview of basic features
--------------------------

This is an overview of the basic features of scikit-fem that are needed in
most finite element computations.
For a more practical approach, see :ref:`poisson` instead.

Creating meshes
###############

The finite element assembly requires using subclasses of
:class:`~skfem.mesh.Mesh`. Underneath, the mesh is stored in two arrays: vertex
locations (:attr:`skfem.mesh.Mesh.p`) and :math:`n`-tuples defining the elements
(:attr:`skfem.mesh.Mesh.t`).

The default constructors of :class:`~skfem.mesh.Mesh`
classes create simple meshes of unit cubes :math:`\Omega = [0,1]^d`.

.. code-block:: python

   In [1]: from skfem.mesh import MeshTri
   In [2]: m = MeshTri()
   In [3]: m
   Out[3]: "Triangular mesh with 4 vertices and 2 elements."

.. note::

   The supported mesh types are :class:`~skfem.mesh.MeshLine`,
   :class:`~skfem.mesh.MeshTri`, :class:`~skfem.mesh.MeshQuad`,
   :class:`~skfem.mesh.MeshTet`, and :class:`~skfem.mesh.MeshHex`.
   
There are also a few additional constructors available such as
:meth:`~skfem.mesh.MeshTri.init_tensor` and
:meth:`~skfem.mesh.MeshTri.init_lshaped`. More importantly, all mesh types can be
loaded from file formats supported by `meshio
<https://github.com/nschloe/meshio>`_:

.. code-block:: python

   In [4]: from skfem.mesh import Mesh
   In [5]: Mesh.load("docs/examples/square.msh")
   Out[5]: "Triangular mesh with 109 vertices and 184 elements."

.. note::

   It's easy to create meshes with the help of external packages, see
   e.g. :ref:`insulated`.

Meshes can be visualized using
:meth:`skfem.visuals.matplotlib.draw`, or with external tools after exporting them
via :meth:`skfem.mesh.Mesh.save`.

Choosing basis functions
########################

The local basis functions are defined in :class:`~skfem.element.Element`
classes. They are combined with meshes to create
:class:`~skfem.assembly.Basis` objects, such as
:class:`~skfem.assembly.InteriorBasis` and :class:`~skfem.assembly.FacetBasis`,
which contain global basis functions evaluated at global quadrature points:

.. code-block:: python

   In [1]: from skfem import MeshTri, ElementTriP2, InteriorBasis
   In [2]: m = MeshTri()
   In [3]: m.refine(3)
   In [4]: basis = InteriorBasis(m, ElementTriP2())

.. note::

   You can find a list of supported elements from the documentation of
   :mod:`skfem.element`.
   
Here :class:`~skfem.element.ElementTriP2` refers to the quadratic Lagrange
finite element for triangular meshes.  You can control the integration order via
a keyword argument:

.. code-block:: python

   In [5]: basis = InteriorBasis(m, ElementTriP2(), intorder=5)

Defining the integration order is optional.
By default, the quadrature rule is chosen so that a mass matrix
for the chosen finite element basis can be integrated exactly.


.. note::

   Integrals over the domain are assembled using :class:`~skfem.assembly.InteriorBasis`.
   In order to assemble weak forms defined on the
   boundary of the domain use :class:`~skfem.assembly.FacetBasis`, see e.g.
   :ref:`integralcondition`.

Assembling finite element matrices
##################################

Forms are defined using the decorators
:class:`~skfem.assembly.BilinearForm` and :class:`~skfem.assembly.LinearForm`.
For example, the mass matrix is assembled as follows:

.. code-block:: python

   In [6]: from skfem import BilinearForm, LinearForm, asm
   In [7]: @BilinearForm
      ...: def mass(u, v, w):
      ...:     return u * v
      ...:
   In [8]: asm(mass, basis)
   Out[8]: """<289x289 sparse matrix of type '<class 'numpy.float64'>'
           with 3073 stored elements in Compressed Sparse Row format>"""


.. note::

   In the decorated function, ``u`` refers to the solution, ``v`` refers to the
   test function, and ``w`` contains additional fields such as the global
   coordinates (``w.x``) and the local mesh parameters (``w.h``).

The discrete laplacian can be defined as follows:

.. code-block:: python

   In [9]: @BilinearForm
      ...: def laplacian(u, v, w):
      ...:     from skfem.helpers import d, dot
      ...:     return dot(d(u), d(v))
      ...:


.. note::
   In reality,
   ``u`` and ``v`` are tuples that store the values of the basis functions (and
   the values of the derivatives) at quadrature points. In order to access the
   derivatives, you can write ``u[1]`` (values are ``u[0]``, first derivatives
   are ``u[1]``, and so on.) or, equivalently, use helper functions from
   the module ``skfem.helpers`` as above.

A load vector corresponding to the linear form :math:`F(v)=\int_\Omega x^2 v
\,\mathrm{d}x` is assembled similarly:

.. code-block:: python

   In [10]: @LinearForm
      ...: def F(v, w):
      ...:     return w.x[0] ** 2 * v
      ...:
   In [11]: asm(F, basis)
   Out[12]: array([-1.35633681e-06,  9.22309028e-05, -5.42534722e-06,  ...])

See :ref:`learning` for more use cases and instructions.

Setting essential boundary conditions
#####################################

.. warning::

   Using the assembled matrices requires basic understanding of
   the finite element method. In particular, to understand
   how the boundary conditions are set, you should be familiar
   with the concepts of Dirichlet and Neumann boundary conditions
   and how they are reflected in the finite element matrices.

The simplest way of obtaining degrees-of-freedom corresponding to a specific
boundary is through :meth:`skfem.assembly.Basis.get_dofs`.

.. code-block:: python

   In [1]: from skfem import MeshTri, ElementTriP2, InteriorBasis
   In [2]: basis = InteriorBasis(MeshTri.init_lshaped(), ElementTriP2())
   In [3]: basis.get_dofs(lambda x: x[0] == 0.0)
   Out[3]: Dofs(nodal={'u': array([0, 2, 4])}, facet={'u': array([ 9, 11])}, edge={}, interior={})

The result value is :class:`skfem.assembly.Dofs` object (a named tuple)
containing the degree-of-freedom numbers corresponding to :math:`x=0`.

In particular, the result tells us that when assembling matrices and vectors
using :class:`~skfem.assembly.Basis` object, the rows 0, 2 and 4 correspond to
the degrees-of-freedom at the vertices of the elements on the boundary
:math:`x=0`, and the rows 9 and 11 correspond to the degrees-of-freedom at the
facets of the elements on the boundary :math:`x=0`.

.. code-block:: python

   In [4]: from skfem.models.poisson import laplace, unit_load
   In [5]: A, b = asm(laplace, basis), asm(unit_load, basis)
   In [6]: A
   Out[6]: """<21x21 sparse matrix of type '<class 'numpy.float64'>'
           with 165 stored elements in Compressed Sparse Row format>"""

   In [7]: b
   Out[7]:
   array([3.12250226e-16, 8.06646416e-17, 8.06646416e-17, 1.61329283e-16,
          1.61329283e-16, 1.61329283e-16, 1.61329283e-16, 1.61329283e-16,
          1.66666667e-01, 1.66666667e-01, 3.33333333e-01, 3.33333333e-01,
          3.33333333e-01, 3.33333333e-01, 3.33333333e-01, 1.66666667e-01,
          1.66666667e-01, 1.66666667e-01, 1.66666667e-01, 1.66666667e-01,
          1.66666667e-01])

The corresponding degrees-of-freedom can be eliminated from the linear
system, e.g., with the help of :func:`skfem.utils.condense`.

.. code-block:: python

   In [10]: condense(A, b, D=basis.get_dofs(lambda x: x[0]==0.0))
   Out[10]:
   ("""<16x16 sparse matrix of type '<class 'numpy.float64'>'
    with 86 stored elements in Compressed Sparse Row format>""",
    array([8.06646416e-17, 1.61329283e-16, 1.61329283e-16, 1.61329283e-16,
           1.61329283e-16, 1.66666667e-01, 3.33333333e-01, 3.33333333e-01,
           3.33333333e-01, 3.33333333e-01, 1.66666667e-01, 1.66666667e-01,
           1.66666667e-01, 1.66666667e-01, 1.66666667e-01, 1.66666667e-01]))

The previous commands cause the boundary degrees-of-freedom to be zero.
In order to set them to prescribed values, you can experiment with the
different keyword arguments of :func:`skfem.utils.condense`; see e.g.
:ref:`inhomo`.

Solving linear systems
######################

The assembly routines output either scipy sparse matrices or numpy arrays.
These can be used, e.g., together with any scipy linear algebra routines.  For
convenience, we have wrapped some of the most commonly used scipy functions into
:func:`skfem.utils.solve`.

.. code-block:: python

   In [1]: from skfem import *
   In [2]: m = MeshTri()
   In [3]: m.refine(3)
   In [4]: basis = InteriorBasis(m, ElementTriP2())
   In [5]: from skfem.models.poisson import laplace, unit_load
   In [6]: A = asm(laplace, basis)
   In [7]: b = asm(unit_load, basis)
   In [8]: x = solve(*condense(A, b, D=basis.get_dofs()))
   In [9]: x.max()
   Out[9]: 0.07367588634940822

By default, :func:`skfem.utils.solve` uses :func:`scipy.sparse.linalg.spsolve`.

Postprocessing the results
##########################

We can now visualize the solution from the previous section using
matplotlib:

.. code-block:: python

   In [10]: from skfem.visuals.matplotlib import draw, plot, savefig
   In [11]: ax = draw(m)   
   In [12]: plot(basis, x, Nrefs=3, shading='gouraud', ax=ax)
   In [13]: savefig('tutorial_solution.png')

For other examples on postprocessing see, e.g., :ref:`tetrapoisson` for saving
the solution to VTK, :ref:`postprocess` and :ref:`laplacemixed` for evaluating
functionals, or :ref:`adaptivepoisson` for evaluating error estimators.
