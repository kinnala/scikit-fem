.. _tutorial:

Tutorial
--------

This is a walkthrough of the basic features of scikit-fem.

Creating meshes
###############

The default constructors of :class:`~skfem.mesh.Mesh`
classes create simple meshes of unit intervals :math:`\Omega = [0,1]^d`.

.. code-block:: python

   In [1]: from skfem.mesh import MeshTri
   In [2]: m = MeshTri()
   In [3]: m
   Out[3]:
   "Triangular mesh with 4 vertices and 2 elements."

There are also a few additional constructors available such as
:meth:`skfem.mesh.MeshTri.init_tensor` and
:meth:`skfem.mesh.MeshTri.init_lshaped`. More importantly, all
mesh types can be loaded from file formats supported by meshio:

.. code-block:: python

   In [4]: from skfem.mesh import Mesh
   In [5]: Mesh.load("docs/examples/square.msh")
   Out[5]:
   "Triangular mesh with 109 vertices and 184 elements."

Meshes can be visualized using :meth:`skfem.mesh.MeshTri.draw`.

Choosing basis functions
########################

The local basis functions are defined in :class:`~skfem.element.Element`
classes. They are combined with meshes to create
:class:`~skfem.assembly.GlobalBasis` objects, such as
:class:`~skfem.assembly.InteriorBasis` and :class:`~skfem.assembly.FacetBasis`,
that contain global basis functions evaluated at global quadrature points:

.. code-block:: python

   In [1]: from skfem import MeshTri, ElementTriP2, InteriorBasis
   In [2]: m = MeshTri()
   In [3]: m.refine(3)
   In [4]: basis = InteriorBasis(m, ElementTriP2())

Here :class:`~skfem.element.ElementTriP2` refers to the quadratic Lagrange
finite element for triangular meshes.  You can control the integration order via
a keyword argument:

.. code-block:: python

   In [5]: basis = InteriorBasis(m, ElementTriP2(), intorder=5)

Now polynomials of order 5 can be integrated exactly by the quadrature
rule. By default, the order of the rule is chosen so that a mass matrix
for the chosen finite element basis can be integrated exactly.

Assembling finite element matrices
##################################

Assembly requires defining forms with the decorators
:func:`~skfem.assembly.bilinear_form` and :func:`~skfem.assembly.linear_form`.
For example, the mass matrix can be assembled as follows:

.. code-block:: python

   In [6]: from skfem import bilinear_form, asm
   In [7]: @bilinear_form
      ...: def mass(u, du, v, dv, w):
      ...:     return u * v
      ...:
   In [8]: asm(mass, basis)
   Out[8]:
   """<289x289 sparse matrix of type '<class 'numpy.float64'>'
   with 3073 stored elements in Compressed Sparse Row format>"""

In the definition of the form ``mass``, ``u`` refers to the solution values and
``du`` refers to its derivatives, ``v`` and ``dv`` refer to the test function
values and derivatives, and ``w`` contains any additional variables such as the
global coordinates (``w.x``) and the local mesh parameters (``w.h``).

The number of arguments required in the form definition depends on the type of
element.  In particular, the number of positional arguments in the bilinear form
definition should be exactly ``2 * len(Element.order) + 1``.

A load vector corresponding to the linear form :math:`F(v)=\int_\Omega x^2 v
\,\mathrm{d}x` is assembled similarly:

.. code-block:: python

   In [9]: @linear_form
      ...: def F(v, dv, w):
      ...:     return w.x[0] ** 2 * v
      ...:
   In [10]: asm(F, basis)
   Out[11]: array([-1.35633681e-06,  9.22309028e-05, -5.42534722e-06,  ...])

Setting essential boundary conditions
#####################################

The simplest way of obtaining degrees-of-freedom corresponding to a specific
boundary is through :meth:`~skfem.assembly.GlobalBasis.get_dofs`.

.. code-block:: python

   In [1]: from skfem import MeshTri, ElementTriP2, InteriorBasis
   In [2]: basis = InteriorBasis(MeshTri.init_lshaped(), ElementTriP2())
   In [3]: basis.get_dofs(lambda x: x[0]==0.0)
   Out[3]: Dofs(nodal={'u': array([0, 2, 4])}, facet={'u': array([ 9, 11])}, edge={}, interior={})

This result tells us that when assembling matrices and vectors using ``basis``
object, the rows 0, 2 and 4 correspond to the degrees-of-freedom at the vertices
of the elements on the boundary :math:`x=0`, and the rows 9 and 11 correspond to
the degrees-of-freedom at the facets of the elements on the boundary :math:`x=0`.

.. code-block:: python

   In [4]: from skfem.models.poisson import laplace, unit_load
   In [5]: A, b = asm(laplace, basis), asm(unit_load, basis)
   In [6]: A
   Out[6]:
   """<21x21 sparse matrix of type '<class 'numpy.float64'>'
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

   In [10]: condense(A, b, D=basis.get_dofs(lambda x: x[0]==0.0).all())
   Out[10]:
   ("""<16x16 sparse matrix of type '<class 'numpy.float64'>'
    with 86 stored elements in Compressed Sparse Row format>""",
    array([8.06646416e-17, 1.61329283e-16, 1.61329283e-16, 1.61329283e-16,
           1.61329283e-16, 1.66666667e-01, 3.33333333e-01, 3.33333333e-01,
           3.33333333e-01, 3.33333333e-01, 1.66666667e-01, 1.66666667e-01,
           1.66666667e-01, 1.66666667e-01, 1.66666667e-01, 1.66666667e-01]))

The previous commands cause the corresponding degrees-of-freedom to be zero.
In order to set them to prescribed values, you can experiment with the
different keyword arguments of :func:`skfem.utils.condense`; c.f. the
examples.

Solving linear systems
######################

Postprocessing the results
##########################
