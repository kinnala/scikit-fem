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
   Out[3]: # Triangular mesh with 4 vertices and 2 elements.

There are also a few additional constructors available such as
:meth:`skfem.mesh.MeshTri.init_tensor` and
:meth:`skfem.mesh.MeshTri.init_lshaped`. More importantly, all
mesh types can be loaded from file formats supported by meshio:

.. code-block:: python

   In [4]: from skfem.mesh import Mesh
   In [5]: Mesh.load("docs/examples/square.msh")
   Out[5]: # Triangular mesh with 109 vertices and 184 elements.

You can also visualize meshes and solutions via
:meth:`skfem.mesh.MeshTri.draw` and :meth:`skfem.mesh.MeshTri.plot`.

Choosing basis functions
########################

The local basis functions are defined in :class:`~skfem.element.Element`
classes. They are combined with meshes to create
:class:`~skfem.assembly.GlobalBasis` objects such as
:class:`~skfem.assembly.InteriorBasis` and :class:`~skfem.assembly.FacetBasis`
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
for the given finite element basis can be integrated exactly.

Assembling finite element matrices
##################################

Assembly requires defining forms using the decorators
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
   # <289x289 sparse matrix of type '<class 'numpy.float64'>'
   # with 3073 stored elements in Compressed Sparse Row format>"""

A load vector corresponding to the linear form :math:`F(v)=\int_\Omega x^2 v
\,\mathrm{d}x` can be assembled as follows:

.. code-block:: python

   In [9]: @linear_form
      ...: def F(v, dv, w):
      ...:     return w.x[0] ** 2 * v
      ...:
   In [10]: asm(F, basis)
   Out[13]: 
   # array([-1.35633681e-06,  9.22309028e-05, -5.42534722e-06,  ...])

Setting boundary conditions
###########################

Solving linear systems
######################

Postprocessing the results
##########################
