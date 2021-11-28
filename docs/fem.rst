.. _fem:

=================================
 Introduction to finite elements
=================================

This is a brief introduction to the finite element method.  It assumes you
have a basic understanding of differential equations and linear algebra.
The purpose of the finite element method is to transform partial
differential equations into linear systems, e.g.,

.. math::

   -\Delta u = f

defined in the domain :math:`\Omega \subset \mathbb{R}^2`, is approximated by
the linear system

.. math::

   A\boldsymbol{x} = \boldsymbol{b}.

In the finite element method, :math:`A` and :math:`\boldsymbol{b}` are created
with the help of a mesh and a basis.  Any domain can be used as long as it can
be meshed.

.. note::

   In scikit-fem, the most commonly used meshes are
   :class:`~skfem.mesh.MeshLine1`, :class:`~skfem.mesh.MeshTri1`,
   :class:`~skfem.mesh.MeshQuad1`, :class:`~skfem.mesh.MeshTet1`,
   :class:`~skfem.mesh.MeshHex1`.

.. figure:: https://user-images.githubusercontent.com/973268/143777838-6bc0e22f-4aa0-4a7a-a22a-a1264d6acaa4.png
   :align: center
   :width: 320px

   Example of a triangular mesh on a polygonal domain.  A circle cannot be
   meshed exactly using triangles although this mesh approximates a circle.

Suppose we have defined a set of basis functions :math:`\{\varphi_j\}_{j=1}^N`.
Then the linear combination

.. math::

   u \approx \sum_{j=1}^N u_j \varphi_j

with the degrees-of-freedom (DOFs) :math:`u_j` leads to

.. math::

   \boldsymbol{x}_i = u_i, \quad
   \boldsymbol{b}_i = \int_\Omega f\varphi_i\,\mathrm{d}x, \quad
   A_{ij} = \int_\Omega \nabla \varphi_j \cdot \nabla \varphi_i\,\mathrm{d}x.

These integrals can be computed numerically after the basis has been fixed.

.. note::

   In scikit-fem, the integrals (or the forms) are defined using
   :class:`~skfem.assembly.BilinearForm` and
   :class:`~skfem.assembly.LinearForm`.

The expression for :math:`A_{ij}` is obtained by multiplying the governing
equation :math:`-\Delta u = f` by a test function :math:`v`, integrating
both sides over the domain :math:`\Omega`, using the Green's identity

.. math::

   -\int_\Omega \Delta u v \,\mathrm{d}x = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x - \int_{\partial \Omega} (\nabla u \cdot \boldsymbol{n})v\,\mathrm{d}s

and applying the boundary condition :math:`u|_{\partial \Omega} = 0`.  In
this case the test function :math:`v` satisfies the same boundary condition
as :math:`u`.  The remaining equation

.. math::

   \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x = \int_\Omega fv\,\mathrm{d}x

is known as the weak formulation because it has less derivatives acting on
the solution :math:`u`.  The linear system follows after substituting

.. math::

   u \approx \sum_{j=1}^N u_j \varphi_j

and replacing :math:`v` by :math:`\varphi_i`.


The basis :math:`\{\varphi_j\}_{j=1}^N` is defined with the help
of a mesh.  Let :math:`N` correspond to the number of vertices in a triangular
mesh.  We give each vertex an index and define the basis
function :math:`\varphi_j` so that it attains the value one at
vertex :math:`j` and the value zero at all the other vertices.  Inside
any triangle :math:`\varphi_j` is linear.  The resulting basis
functions are called the hat functions because of their shape.

.. figure:: https://user-images.githubusercontent.com/973268/143777834-7f91fdf6-83c7-4201-a5b1-85023d7effa2.png
   :align: center
   :width: 400px

   An example of a hat basis function associated with the fifth vertex.

As a consequence, :math:`u_j` is the coefficient of the hat function associated
with vertex :math:`j` and the linear combination

.. math::

   u \approx \sum_{j=1}^N u_j \varphi_j

represents any piecewise linear function.  The coefficients :math:`u_j` are
obtained by solving the linear system :math:`A\boldsymbol{x} = \boldsymbol{b}`.


.. note::

   The main purpose of scikit-fem is the creation of such linear systems given
   a mesh and the forms for :math:`A_{ij}` and :math:`\boldsymbol{b}_i`.  The
   finite element basis is initialized using :class:`~skfem.assembly.Basis`
   with the help of the mesh and :class:`~skfem.element.ElementTriP1` which
   represents the piecewise linear basis discussed above.

.. figure:: https://user-images.githubusercontent.com/973268/143787130-7137927d-8e47-46c0-9149-8fa2497c09ca.png
   :align: center
   :width: 400px

   The solution to the above example with :math:`f=1` and the boundary
   condition :math:`u|_{\partial \Omega} = 0`.

Before we mentioned the boundary condition :math:`u|_{\partial \Omega} = 0` but
did not really discuss how it is implemented in practice.  A common approach is
to first create the linear system :math:`A\boldsymbol{x} = \boldsymbol{b}` for
all vertices and then eliminate boundary DOFs by reordering the rows and the
columns of the linear system:

.. math::

   \begin{bmatrix}
   A_{II} & A_{ID} \\
   A_{DI} & A_{DD}
   \end{bmatrix}
   \begin{bmatrix}
   \boldsymbol{x}_I \\
   \boldsymbol{x}_D
   \end{bmatrix}
   =
   \begin{bmatrix}
   \boldsymbol{b}_I \\
   \boldsymbol{b}_D
   \end{bmatrix}.

Since we already know the values of the boundary DOFs,
i.e. :math:`\boldsymbol{x}_D`, this simplifies to

.. math::

   A_{II} \boldsymbol{x}_I = \boldsymbol{b}_I - A_{ID} \boldsymbol{x}_D.

For identically zero boundary DOFs this reads

.. math::

   A_{II} \boldsymbol{x}_I = \boldsymbol{b}_I

and we solve only the interior DOFs, i.e. :math:`\boldsymbol{x}_I`.

.. note::

   There are various boundary conditions one can study using the finite element
   method.  This is example is known as the "essential boundary condition"
   because it requires modifying the linear system.  Setting the values of the
   unknown function on the boundary is also known as the Dirichlet boundary
   condition.
