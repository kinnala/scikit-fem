.. _forms:

==================
 Anatomy of forms
==================

We consider forms as the basic building blocks of finite element assembly.
Thus, it is important to understand how forms are used in scikit-fem and how to
express them correctly.

Let us begin with an example.  The bilinear form corresponding to the Laplace
operator :math:`-\Delta` is

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x.

In order to express this in scikit-fem we pick the integrand and write it as a
Python function:

.. code-block:: python

   from skfem import *
   from skfem.helpers import grad, dot

   @BilinearForm
   def integrand(u, v, w):
       return dot(grad(u), grad(v))

Forms return NumPy arrays
=========================

The form definition should always return a two-dimensional NumPy array.  This
can be verified by using the Python debugger:

.. code-block:: python

   from skfem import *
   from skfem.helpers import grad, dot

   @BilinearForm
   def integrand(u, v, w):
       import pdb; pdb.set_trace()  # breakpoint
       return dot(grad(u), grad(v))

Now saving the above snippet as ``test.py`` and running it via ``python -i
test.py`` allows experimenting with the form:

.. code-block::

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) dot(grad(u), grad(v))
   array([[2., 2., 2.],
          [1., 1., 1.]])

Notice how ``dot(grad(u), grad(v))`` evaluates to a NumPy array with the shape
`number of elements` x `number of quadrature points per element`.  The return
value should always have such shape no matter which mesh or element type is
used.

Helpers are useful but not necessary
====================================

The module ``skfem.helpers`` contains functions that make the forms more
readable.  An alternative way to write the above form is

.. code-block:: python

   @BilinearForm
   def integrand(u, v, w):
       return u[1][0] * v[1][0] + u[1][1] * v[1][1]

In fact, ``u`` and ``v`` are more or less tuples of NumPy arrays
with the values of the function at ``u[0]`` and the values
of the gradient at ``u[1]`` (and some additional magic such as
implementing ``__array__`` and ``__mul__``
so that expressions such as ``u * v`` work as expected).

Notice how the shape of ``u[0]`` is what we expect also from the return value:

.. code-block::

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !u[0]
   array([[0.66666667, 0.16666667, 0.16666667],
          [0.66666667, 0.16666667, 0.16666667]])


Use of predefined functions in the forms
========================================

It is sometimes necessary to use a previous solution vector in the form
definition, e.g., when solving nonlinear problems.
A simple fixed-point iteration for solving the nonlinear boundary
value problem

.. math::

   \begin{aligned}
      -\nabla \cdot ((u + 1)\nabla u) &= 1 \quad \text{in $\Omega$} \\
      u &= 0 \quad \text{on $\partial \Omega$}
   \end{aligned}

would correspond to repeatedly
finding :math:`u_{k+1} \in H^1_0(\Omega)` which satisfies

.. math::

   \int_\Omega (u_{k} + 1) \nabla u_{k+1} \cdot \nabla v \,\mathrm{d}x = \int_\Omega v\,\mathrm{d}x

for every :math:`v \in H^1_0(\Omega)`.
Defining such forms requires the use of the argument ``w``:

.. code-block:: python

   >>> from skfem import *
   >>> from skfem.models.poisson import unit_load
   >>> from skfem.helpers import grad, dot
   >>> @BilinearForm
   ... def bilinf(u, v, w):
   ...     return (w.u_k + 1.) * dot(grad(u), grad(v))

When performing the fixed point iteration, we provide a previous
solution to :func:`skfem.assembly.asm` as a keyword argument:

.. code-block:: python

   >>> m = MeshTri()
   >>> m.refine(3)
   >>> basis = InteriorBasis(m, ElementTriP1())
   >>> b = asm(unit_load, basis)
   >>> x = 0. * b.copy()
   >>> for itr in range(10):  # fixed point iteration
   ...     A = asm(bilinf, basis, u_k=basis.interpolate(x))
   ...     x = solve(*condense(A, b, I=m.interior_nodes()))
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

In the form definition, ``w`` is actually a dictionary with
the user provided arguments and additional precomputed keys:

.. code-block::

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !w.keys()
   dict_keys(['x', 'h'])

By default, ``w.x`` corresponds to the global coordinates and ``w.h``
corresponds to the local mesh parameter.
