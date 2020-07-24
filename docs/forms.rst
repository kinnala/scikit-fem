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

In order to express this in scikit-fem we pick the integrand,

.. math::

   a(u, v) = \int_\Omega \underbrace{\nabla u \cdot \nabla v}_{~} \,\mathrm{d}x,

and write it as a Python function:

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

In fact, ``u`` and ``v`` are simply tuples of NumPy arrays
with the values of the function at ``u[0]`` and the values
of the gradient at ``u[1]``.
In addition, they implement ``__array__`` and ``__mul__``
so that expressions such as ``u * v`` work as expected.

Notice how the shape of ``u[0]`` is what we expect also from the return value:

.. code-block::

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !u[0]
   array([[0.66666667, 0.16666667, 0.16666667],
          [0.66666667, 0.16666667, 0.16666667]])

