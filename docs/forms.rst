.. _forms:

==================
 Anatomy of forms
==================

We consider forms as the basic building blocks of finite element assembly.
Thus, it is important to understand how forms are used in scikit-fem and how to
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
=========================

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
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
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
====================================

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
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !u[0]
   array([[0.66666667, 0.16666667, 0.16666667],
          [0.66666667, 0.16666667, 0.16666667]])


.. _predefined:

Use of predefined functions in the forms
========================================

Sometimes we use a previous solution vector in the form
definition, e.g., when solving nonlinear problems.
A simple fixed-point iteration for

.. math::

   \begin{aligned}
      -\nabla \cdot ((u + 1)\nabla u) &= 1 \quad \text{in $\Omega$}, \\
      u &= 0 \quad \text{on $\partial \Omega$},
   \end{aligned}

would correspond to repeatedly
finding :math:`u_{k+1} \in H^1_0(\Omega)` which satisfies

.. math::

   \int_\Omega (u_{k} + 1) \nabla u_{k+1} \cdot \nabla v \,\mathrm{d}x = \int_\Omega v\,\mathrm{d}x

for every :math:`v \in H^1_0(\Omega)`.
The argument ``w`` is used to define such forms:

.. doctest::

   >>> from skfem import *
   >>> from skfem.models.poisson import unit_load
   >>> from skfem.helpers import grad, dot
   >>> @BilinearForm
   ... def bilinf(u, v, w):
   ...     return (w.u_k + 1.) * dot(grad(u), grad(v))

The previous solution :math:`u_k` must be provided to
:func:`~skfem.assembly.asm` as a keyword argument:

.. doctest::

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

Inside the form definition, ``w`` is a dictionary of user provided arguments and
additional default keys:

.. code-block:: none

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, InteriorBasis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !w.keys()
   dict_keys(['x', 'h'])

By default, ``w['x']`` (available also as ``w.x``) corresponds to the global
coordinates and ``w['h']`` (available also as ``w.h``) corresponds to the local
mesh parameter.
