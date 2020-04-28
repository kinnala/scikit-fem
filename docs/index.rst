The documentation of scikit-fem
=====

`scikit-fem <https://github.com/kinnala/scikit-fem>`_ is a lightweight Python 3.6+
library for performing `finite element assembly
<https://en.wikipedia.org/wiki/Finite_element_method>`_. Its main purpose is
the transformation of bilinear forms into sparse matrices and linear forms into
vectors.  The library supports triangular, quadrilateral, tetrahedral and
hexahedral meshes as well as one-dimensional problems.

.. note::

    Installing the library is as simple as running

    .. code-block:: bash

        pip install scikit-fem

    Full examples can be found `in the source code distribution <https://github.com/kinnala/scikit-fem/tree/master/docs/examples>`_.
  
A brief overview of the package
=====

The most important modules of the library are :mod:`skfem.mesh` and
:mod:`skfem.element`. For finite element assembly, you also need either an
:class:`~skfem.assembly.InteriorBasis` or a :class:`~skfem.assembly.FacetBasis`
object, and a form defined using one of the following decorators:
:class:`~skfem.assembly.BilinearForm`, :class:`~skfem.assembly.LinearForm`, or
:class:`~skfem.assembly.Functional`.


Module: skfem.mesh
-----

.. automodule:: skfem.mesh

Module: skfem.element
-----

.. automodule:: skfem.element

Module: skfem.assembly
-----

.. automodule:: skfem.assembly
