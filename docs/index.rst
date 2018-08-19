The documentation of scikit-fem
===============================

`scikit-fem <https://github.com/kinnala/scikit-fem>`_ is a lightweight Python 3.6
library for performing `finite element assembly
<https://en.wikipedia.org/wiki/Finite_element_method>`_. Its main purpose is
the transformation of bilinear forms into sparse matrices and linear forms into
vectors.  The library supports triangular, quadrilateral, tetrahedral and
hexahedral meshes as well as one-dimensional problems.

Moreover, the library supports
the implementation of various different :math:`h`-type finite element methods,
and has built-in elements for the usual :math:`H^1`-, :math:`H^2`-,
:math:`H(\text{div})`- and :math:`H(\text{curl})`-conforming problems. A major
portion of the features are available for users that have only SciPy (and its
dependencies) installed. The library contains no compiled code and is licensed
with the permissive 3-clause BSD license.

This document contains the efforts to improve the user documentation of the
project. Currently, the main learning resource consists of the extended
examples that are continuously improved. In case you have any further
questions, do not hesitate to drop in and say hello at our `Gitter chat
<https://gitter.im/scikit-fem>`_.

.. toctree::
    :maxdepth: 2

    gettingstarted
    examples
    api
