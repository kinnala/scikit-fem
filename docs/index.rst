Introduction
============

`scikit-fem <https://github.com/kinnala/scikit-fem>`_ is a lightweight Python
library for performing `finite element assembly
<https://en.wikipedia.org/wiki/Finite_element_method>`_. Its main purpose is
the transformation of bilinear forms into sparse matrices and linear forms into
vectors.

Features
========

The library supports triangular, quadrilateral, tetrahedral and
hexahedral meshes as well as one-dimensional problems. Moreover, it supports
the implementation of various different :math:`h`-type finite element methods,
and has built-in elements for the usual :math:`H^1`-, :math:`H^2`-,
:math:`H(\text{div})`- and :math:`H(\text{curl})`-conforming problems. A major
portion of the features are available for users that have only SciPy (and its
dependencies) installed. The library contains no compiled code and is licensed
with the permissive 3-clause BSD license.


Installation
============

The latest release can be installed from PyPI:

.. code-block:: bash

    pip install scikit-fem

For more recent features, you can clone the project's Github repository:

.. code-block:: bash

    git clone https://github.com/kinnala/scikit-fem

Examples
========

.. include:: ../examples/ex01.py
    :start-after: """
    :end-before: """

.. literalinclude:: ../examples/ex01.py
    :end-before: """




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
