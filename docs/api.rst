Modules
=======

This section contains API documentation for the most important modules and
interfaces that are necessary when using the library.  Equivalently, one can
access the documentation through the standard Python help system, i.e. using
:func:`help`.

skfem.mesh
----------

.. automodule:: skfem.mesh

MeshTri
~~~~~~~

.. autoclass:: skfem.mesh.MeshTri

Different constructors
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.__init__

.. automethod:: skfem.mesh.MeshTri.load

.. automethod:: skfem.mesh.MeshTri.init_symmetric

.. automethod:: skfem.mesh.MeshTri.init_sqsymmetric

.. automethod:: skfem.mesh.MeshTri.init_refdom

.. automethod:: skfem.mesh.MeshTri.init_tensor

Methods for visualisation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.draw

.. automethod:: skfem.mesh.MeshTri.plot

.. automethod:: skfem.mesh.MeshTri.plot3

MeshQuad
~~~~~~~~

.. autoclass:: skfem.mesh.MeshQuad

MeshTet
~~~~~~~

.. autoclass:: skfem.mesh.MeshTet

MeshHex
~~~~~~~

.. autoclass:: skfem.mesh.MeshHex

MeshLine
~~~~~~~~

.. autoclass:: skfem.mesh.MeshLine

skfem.assembly
--------------

.. automodule:: skfem.assembly

Defining a global basis
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.assembly.GlobalBasis

.. autoclass:: skfem.assembly.InteriorBasis

.. autoclass:: skfem.assembly.FacetBasis

Assembling matrices
~~~~~~~~~~~~~~~~~~~

.. autofunction:: skfem.assembly.bilinear_form

.. autofunction:: skfem.assembly.linear_form

.. autofunction:: skfem.assembly.asm

skfem.utils
-----------

.. automodule:: skfem.utils

.. autofunction:: skfem.utils.solve

.. autofunction:: skfem.utils.condense
