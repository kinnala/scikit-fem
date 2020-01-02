Modules
=======

This section contains API documentation for the most important modules and
interfaces that are necessary when using the library.  Equivalently, one can
access the documentation through the standard Python help system, i.e. using
:func:`help`.

skfem.mesh
----------

.. automodule:: skfem.mesh

.. automethod:: skfem.mesh.Mesh.save

MeshTri
~~~~~~~

.. autoclass:: skfem.mesh.MeshTri

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.__init__

.. automethod:: skfem.mesh.MeshTri.load

.. automethod:: skfem.mesh.MeshTri.init_symmetric

.. automethod:: skfem.mesh.MeshTri.init_sqsymmetric

.. automethod:: skfem.mesh.MeshTri.init_refdom

.. automethod:: skfem.mesh.MeshTri.init_tensor

.. automethod:: skfem.mesh.MeshTri.init_lshaped


MeshQuad
~~~~~~~~

.. autoclass:: skfem.mesh.MeshQuad

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshQuad.__init__

.. automethod:: skfem.mesh.MeshQuad.load

.. automethod:: skfem.mesh.MeshQuad.init_refdom

.. automethod:: skfem.mesh.MeshQuad.init_tensor


MeshTet
~~~~~~~

.. autoclass:: skfem.mesh.MeshTet

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTet.__init__

.. automethod:: skfem.mesh.MeshTet.load


MeshHex
~~~~~~~

.. autoclass:: skfem.mesh.MeshHex

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshHex.__init__

.. automethod:: skfem.mesh.MeshHex.init_tensor


MeshLine
~~~~~~~~

.. autoclass:: skfem.mesh.MeshLine


skfem.mapping
-------------

.. automodule:: skfem.mapping

.. autoclass:: skfem.mapping.MappingAffine

.. autoclass:: skfem.mapping.MappingIsoparametric

skfem.element
-------------

.. automodule:: skfem.element

.. automodule:: skfem.element.Element

Triangular elements
~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.element.ElementTriP1

.. autoclass:: skfem.element.ElementTriP2

.. autoclass:: skfem.element.ElementTriP0

.. autoclass:: skfem.element.ElementTriArgyris

.. autoclass:: skfem.element.ElementTriMorley

.. autoclass:: skfem.element.ElementTriDG

.. autoclass:: skfem.element.ElementTriRT0

Quadrilateral elements
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.element.ElementQuad1

.. autoclass:: skfem.element.ElementQuad2

.. autoclass:: skfem.element.ElementQuad0

Tetrahedral elements
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.element.ElementTetP1

.. autoclass:: skfem.element.ElementTetP2

.. autoclass:: skfem.element.ElementTetP0

.. autoclass:: skfem.element.ElementTetRT0

.. autoclass:: skfem.element.ElementTetN0

Hexahedral elements
~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.element.ElementHex1

1D elements
~~~~~~~~~~~

.. autoclass:: skfem.element.ElementLineP1

.. autoclass:: skfem.element.ElementLineP2

Other
~~~~~

.. autoclass:: skfem.element.ElementVectorH1

skfem.assembly
--------------

.. automodule:: skfem.assembly

Defining a global basis
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: skfem.assembly.Basis

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

skfem.visuals
-------------

.. automodule:: skfem.visuals.matplotlib

.. autofunction:: skfem.visuals.matplotlib.draw

.. autofunction:: skfem.visuals.matplotlib.plot

.. autofunction:: skfem.visuals.matplotlib.plot3
