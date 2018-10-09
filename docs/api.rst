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

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.__init__

.. automethod:: skfem.mesh.MeshTri.load

.. automethod:: skfem.mesh.MeshTri.init_symmetric

.. automethod:: skfem.mesh.MeshTri.init_sqsymmetric

.. automethod:: skfem.mesh.MeshTri.init_refdom

.. automethod:: skfem.mesh.MeshTri.init_tensor

Modify
^^^^^^

.. automethod:: skfem.mesh.MeshTri.refine

.. automethod:: skfem.mesh.MeshTri.remove_elements

.. automethod:: skfem.mesh.MeshTri.scale

.. automethod:: skfem.mesh.MeshTri.translate

.. automethod:: skfem.mesh.MeshTri.mirror

Explore
^^^^^^^

.. automethod:: skfem.mesh.MeshTri.nodes_satisfying

.. automethod:: skfem.mesh.MeshTri.facets_satisfying

.. automethod:: skfem.mesh.MeshTri.elements_satisfying

.. automethod:: skfem.mesh.MeshTri.interior_nodes

.. automethod:: skfem.mesh.MeshTri.boundary_nodes

.. automethod:: skfem.mesh.MeshTri.interior_facets

.. automethod:: skfem.mesh.MeshTri.boundary_facets

Visualise
^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.save

.. automethod:: skfem.mesh.MeshTri.draw

.. automethod:: skfem.mesh.MeshTri.plot

.. automethod:: skfem.mesh.MeshTri.plot3

Other methods
^^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTri.submesh

MeshQuad
~~~~~~~~

.. autoclass:: skfem.mesh.MeshQuad

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshQuad.__init__

.. automethod:: skfem.mesh.MeshQuad.load

.. automethod:: skfem.mesh.MeshQuad.init_refdom

.. automethod:: skfem.mesh.MeshQuad.init_tensor

Modify
^^^^^^

.. automethod:: skfem.mesh.MeshQuad.refine

.. automethod:: skfem.mesh.MeshQuad.remove_elements

.. automethod:: skfem.mesh.MeshQuad.scale

.. automethod:: skfem.mesh.MeshQuad.translate

.. automethod:: skfem.mesh.MeshQuad.mirror

Explore
^^^^^^^

.. automethod:: skfem.mesh.MeshQuad.nodes_satisfying

.. automethod:: skfem.mesh.MeshQuad.facets_satisfying

.. automethod:: skfem.mesh.MeshQuad.elements_satisfying

.. automethod:: skfem.mesh.MeshQuad.interior_nodes

.. automethod:: skfem.mesh.MeshQuad.boundary_nodes

.. automethod:: skfem.mesh.MeshQuad.interior_facets

.. automethod:: skfem.mesh.MeshQuad.boundary_facets

Visualise
^^^^^^^^^

.. automethod:: skfem.mesh.MeshQuad.draw

.. automethod:: skfem.mesh.MeshQuad.plot

.. automethod:: skfem.mesh.MeshQuad.plot3

MeshTet
~~~~~~~

.. autoclass:: skfem.mesh.MeshTet

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTet.__init__

.. automethod:: skfem.mesh.MeshTet.load

Modify
^^^^^^
.. automethod:: skfem.mesh.MeshTet.refine

.. automethod:: skfem.mesh.MeshTet.remove_elements

.. automethod:: skfem.mesh.MeshTet.scale

.. automethod:: skfem.mesh.MeshTet.translate

Visualisation
^^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshTet.save

MeshHex
~~~~~~~

.. autoclass:: skfem.mesh.MeshHex

Constructors
^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshHex.__init__

.. automethod:: skfem.mesh.MeshHex.init_tensor

Modify
^^^^^^

.. automethod:: skfem.mesh.MeshHex.refine

.. automethod:: skfem.mesh.MeshHex.remove_elements

.. automethod:: skfem.mesh.MeshHex.scale

.. automethod:: skfem.mesh.MeshHex.translate

Visualisation
^^^^^^^^^^^^^

.. automethod:: skfem.mesh.MeshHex.save

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

Other
~~~~~

.. autoclass:: skfem.element.ElementVectorH1

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
