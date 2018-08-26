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

.. automethod:: skfem.mesh.MeshTri.mirror_mesh

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

.. automethod:: skfem.mesh.MeshTri.interpolator

.. automethod:: skfem.mesh.MeshTri.const_interpolator

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

.. automethod:: skfem.mesh.MeshQuad.mirror_mesh

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

Visualisation
^^^^^^^^^^^^^

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

MeshLine
~~~~~~~~

.. autoclass:: skfem.mesh.MeshLine

Constructors
^^^^^^^^^^^^

Visualisation
^^^^^^^^^^^^^

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
