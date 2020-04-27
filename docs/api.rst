===================================
 A detailed description of the API
===================================

This section contains a detailed API documentation for the most commonly used
interfaces of the library.

Module: skfem.mesh
==================

.. autoclass:: skfem.mesh.Mesh

.. automethod:: skfem.mesh.Mesh.load

.. automethod:: skfem.mesh.Mesh.save

.. automethod:: skfem.mesh.Mesh.refine

.. automethod:: skfem.mesh.Mesh.define_boundary

Class: MeshTri
--------------

.. autoclass:: skfem.mesh.MeshTri

.. automethod:: skfem.mesh.MeshTri.__init__

.. automethod:: skfem.mesh.MeshTri.init_symmetric

.. automethod:: skfem.mesh.MeshTri.init_sqsymmetric

.. automethod:: skfem.mesh.MeshTri.init_refdom

.. automethod:: skfem.mesh.MeshTri.init_tensor

.. automethod:: skfem.mesh.MeshTri.init_lshaped


Class: MeshQuad
---------------

.. autoclass:: skfem.mesh.MeshQuad

.. automethod:: skfem.mesh.MeshQuad.__init__

.. automethod:: skfem.mesh.MeshQuad.init_refdom

.. automethod:: skfem.mesh.MeshQuad.init_tensor


Class: MeshTet
--------------

.. autoclass:: skfem.mesh.MeshTet

.. automethod:: skfem.mesh.MeshTet.__init__

.. automethod:: skfem.mesh.MeshTet.init_refdom

.. automethod:: skfem.mesh.MeshTet.init_tensor


Class: MeshHex
--------------

.. autoclass:: skfem.mesh.MeshHex

.. automethod:: skfem.mesh.MeshHex.__init__

.. automethod:: skfem.mesh.MeshHex.init_tensor


Class: MeshLine
---------------

.. autoclass:: skfem.mesh.MeshLine

Module: skfem.element
=====================

Triangular elements
-------------------

.. autoclass:: skfem.element.ElementTriP1

.. autoclass:: skfem.element.ElementTriP2

.. autoclass:: skfem.element.ElementTriP0

.. autoclass:: skfem.element.ElementTriArgyris

.. autoclass:: skfem.element.ElementTriMorley

.. autoclass:: skfem.element.ElementTriDG

.. autoclass:: skfem.element.ElementTriRT0

Quadrilateral elements
----------------------

.. autoclass:: skfem.element.ElementQuad1

.. autoclass:: skfem.element.ElementQuad2

.. autoclass:: skfem.element.ElementQuad0

Tetrahedral elements
--------------------

.. autoclass:: skfem.element.ElementTetP1

.. autoclass:: skfem.element.ElementTetP2

.. autoclass:: skfem.element.ElementTetP0

.. autoclass:: skfem.element.ElementTetRT0

.. autoclass:: skfem.element.ElementTetN0

Hexahedral elements
-------------------

.. autoclass:: skfem.element.ElementHex1

1D elements
-----------

.. autoclass:: skfem.element.ElementLineP1

.. autoclass:: skfem.element.ElementLineP2

Other
-----

.. autoclass:: skfem.element.ElementVectorH1

Module: skfem.assembly
======================

.. autoclass:: skfem.assembly.Basis

.. automethod:: skfem.assembly.Basis.get_dofs

.. autoclass:: skfem.assembly.InteriorBasis

.. autoclass:: skfem.assembly.FacetBasis

.. autoclass:: skfem.assembly.BilinearForm

.. autoclass:: skfem.assembly.LinearForm

.. autoclass:: skfem.assembly.Functional

.. autofunction:: skfem.assembly.asm

Module: skfem.utils
===================

.. autofunction:: skfem.utils.solve

.. autofunction:: skfem.utils.condense

Module: skfem.visuals
=====================

.. automodule:: skfem.visuals.matplotlib

.. autofunction:: skfem.visuals.matplotlib.draw

.. autofunction:: skfem.visuals.matplotlib.plot

.. autofunction:: skfem.visuals.matplotlib.plot3
