===================================
 A detailed description of the API
===================================

This section contains a detailed API documentation for the most commonly used
interfaces of the library.

Class: Mesh
===========

.. autoclass:: skfem.mesh.Mesh
   :members: load, save, refine, define_boundary

Class: MeshTri
--------------

.. autoclass:: skfem.mesh.MeshTri
     :members: __init__, init_symmetric, init_sqsymmetric, init_refdom, init_tensor, init_lshaped

Class: MeshQuad
---------------

.. autoclass:: skfem.mesh.MeshQuad
   :members: __init__, init_refdom, init_tensor

Class: MeshTet
--------------

.. autoclass:: skfem.mesh.MeshTet
   :members: __init__, init_refdom, init_tensor

Class: MeshHex
--------------

.. autoclass:: skfem.mesh.MeshHex
   :members: __init__, init_tensor

Class: MeshLine
---------------

.. autoclass:: skfem.mesh.MeshLine
   :members: __init__

Class: Element
==============

.. autoclass:: skfem.element.Element

Triangular elements
-------------------

.. autosummary::

   skfem.element.ElementTriP1
   skfem.element.ElementTriP2
   skfem.element.ElementTriP0
   skfem.element.ElementTriArgyris
   skfem.element.ElementTriMorley
   skfem.element.ElementTriMini
   skfem.element.ElementTriDG
   skfem.element.ElementTriRT0

Quadrilateral elements
----------------------

.. autosummary::

   skfem.element.ElementQuad1
   skfem.element.ElementQuad2
   skfem.element.ElementQuadS2
   skfem.element.ElementQuad0
   skfem.element.ElementQuadP
   skfem.element.ElementQuadDG
   skfem.element.ElementQuadBFS

Tetrahedral elements
--------------------

.. autosummary::

   skfem.element.ElementTetP1
   skfem.element.ElementTetP2
   skfem.element.ElementTetP0
   skfem.element.ElementTetRT0
   skfem.element.ElementTetN0

Hexahedral elements
-------------------

.. autosummary::

   skfem.element.ElementHex1

1D elements
-----------

.. autosummary::

   skfem.element.ElementLineP1
   skfem.element.ElementLineP2
   skfem.element.ElementLinePp
   skfem.element.ElementLineHermite

Other
-----

.. autosummary::

   skfem.element.ElementVectorH1
   skfem.element.ElementComposite

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
