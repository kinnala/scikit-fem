==========================
 Detailed API description
==========================

This section contains API documentation for the most commonly used interfaces
of the library.

Module: skfem.mesh
==================

.. automodule:: skfem.mesh

Abstract class: Mesh
--------------------

.. autoclass:: skfem.mesh.Mesh
   :members: load, save, refined, element_finder, doflocs, t

Class: MeshTri
**************

.. autoclass:: skfem.mesh.MeshTri

.. autoclass:: skfem.mesh.mesh.MeshTri1
   :members: __init__, init_symmetric, init_sqsymmetric, init_refdom, init_tensor, init_lshaped, init_circle, load

.. autoclass:: skfem.mesh.mesh.MeshTri2
   :members: init_circle, load

Class: MeshQuad
***************

.. autoclass:: skfem.mesh.MeshQuad

.. autoclass:: skfem.mesh.mesh.MeshQuad1
   :members: __init__, init_refdom, init_tensor, to_meshtri, load

.. autoclass:: skfem.mesh.mesh.MeshQuad2
   :members: load

Class: MeshTet
**************

.. autoclass:: skfem.mesh.MeshTet

.. autoclass:: skfem.mesh.mesh.MeshTet1
   :members: __init__, init_refdom, init_tensor, init_ball, load

.. autoclass:: skfem.mesh.mesh.MeshTet2
   :members: init_ball, load

Class: MeshHex
**************

.. autoclass:: skfem.mesh.MeshHex

.. autoclass:: skfem.mesh.mesh.MeshHex1
   :members: __init__, init_tensor, to_meshtet, load

Class: MeshLine
***************

.. autoclass:: skfem.mesh.MeshLine

.. autoclass:: skfem.mesh.mesh.MeshLine1
   :members: __init__

Module: skfem.assembly
======================

.. automodule:: skfem.assembly

Abstract class: AbstractBasis
-----------------------------

Subclasses of :class:`~skfem.assembly.basis.AbstractBasis` represent a global
finite element basis evaluated at quadrature points.

Class: CellBasis
****************

.. autoclass:: skfem.assembly.Basis

.. autoclass:: skfem.assembly.CellBasis
   :members: __init__


Class: BoundaryFacetBasis
*************************

.. autoclass:: skfem.assembly.FacetBasis

.. autoclass:: skfem.assembly.BoundaryFacetBasis
   :members: __init__, trace

Class: InteriorFacetBasis
*************************

.. autoclass:: skfem.assembly.InteriorFacetBasis
   :members: __init__


Abstract class: Form
--------------------

Class: BilinearForm
*******************

.. autoclass:: skfem.assembly.BilinearForm
   :members: assemble

Class: LinearForm
*****************

.. autoclass:: skfem.assembly.LinearForm
   :members: assemble

Class: Functional
*****************

.. autoclass:: skfem.assembly.Functional
   :members: assemble, elemental

Module: skfem.element
=====================

.. automodule:: skfem.element
   :members:
   :show-inheritance:
   :exclude-members: DiscreteField, ElementVectorH1

Module: skfem.utils
===================

Function: solve
---------------

.. autofunction:: skfem.utils.solve

Function: condense
------------------

.. autofunction:: skfem.utils.condense

Function: project
-----------------

.. autofunction:: skfem.utils.project

Module: skfem.helpers
=====================

.. automodule:: skfem.helpers

.. autofunction:: skfem.helpers.grad

.. autofunction:: skfem.helpers.div

.. autofunction:: skfem.helpers.curl

.. autofunction:: skfem.helpers.d

.. autofunction:: skfem.helpers.dd

.. autofunction:: skfem.helpers.sym_grad

.. autofunction:: skfem.helpers.dot

.. autofunction:: skfem.helpers.ddot
