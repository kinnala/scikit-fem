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
   :members: load, save, refined, facets_satisfying, nodes_satisfying, elements_satisfying, with_subdomains, with_boundaries

Class: MeshTri
**************

.. autoclass:: skfem.mesh.MeshTri

.. autoclass:: skfem.mesh.MeshTri1
   :members: __init__, init_symmetric, init_sqsymmetric, init_refdom, init_tensor, init_lshaped, init_circle, load

.. autoclass:: skfem.mesh.MeshTri2
   :members: init_circle, load

Class: MeshQuad
***************

.. autoclass:: skfem.mesh.MeshQuad

.. autoclass:: skfem.mesh.MeshQuad1
   :members: __init__, init_refdom, init_tensor, to_meshtri, load

.. autoclass:: skfem.mesh.MeshQuad2
   :members: load

Class: MeshTet
**************

.. autoclass:: skfem.mesh.MeshTet

.. autoclass:: skfem.mesh.MeshTet1
   :members: __init__, init_refdom, init_tensor, init_ball, load

.. autoclass:: skfem.mesh.MeshTet2
   :members: init_ball, load

Class: MeshHex
**************

.. autoclass:: skfem.mesh.MeshHex

.. autoclass:: skfem.mesh.MeshHex1
   :members: __init__, init_tensor, to_meshtet, load

Class: MeshLine
***************

.. autoclass:: skfem.mesh.MeshLine

.. autoclass:: skfem.mesh.MeshLine1
   :members: __init__

Module: skfem.assembly
======================

.. automodule:: skfem.assembly

.. autofunction:: skfem.assembly.asm

Abstract class: AbstractBasis
-----------------------------

Subclasses of :class:`~skfem.assembly.basis.AbstractBasis` represent a global
finite element basis evaluated at quadrature points.

.. autoclass:: skfem.assembly.basis.AbstractBasis
   :members: get_dofs, interpolate, interpolate_multiple, project

Class: CellBasis
****************

.. autoclass:: skfem.assembly.Basis

.. autoclass:: skfem.assembly.CellBasis
   :members: __init__, interpolate, interpolate_multiple, project


Class: FacetBasis
*****************

.. autoclass:: skfem.assembly.FacetBasis
   :members: __init__

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
   :show-inheritance:

.. autosummary::

    ElementH1
    ElementVector
    ElementHdiv
    ElementHcurl
    ElementGlobal
    ElementDG
    ElementComposite
    ElementTriP1
    ElementTriP2
    ElementTriP3
    ElementTriP4
    ElementTriP
    ElementTriP0
    ElementTriCR
    ElementTriCCR
    ElementTriRT0
    ElementTriRT
    ElementTriMorley
    ElementTri15ParamPlate
    ElementTriArgyris
    ElementTriMini
    ElementTriHermite
    ElementTriSkeletonP0
    ElementTriSkeletonP1
    ElementTriBDM1
    ElementTriBubble
    ElementQuad0
    ElementQuad1
    ElementQuad2
    ElementQuadS2
    ElementQuadP
    ElementQuadBFS
    ElementQuadRT0
    ElementQuadDPC
    ElementTetP0
    ElementTetP1
    ElementTetP2
    ElementTetRT0
    ElementTetN0
    ElementTetMini
    ElementTetCR
    ElementTetCCR
    ElementTetBubble
    ElementTetMorley
    ElementTetMWX3
    ElementHex0
    ElementHex1
    ElementHex2
    ElementHexS2
    ElementHexDPC
    ElementLineP0
    ElementLineP1
    ElementLineP2
    ElementLinePp
    ElementLineHermite
    ElementLineMini
    ElementLineBubble
   

.. note::

   The element global basis is calculated at quadrature points and stored inside
   :class:`~skfem.element.DiscreteField` objects.
   The different element types precalculate different fields of
   :class:`~skfem.element.DiscreteField`.  E.g., for :math:`H(div)`
   finite elements it is natural to precalculate ``DiscreteField.div``.
   The high order derivatives are created only when using subclasses of
   :class:`~skfem.element.ElementGlobal`.
                 

.. autoclass:: skfem.element.DiscreteField

Module: skfem.utils
===================

Function: solve
---------------

.. autofunction:: skfem.utils.solve

Function: condense
------------------

.. autofunction:: skfem.utils.condense

Function: enforce
-----------------

.. autofunction:: skfem.utils.enforce

Function: solver_petsc
----------------------

.. autofunction:: skfem.utils.solver_petsc

Module: skfem.distributed
=========================

.. automodule:: skfem.distributed

Function: distribute
--------------------

.. autofunction:: skfem.distributed.distribute

Function: allreduce
-------------------

.. autofunction:: skfem.distributed.allreduce

Module: skfem.helpers
=====================

.. automodule:: skfem.helpers

.. autofunction:: skfem.helpers.grad

.. autofunction:: skfem.helpers.div

.. autofunction:: skfem.helpers.curl

.. autofunction:: skfem.helpers.d

.. autofunction:: skfem.helpers.dd

.. autofunction:: skfem.helpers.ddd

.. autofunction:: skfem.helpers.dddd

.. autofunction:: skfem.helpers.sym_grad

.. autofunction:: skfem.helpers.dot

.. autofunction:: skfem.helpers.ddot

.. autofunction:: skfem.helpers.dddot

.. autofunction:: skfem.helpers.mul

.. autofunction:: skfem.helpers.trace

.. autofunction:: skfem.helpers.transpose

.. autofunction:: skfem.helpers.prod

.. autofunction:: skfem.helpers.inv

Module: skfem.visuals
=====================

.. autofunction:: skfem.visuals.matplotlib.plot
