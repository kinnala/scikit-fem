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
   :members: load, save, refine

Class: MeshTri
**************

.. autoclass:: skfem.mesh.MeshTri
   :members: __init__, init_symmetric, init_sqsymmetric, init_refdom, init_tensor, init_lshaped

Class: MeshQuad
***************

.. autoclass:: skfem.mesh.MeshQuad
   :members: __init__, init_refdom, init_tensor

Class: MeshTet
**************

.. autoclass:: skfem.mesh.MeshTet
   :members: __init__, init_refdom, init_tensor

Class: MeshHex
**************

.. autoclass:: skfem.mesh.MeshHex
   :members: __init__, init_tensor

Class: MeshLine
***************

.. autoclass:: skfem.mesh.MeshLine
   :members: __init__

Module: skfem.assembly
======================

.. automodule:: skfem.assembly

Abstract class: Basis
---------------------

.. autoclass:: skfem.assembly.Basis
   :members: find_dofs, get_dofs, interpolate

Class: InteriorBasis
********************

.. autoclass:: skfem.assembly.InteriorBasis
   :members: __init__

Class: ExteriorFacetBasis
*************************

.. autoclass:: skfem.assembly.ExteriorFacetBasis
   :members: __init__, trace

Class: InteriorFacetBasis
*************************

.. autoclass:: skfem.assembly.InteriorFacetBasis
   :members: __init__, trace


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
   :exclude-members: DiscreteField

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
