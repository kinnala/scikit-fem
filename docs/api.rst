==========================
 Detailed API description
==========================

This section contains a more detailed API documentation for the most commonly
used interfaces of the library.

Class: Mesh
===========

.. autoclass:: skfem.mesh.Mesh
   :members: load, save, refine

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

Class: Basis
============

.. autoclass:: skfem.assembly.Basis
   :members: find_dofs, get_dofs, interpolate

.. autoclass:: skfem.assembly.InteriorBasis
   :members: __init__

.. autoclass:: skfem.assembly.FacetBasis
   :members: __init__


Class: Form
===========

.. autoclass:: skfem.assembly.BilinearForm

.. autoclass:: skfem.assembly.LinearForm

.. autoclass:: skfem.assembly.Functional

.. autofunction:: skfem.assembly.asm

Module: skfem.utils
===================

.. autofunction:: skfem.utils.solve

.. autofunction:: skfem.utils.condense
