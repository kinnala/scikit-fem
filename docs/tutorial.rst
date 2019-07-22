.. _tutorial:

Tutorial
--------

This is a walkthrough of the basic features of scikit-fem.

Creating meshes
###############

The default constructors of :class:`~skfem.mesh.Mesh`
classes create simple meshes of unit intervals :math:`[0,1]^d`.

.. code-block:: python

   In [1]: from skfem.mesh import MeshTri
   In [2]: m = MeshTri()
   In [3]: m
   Out[3]: "Triangular mesh with 4 vertices and 2 elements."

There are also a few additional constructors available such as
:meth:`skfem.mesh.MeshTri.init_tensor` and
:meth:`skfem.mesh.MeshTri.init_lshaped`. More importantly, all
mesh types can be loaded from file formats supported by meshio:

.. code-block:: python

   In [4]: from skfem.mesh import Mesh
   In [5]: Mesh.load("docs/examples/square.msh")
   Out[5]: "Triangular mesh with 109 vertices and 184 elements."

You can also visualize meshes and solutions via
:meth:`skfem.mesh.MeshTri.draw` and :meth:`skfem.mesh.MeshTri.plot`.

Choosing basis functions
########################

The local basis functions are defined in :class:`~skfem.element.Element`
classes. They are combined with meshes to create
:class:`~skfem.assembly.GlobalBasis` objects that contain global basis
functions evaluated at quadrature points:

.. code-block:: python

Assembling finite element matrices
##################################

Setting boundary conditions
###########################

Solving linear systems
######################

Postprocessing the results
##########################
