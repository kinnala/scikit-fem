.. _poisson:

Solving Poisson equation
------------------------

In this short tutorial, we demonstrate the basic concepts and workflow of
scikit-fem. We solve the canonical model problem for second-order PDE's: find
:math:`u : \Omega \rightarrow \mathbb{R}` that satisfies

.. math::
    \begin{aligned}
        -\Delta u &= f, && \text{in $\Omega$},\\
        u &= 0, && \text{on $\partial \Omega$},
    \end{aligned}

where :math:`\Omega = (0,1)^2` and :math:`f` is the loading.  This simple model
problem has applications, e.g., in `electrostatics
<https://en.wikipedia.org/wiki/Laplace%27s_equation#Electrostatics_2>`_,
`acoustics <https://en.wikipedia.org/wiki/Helmholtz_equation>`_ and `fluid flow
<https://en.wikipedia.org/wiki/Potential_flow#Analysis_for_two-dimensional_flow>`_.
Finite element method solves the problem's weak formulation:
find :math:`u \in H^1_0(\Omega)` that satisfies

.. math::
    (\nabla u, \nabla v) = (f, v)  \quad \forall v \in H^1_0(\Omega).

Mesh and basis
##############
    
We discretise the problem using piecewise-linear triangular finite elements.  A
mesh for the unit square is given by the default constructor of
:class:`~skfem.mesh.MeshTri`.  In the following, we further refine the mesh four
times:

.. literalinclude:: ../examples/ex01.py
    :start-at: skfem
    :end-at: refine	       

After creating the mesh, we evaluate the finite element basis at the global
quadrature points.

.. literalinclude:: ../examples/ex01.py
    :start-at: ElementTriP1
    :end-at: InteriorBasis	       

:class:`~skfem.element.ElementTriP1` defines the piecewise linear basis
functions and local-to-global transformation rules.

Assembly
########

The bilinear and linear forms are defined using the decorators
:func:`~skfem.assembly.BilinearForm` and
:func:`~skfem.assembly.LinearForm`. It is important to have the order of the
form arguments correct.

.. literalinclude:: ../../skfem/models/poisson.py
    :start-at: BilinearForm
    :end-at: return

.. literalinclude:: ../../skfem/models/poisson.py
    :start-at: LinearForm
    :end-at: return

All assembly operations are performed using the function :func:`~skfem.assembly.asm`.

.. literalinclude:: ../examples/ex01.py
    :start-at: asm(laplace, basis)
    :end-at: asm(load, basis)

Solving and plotting
####################

We are left with solving the assembled linear system.
We eliminate the boundary degrees-of-freedom using
:func:`~skfem.utils.condense` and call :func:`~skfem.utils.solve`.

.. literalinclude:: ../examples/ex01.py
    :start-at: solve
    :end-at: solve	       

The solution can now be visualised using :meth:`~skfem.visuals.matplotlib.plot`. 

.. literalinclude:: ../examples/ex01.py
    :start-at: main
    :end-at: solution.png

.. figure:: ../examples/ex01_solution.png

    The solution of Poisson equation.

The complete source code reads as follows:

.. literalinclude:: ../examples/ex01.py
    :linenos:
