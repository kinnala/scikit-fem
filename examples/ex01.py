from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
basis = InteriorBasis(m, e)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

I = m.interior_nodes()

x = 0*b
x[I] = solve(*condense(A, b, I=I))

if __name__ == "__main__":
    m.plot3(x)
    m.show()
"""
Poisson equation, zero on boundary
----------------------------------

The canonical model problem for second-order partial
differential equations reads: find :math:`u : \Omega \rightarrow \mathbb{R}`
that satisfies

.. math::
    \begin{aligned}
        -\Delta u &= f, && \text{in $\Omega$},\\
        u &= 0, && \text{on $\partial \Omega$},
    \end{aligned}

where :math:`\Omega = (0,1)^2` and :math:`f` is the loading.  This simple model
problem and its variants have applications, e.g., in `electrostatics
<https://en.wikipedia.org/wiki/Laplace%27s_equation#Electrostatics_2>`_,
`acoustics <https://en.wikipedia.org/wiki/Helmholtz_equation>`_ and `fluid flow
<https://en.wikipedia.org/wiki/Potential_flow#Analysis_for_two-dimensional_flow>`_.


In this example, we solve the problem using piecewise-linear triangular finite
elements. A square domain can be meshed using the default constructor of
MeshTri. In the following excerpt, the initial mesh is further refined four
times.

.. literalinclude:: ../examples/ex01.py
    :lines: 1-4

The continuous weak formulation reads:
find :math:`u \in H^1_0(\Omega)` that satisfies

.. math::
    (\nabla u, \nabla v) = (f, v), \quad \forall v \in H^1_0(\Omega).

"""
