r"""Euler-Bernoulli beam.

This example solves the Euler-Bernoulli beam equation

.. math::
   (EI u'')'' = 1 \quad \text{in $[0,1]$},
with the boundary conditions
:math:`u(0)=u'(0) = 0` and using cubic Hermite elements.
The analytical solution gives :math:`u(1)=1/8`.

"""
from skfem import *

m = MeshLine().refined(3).with_boundaries({"left": lambda x: x[0] == 0})
e = ElementLineHermite()
basis = Basis(m, e)

@BilinearForm
def bilinf(u, v, w):
    from skfem.helpers import dd, ddot
    return ddot(dd(u), dd(v))

@LinearForm
def linf(v, w):
    return 1.0 * v

A = asm(bilinf, basis)
f = asm(linf, basis)

D = basis.get_dofs("left")

x = solve(*condense(A, f, D=D))

# compare to analytical solution
err = max(x[basis.nodal_dofs[0]]) - 1. / 8.
print(err)

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    name = splitext(argv[0])[0]

    from skfem.visuals.matplotlib import *
    plot(basis, x, Nrefs=3)
    savefig(f'{name}_solution.png')
