"""Visualize Argyris basis."""

from skfem import *
import numpy as np

m = MeshTri.init_sqsymmetric()
e = ElementTriArgyris()

ib = Basis(m, e, intorder=5)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skfem.visuals.matplotlib import plot, draw
    f, axes = plt.subplots(3,3)

    ixs = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0)]
    i = 0

    for itr in ib.nodal_dofs[:,4]:
        axi = axes[ixs[i]]
        axi.set_axis_off()
        X = ib.zeros()
        X[itr] = 1.0
        plot(ib, X, Nrefs=5, shading='gouraud', ax=axi)
        i += 1

    axi = axes[(1,1)]
    axi.set_axis_off()
    draw(m, ax=axi)

    axi = axes[(2,1)]
    axi.set_axis_off()
    X = ib.zeros()
    X[np.array([56,59,64,66])] = 1.0
    plot(ib, X, Nrefs=5, shading='gouraud', ax=axi)

    axi = axes[(2,2)]
    axi.set_axis_off()
    X = ib.zeros()
    X[np.array([58,61,63,65])] = 1.0
    plot(ib, X, Nrefs=5, shading='gouraud', ax=axi)

    plt.axis('off')
    plt.show()
