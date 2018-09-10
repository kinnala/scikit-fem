"""
Author: kinnala

Visualise the Argyris basis functions.
"""
from skfem import *
import matplotlib.pyplot as plt

m = MeshTri.init_sqsymmetric()
map = MappingAffine(m)
e = ElementTriArgyris()

ib = InteriorBasis(m, e, map, 5)

f, axes = plt.subplots(3,3)

ixs = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0)]
i=0

for itr in ib.nodal_dofs[:,4]:
    axi=axes[ixs[i]]
    axi.set_axis_off()
    X = np.zeros(ib.N)
    X[itr] = 1.0
    M,x = ib.refinterp(X, 5)
    M.plot(x,smooth=True,ax=axi)
    i+=1

axi = axes[(1,1)]
axi.set_axis_off()
m.draw(ax=axi)

axi = axes[(2,1)]
axi.set_axis_off()
X = np.zeros(ib.N)
X[np.array([56,59,64,66])] = 1.0
M,x = ib.refinterp(X, 5)
M.plot(x,smooth=True,ax=axi)

axi = axes[(2,2)]
axi.set_axis_off()
X = np.zeros(ib.N)
X[np.array([58,61,63,65])] = 1.0
M,x = ib.refinterp(X, 5)
M.plot(x,smooth=True,ax=axi)



plt.axis('off')
M.show()
