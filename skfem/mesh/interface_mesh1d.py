import numpy as np

from ..mesh import Mesh


class InterfaceMesh1D(Mesh):
    """An interface mesh for mortar methods."""

    def __init__(self, mesh1, mesh2, rule, param, debug_plot=False):
        self.brefdom = mesh1.brefdom

        p1_ix = mesh1.nodes_satisfying(rule)
        p2_ix = mesh2.nodes_satisfying(rule)

        p1 = mesh1.p[:, p1_ix]
        p2 = mesh2.p[:, p2_ix]
        _, ix = np.unique(np.concatenate((param(p1[0, :], p1[1, :]), param(p2[0, :], p2[1, :]))), return_index=True)

        np1 = mesh1.p.shape[1]
        nt1 = mesh1.t.shape[1]
        ixorig = np.concatenate((p1_ix, p2_ix + np1))[ix]

        self.p = np.hstack((mesh1.p, mesh2.p))
        self.t = np.hstack((mesh1.t, mesh2.t + np1))
        self.facets = np.array([ixorig[:-1], ixorig[1:]])
        self.t2f = -1 + 0*np.hstack((mesh1.t2f, mesh2.t2f))

        # construct normals
        tangent_x = self.p[0, self.facets[0, :]] - self.p[0, self.facets[1, :]]
        tangent_y = self.p[1, self.facets[0, :]] - self.p[1, self.facets[1, :]]
        tangent_lengths = np.sqrt(tangent_x**2 + tangent_y**2)

        self.normals = np.array([-tangent_y/tangent_lengths, tangent_x/tangent_lengths])

        if debug_plot:
            ax = mesh1.draw()
            mesh2.draw(ax=ax)
            xs = np.array([self.p[0, self.facets[0, :]], self.p[0, self.facets[1, :]]])
            midx = np.sum(xs, axis=0)/2.0
            ys = np.array([self.p[1, self.facets[0, :]], self.p[1, self.facets[1, :]]])
            midy = np.sum(ys, axis=0)/2.0
            xs = 0.9*(xs - midx) + midx
            ys = 0.9*(ys - midy) + midy
            ax.plot(xs, ys, 'x-')

        # mappings from facets to the original triangles
        # TODO vectorize
        self.f2t = self.facets*0-1
        for itr in range(self.facets.shape[1]):
            mx = .5*(self.p[0, self.facets[0, itr]] + self.p[0, self.facets[1, itr]])
            my = .5*(self.p[1, self.facets[0, itr]] + self.p[1, self.facets[1, itr]])
            val = param(mx, my)
            for jtr in mesh1.boundary_facets():
                fix1 = mesh1.facets[0, jtr]
                x1 = mesh1.p[0, fix1]
                y1 = mesh1.p[1, fix1]
                fix2 = mesh1.facets[1, jtr]
                x2 = mesh1.p[0, fix2]
                y2 = mesh1.p[1, fix2]
                if rule(x1, y1) > 0 or rule(x2, y2) > 0:
                    if val > param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
                    elif val < param(x1, y1) and val > param(x2, y2): # ye olde
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
                    elif val >= param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
                    elif val > param(x1, y1) and val <= param(x2, y2):
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
                    elif val <= param(x1, y1) and val > param(x2, y2):
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
                    elif val < param(x1, y1) and val >= param(x2, y2):
                        # OK
                        self.f2t[0, itr] = mesh1.f2t[0, jtr]
                        break
            for jtr in mesh2.boundary_facets():
                fix1 = mesh2.facets[0, jtr]
                x1 = mesh2.p[0, fix1]
                y1 = mesh2.p[1, fix1]
                fix2 = mesh2.facets[1, jtr]
                x2 = mesh2.p[0, fix2]
                y2 = mesh2.p[1, fix2]
                if rule(x1, y1) > 0 or rule(x2, y2) > 0:
                    if val > param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
                    elif val < param(x1, y1) and val > param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
                    elif val >= param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
                    elif val > param(x1, y1) and val <= param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
                    elif val <= param(x1, y1) and val > param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
                    elif val < param(x1, y1) and val >= param(x2, y2):
                        # OK
                        self.f2t[1, itr] = mesh2.f2t[0, jtr] + nt1
                        break
        if (self.f2t>-1).all():
            self.f2t[0, :]
            return
        else:
            print(self.f2t)
            raise Exception("All mesh facets corresponding to mortar facets not found!")
