from ..element_h2 import ElementH2


class ElementTriArgyris(ElementH2):
    nodal_dofs = 6
    facet_dofs = 1
    dim = 2
    maxdeg = 5
    dofnames = ['u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_n']

    def gdof(self, u, du, ddu, v, e, n, i):
        if i < 18:
            j = i % 6
            k = int(i/6)
            if j == 0:
                return u(*v[k])
            elif j == 1:
                return du[0](*v[k])
            elif j == 2:
                return du[1](*v[k])
            elif j == 3:
                return ddu[0](*v[k])
            elif j == 4:
                return ddu[1](*v[k])
            elif j == 5:
                return ddu[2](*v[k])
        elif i == 18:
            return du[0](*e[0]) * n[0, 0] + du[1](*e[0]) * n[0, 1]
        elif i == 19:
            return du[0](*e[1]) * n[1, 0] + du[1](*e[1]) * n[1, 1]
        elif i == 20:
            return du[0](*e[2]) * n[2, 0] + du[1](*e[2]) * n[2, 1]
