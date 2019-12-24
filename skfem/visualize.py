from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from numpy import ndarray

from .mesh import *


# tet
def draw_tet(m):
    """Draw the (surface) mesh."""
    bnd_facets = m.boundary_facets()
    fig = plt.figure()
    ax = Axes3D(fig)
    indexing = m.facets[:, bnd_facets].T
    ax.plot_trisurf(m.p[0, :], m.p[1, :], m.p[2,:],
                    triangles=indexing, cmap=plt.cm.viridis, edgecolor='k')
    ax.set_axis_off()
    return ax


# 1d
def plot_line(m, u, ax=None, color='ko-'):
    """Plot a function defined at the nodes of the mesh."""
    if ax is None:
        # create new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs = []
        ys = []
    for y1, y2, s, t in zip(u[m.t[0, :]],
                            u[m.t[1, :]],
                            m.p[0, m.t[0, :]],
                            m.p[0, m.t[1, :]]):
        xs.append(s)
        xs.append(t)
        xs.append(None)
        ys.append(y1)
        ys.append(y2)
        ys.append(None)
        ax.plot(xs, ys, color)
    return ax


# 2d
def draw_2d(m,
            ax: Axes = None,
            node_numbering: bool = False,
            facet_numbering: bool = False,
            element_numbering: bool = False,
            aspect: float = 1.) -> Axes:
    """Visualise a mesh by drawing the edges.

    Parameters
    ----------
    ax
        A preinitialised Matplotlib axes for plotting.
    node_numbering
        If true, draw node numbering.
    facet_numbering
        If true, draw facet numbering.
    element_numbering
        If true, draw element numbering.
    aspect
        Ratio of vertical to horizontal length-scales; ignored if ax
        specified

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if ax is None:
        # create new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(aspect)
        ax.set_axis_off()
    # visualize the mesh faster plotting is achieved through
    # None insertion trick.
    xs = []
    ys = []
    for s, t, u, v in zip(m.p[0, m.facets[0, :]],
                          m.p[1, m.facets[0, :]],
                          m.p[0, m.facets[1, :]],
                          m.p[1, m.facets[1, :]]):
        xs.append(s)
        xs.append(u)
        xs.append(None)
        ys.append(t)
        ys.append(v)
        ys.append(None)
    ax.plot(xs, ys, 'k', linewidth='0.5')

    if node_numbering:
        for itr in range(m.p.shape[1]):
            ax.text(m.p[0, itr], m.p[1, itr], str(itr))

    if facet_numbering:
        mx = .5*(m.p[0, m.facets[0, :]] +
                 m.p[0, m.facets[1, :]])
        my = .5*(m.p[1, m.facets[0, :]] +
                 m.p[1, m.facets[1, :]])
        for itr in range(m.facets.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    if element_numbering:
        mx = np.sum(m.p[0, m.t], axis=0) / m.t.shape[0]
        my = np.sum(m.p[1, m.t], axis=0) / m.t.shape[0]
        for itr in range(m.t.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    return ax


# quad
def plot_quad(m, z, **kwargs):
    """Visualise piecewise-linear or piecewise-constant function.

    The quadrilaterals are split into two triangles
    (:class:`skfem.mesh.MeshTri`) and the respective plotting function for
    the triangular mesh is used.

    """
    if len(z) == m.t.shape[-1]:
        m, z = m._splitquads(z)
    else:
        m = m._splitquads()
    return m.plot(z, **kwargs)


def plot3_quad(m, z, **kwargs):
    """Visualise nodal function (3d i.e. three axes).

    The quadrilateral mesh is split into triangular mesh (MeshTri)
    and the respective plotting function for the triangular mesh is
    used.

    """
    m, z = m._splitquads(z)
    return m.plot3(z, **kwargs)


# tri
def plot_tri(m,
             z: ndarray,
             smooth: bool = False,
             ax: Axes = None,
             zlim: Tuple[float, float] = None,
             edgecolors: str = None,
             aspect: float = 1.,
             colorbar: bool = False) -> Axes:
    """Visualise piecewise-linear or piecewise-constant function, 2D plot.

    Parameters
    ----------
    z
        An array of nodal values (Nvertices) or elemental values (Nelems).
    smooth
        If true, use gouraud shading.
    ax
        Plot onto the given preinitialised Matplotlib axes.
    zlim
        Use the given minimum and maximum values for coloring.
    edgecolors
        A string describing the edge coloring, e.g. 'k' for black.
    aspect
        The ratio of vertical to horizontal length-scales; ignored if ax
        specified.
    colorbar
        If True, show colorbar. By default not shown.

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    Examples
    --------
    Mesh the unit square :math:`(0,1)^2` and visualise the function
    :math:`f(x)=x^2`.

    >>> from skfem.mesh import MeshTri
    >>> m = MeshTri()
    >>> m.refine(3)
    >>> ax = m.plot(m.p[0, :]**2, smooth=True)
    >>> m.show()

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(aspect)
        ax.set_axis_off()
    if edgecolors is None:
        edgecolors = 'k'
    if zlim == None:
        if smooth:
            im = ax.tripcolor(m.p[0, :],
                              m.p[1, :],
                              m.t.T,
                              z,
                              shading='gouraud',
                              edgecolors=edgecolors)
        else:
            im = ax.tripcolor(m.p[0, :],
                              m.p[1, :],
                              m.t.T,
                              z,
                              edgecolors=edgecolors)
    else:
        if smooth:
            im = ax.tripcolor(m.p[0, :],
                              m.p[1, :],
                              m.t.T,
                              z,
                              shading='gouraud',
                              vmin=zlim[0],
                              vmax=zlim[1],
                              edgecolors=edgecolors)
        else:
            im = ax.tripcolor(m.p[0, :],
                              m.p[1, :],
                              m.t.T,
                              z,
                              vmin=zlim[0],
                              vmax=zlim[1],
                              edgecolors=edgecolors)

    if colorbar:
        plt.colorbar(im)
    return ax


def plot3_tri(m,
              z: ndarray,
              ax: Axes = None) -> Axes:
    """Visualise piecewise-linear or piecewise-constant function, 3D plot.

    Parameters
    ----------
    z
        An array of nodal values (Nvertices), elemental values (Nelems)
        or three elemental values (3 x Nelems, piecewise linear DG).
    ax
        Plot onto the given preinitialised Matplotlib axes.

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    Examples
    --------
    Mesh the unit square :math:`(0,1)^2` and visualise the function
    :math:`f(x)=x^2`.

    >>> from skfem.mesh import MeshTri
    >>> m = MeshTri()
    >>> m.refine(3)
    >>> ax = m.plot3(m.p[1, :]**2)
    >>> m.show()

    """
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    if len(z) == m.p.shape[1]:
        # use matplotlib
        ax.plot_trisurf(m.p[0, :],
                        m.p[1, :],
                        z,
                        triangles=m.t.T,
                        cmap=plt.cm.viridis)
    elif len(z) == m.t.shape[1]:
        # one value per element (piecewise const)
        nt = m.t.shape[1]
        newt = np.arange(3 * nt, dtype=np.int64).reshape((nt, 3))
        newpx = m.p[0, m.t].flatten(order='F')
        newpy = m.p[1, m.t].flatten(order='F')
        newz = np.vstack((z, z, z)).flatten(order='F')
        ax.plot_trisurf(newpx, newpy, newz,
                        triangles=newt.T,
                        cmap=plt.cm.viridis)
    elif len(z) == 3 * m.t.shape[1]:
        # three values per element (piecewise linear)
        nt = m.t.shape[1]
        newt = np.arange(3 * nt, dtype=np.int64).reshape((nt, 3))
        newpx = m.p[0, m.t].flatten(order='F')
        newpy = m.p[1, m.t].flatten(order='F')
        ax.plot_trisurf(newpx,
                        newpy,
                        z,
                        triangles=newt.T,
                        cmap=plt.cm.viridis)
    else:
        raise NotImplementedError("MeshTri.plot3: not implemented for "
                                  "the given shape of input vector!")
    return ax


# generic
def draw(m, *args, **kwargs):
    if isinstance(m, Mesh2D):
        draw_2d(m, *args, **kwargs)
    elif isinstance(m, MeshTet):
        draw_tet(m, *args, **kwargs)
    else:
        raise NotImplementedError("The given Mesh type not supported!")


def plot(m, *args, **kwargs):
    if isinstance(m, MeshTri):
        plot_tri(m, *args, **kwargs)
    elif isinstance(m, MeshQuad):
        plot_quad(m, *args, **kwargs)
    elif isinstance(m, MeshLine):
        plot_line(m, *args, **kwargs)
    else:
        raise NotImplementedError("The given Mesh type not supported!")


def plot3(m, *args, **kwargs):
    if isinstance(m, MeshTri):
        plot3_tri(m, *args, **kwargs)
    elif isinstance(m, MeshQuad):
        plot3_quad(m, *args, **kwargs)
    else:
        raise NotImplementedError("The given Mesh type not supported!")


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)


def show(*args, **kwargs):
    plt.show(*args, **kwargs)
