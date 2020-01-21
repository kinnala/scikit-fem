"""Drawing meshes and solutions using matplotlib."""

from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

from ..assembly import InteriorBasis
from ..mesh import *


@singledispatch
def draw(m, **kwargs) -> Axes:
    """Visualize meshes."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@draw.register(MeshTet)
def _(m: MeshTet, **kwargs) -> Axes:
    """Visualize a tetrahedral mesh by drawing the boundary facets."""
    bnd_facets = m.boundary_facets()
    fig = plt.figure()
    ax = Axes3D(fig)
    indexing = m.facets[:, bnd_facets].T
    ax.plot_trisurf(m.p[0, :], m.p[1, :], m.p[2,:],
                    triangles=indexing, cmap=plt.cm.viridis, edgecolor='k')
    ax.set_axis_off()
    return ax


@draw.register(Mesh2D)
def _(m: Mesh2D, **kwargs) -> Axes:
    """Visualise a two-dimensional mesh by drawing the edges.

    Parameters
    ----------
    m
        A two-dimensional mesh.
    ax (optional)
        A preinitialised Matplotlib axes for plotting.
    node_numbering (optional)
        If true, draw node numbering.
    facet_numbering (optional)
        If true, draw facet numbering.
    element_numbering (optional)
        If true, draw element numbering.
    aspect (optional)
        Ratio of vertical to horizontal length-scales; ignored if ax
        specified

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if "ax" not in kwargs:
        # create new figure
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
        ax.set_axis_off()
    else:
        ax = kwargs["ax"]
    # faster plotting is achieved through
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

    if "node_numbering" in kwargs:
        for itr in range(m.p.shape[1]):
            ax.text(m.p[0, itr], m.p[1, itr], str(itr))

    if "facet_numbering" in kwargs:
        mx = .5*(m.p[0, m.facets[0, :]] +
                 m.p[0, m.facets[1, :]])
        my = .5*(m.p[1, m.facets[0, :]] +
                 m.p[1, m.facets[1, :]])
        for itr in range(m.facets.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    if "element_numbering" in kwargs:
        mx = np.sum(m.p[0, m.t], axis=0) / m.t.shape[0]
        my = np.sum(m.p[1, m.t], axis=0) / m.t.shape[0]
        for itr in range(m.t.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    return ax



@singledispatch
def plot(m, u, **kwargs) -> Axes:
    """Plot functions defined on nodes of the mesh."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot.register(MeshLine)
def _(m: MeshLine, z: ndarray, **kwargs):
    """Plot a function defined at the nodes of the 1D mesh."""
    if "ax" not in kwargs:
        # create new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs = []
        ys = []
    else:
        ax = kwargs["ax"]

    color = kwargs["color"] if "color" in kwargs else 'ko-'
    for y1, y2, s, t in zip(z[m.t[0, :]],
                            z[m.t[1, :]],
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


@plot.register(MeshTri)
def _(m: MeshTri, z: ndarray, **kwargs) -> Axes:
    """Visualise piecewise-linear function on a triangular mesh.

    Parameters
    ----------
    m
        A triangular mesh.
    z
        An array of nodal values.
    ax (optional)
        Plot onto the given preinitialised Matplotlib axes.
    aspect (optional)
        The ratio of vertical to horizontal length-scales; ignored if ax
        specified.
    colorbar (optional)
        If True, show colorbar. By default not shown.
    figsize (optional)
        Passed on to matplotlib.
    shading (optional)
    edgecolors (optional)
    vmin (optional)
    vmax (optional)
        Passed on to matplotlib.

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if "ax" not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
        ax.set_axis_off()
    else:
        ax = kwargs["ax"]

    im = ax.tripcolor(m.p[0, :], m.p[1, :], m.t.T, z,
                      **{k: v for k, v in kwargs.items()
                         if k in ['shading',
                                  'edgecolors',
                                  'vmin',
                                  'vmax']})

    if "colorbar" in kwargs:
        plt.colorbar(im)
    return ax


@plot.register(MeshQuad)
def _(m: MeshQuad, z, **kwargs):
    """Visualise nodal functions on quadrilateral meshes.

    The quadrilaterals are split into two triangles
    (:class:`skfem.mesh.MeshTri`) and the respective plotting function for the
    triangular mesh is used.

    """
    if len(z) == m.t.shape[-1]:
        m, z = m._splitquads(z)
    else:
        m = m._splitquads()
    return plot(m, z, **kwargs)


@plot.register(InteriorBasis)
def _(basis: InteriorBasis, z: ndarray, **kwargs) -> Axes:
    """Plot on a refined mesh via :meth:`InteriorBasis.refinterp`."""
    Nrefs = kwargs["Nrefs"] if "Nrefs" in kwargs else 1
    return plot(*basis.refinterp(z, Nrefs=Nrefs), **kwargs)


@singledispatch
def plot3(m, z: ndarray, **kwargs) -> Axes:
    """Plot functions defined on nodes of the mesh (3D)."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot3.register(MeshTri)
def _(m: MeshTri, z: ndarray, **kwargs) -> Axes:
    """Visualise piecewise-linear function, 3D plot.

    Parameters
    ----------
    z
        An array of nodal values (Nvertices).
    ax (optional)
        Plot onto the given preinitialised Matplotlib axes.

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if "ax" not in kwargs:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        ax = kwargs["ax"]

    ax.plot_trisurf(m.p[0], m.p[1], z,
                    triangles=m.t.T,
                    cmap=plt.cm.viridis,
                    antialiased=False)

    return ax


@plot3.register(InteriorBasis)
def _(basis: InteriorBasis, z: ndarray, **kwargs) -> Axes:
    """Plot on a refined mesh via :meth:`InteriorBasis.refinterp`."""
    Nrefs = kwargs["Nrefs"] if "Nrefs" in kwargs else 1
    return plot3(*basis.refinterp(z, Nrefs=Nrefs), **kwargs)


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)


def show(*args, **kwargs):
    plt.show(*args, **kwargs)
