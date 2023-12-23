# type: ignore
"""Drawing meshes and solutions using matplotlib."""

from functools import singledispatch

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

from ..assembly import CellBasis
from ..mesh import Mesh2D, MeshLine1, MeshQuad1, MeshTri1, Mesh3D


@singledispatch
def draw(m, **kwargs) -> Axes:
    """Visualize meshes."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@draw.register(CellBasis)
def draw_basis(ib: CellBasis, **kwargs) -> Axes:
    if "nrefs" in kwargs:
        nrefs = kwargs["nrefs"]
    elif "Nrefs" in kwargs:
        nrefs = kwargs["Nrefs"]
    else:
        nrefs = 1
    m, _ = ib.refinterp(ib.mesh.p[0], nrefs=nrefs)
    return draw(m, boundaries_only=True, **kwargs)


@draw.register(Mesh3D)
def draw_mesh3d(m: Mesh3D, **kwargs) -> Axes:
    """Visualize a three-dimensional mesh by drawing the edges."""
    if 'ax' not in kwargs:
        ax = plt.figure().add_subplot(1, 1, 1, projection='3d')
    else:
        ax = kwargs['ax']
    for ix in m.boundary_edges():
        ax.plot3D(
            m.p[0, m.edges[:, ix]].flatten(),
            m.p[1, m.edges[:, ix]].flatten(),
            m.p[2, m.edges[:, ix]].flatten(),
            kwargs['color'] if 'color' in kwargs else 'k',
            linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else .5,
        )
    ax.set_axis_off()
    ax.show = lambda: plt.show()
    return ax


@draw.register(Mesh2D)
def draw_mesh2d(m: Mesh2D, **kwargs) -> Axes:
    """Visualise a two-dimensional mesh by drawing the edges.

    Parameters
    ----------
    m
        A two-dimensional mesh.
    ax (optional)
        A preinitialised Matplotlib axes for plotting.
    node_numbering (optional)
        If ``True``, draw node numbering.
    facet_numbering (optional)
        If ``True``, draw facet numbering.
    element_numbering (optional)
        If ``True``, draw element numbering.
    aspect (optional)
        Ratio of vertical to horizontal length-scales; ignored if ``ax`` is
        specified
    boundaries_only (optional)
        If ``True``, draw only boundary edges.

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
    if "boundaries_only" in kwargs:
        facets = m.facets[:, m.boundary_facets()]
    else:
        facets = m.facets
    plot_kwargs = kwargs["plot_kwargs"] if "plot_kwargs" in kwargs else {}
    # faster plotting is achieved through
    # None insertion trick.
    xs = []
    ys = []
    for s, t, u, v in zip(m.p[0, facets[0]],
                          m.p[1, facets[0]],
                          m.p[0, facets[1]],
                          m.p[1, facets[1]]):
        xs.append(s)
        xs.append(u)
        xs.append(None)
        ys.append(t)
        ys.append(v)
        ys.append(None)
    ax.plot(xs,
            ys,
            kwargs['color'] if 'color' in kwargs else 'k',
            linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else .5,
            **plot_kwargs)

    if "subdomain" in kwargs:
        y = np.zeros(m.t.shape[1])
        y[m.subdomains[kwargs['subdomain']]] = 1
        plot(m, y, ax=ax)

    if "boundaries" in kwargs:
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1.*i/len(m.boundaries)) for i in range(len(m.boundaries))]
        for i, k in enumerate(m.boundaries):
            facets = m.facets[:, m.boundaries[k]]
            xs = []
            ys = []
            for s, t, u, v in zip(m.p[0, facets[0]],
                                  m.p[1, facets[0]],
                                  m.p[0, facets[1]],
                                  m.p[1, facets[1]]):
                xs.append(s)
                xs.append(u)
                xs.append(None)
                ys.append(t)
                ys.append(v)
                ys.append(None)
            ax.plot(xs,
                    ys,
                    color=colors[i % len(colors)],
                    linewidth=(kwargs['linewidth']
                               if 'linewidth' in kwargs else 2.),
                    **plot_kwargs)
            if hasattr(m.boundaries[k], 'ori'):
                tris = m.f2t[m.boundaries[k].ori, m.boundaries[k]]
                color = colors[i % len(colors)][:3] + (.1,)
                collec = PolyCollection(m.p[:, m.t[:, tris]].T,
                                        color=color)
                ax.add_collection(collec)

    if "node_numbering" in kwargs:
        for itr in range(m.p.shape[1]):
            ax.text(m.p[0, itr], m.p[1, itr], str(itr))

    if "facet_numbering" in kwargs:
        mx = .5*(m.p[0, m.facets[0]] +
                 m.p[0, m.facets[1]])
        my = .5*(m.p[1, m.facets[0]] +
                 m.p[1, m.facets[1]])
        for itr in range(m.facets.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    if "element_numbering" in kwargs:
        mx = np.sum(m.p[0, m.t], axis=0) / m.t.shape[0]
        my = np.sum(m.p[1, m.t], axis=0) / m.t.shape[0]
        for itr in range(m.t.shape[1]):
            ax.text(mx[itr], my[itr], str(itr))

    ax.show = lambda: plt.show()
    return ax


@draw.register(MeshLine1)
def draw_meshline(m: MeshLine1, **kwargs):
    """Draw the nodes of one-dimensional mesh."""
    if "ax" not in kwargs:
        # create new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = kwargs["ax"]

    color = kwargs["color"] if "color" in kwargs else 'ko-'
    ix = np.argsort(m.p[0])

    plot_kwargs = kwargs["plot_kwargs"] if "plot_kwargs" in kwargs else {}

    ax.plot(m.p[0][ix], 0. * m.p[0][ix], color, **plot_kwargs)

    ax.show = lambda: plt.show()
    return ax


@singledispatch
def plot(m, u, **kwargs) -> Axes:
    """Plot functions defined on nodes of the mesh."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot.register(MeshLine1)
def plot_meshline(m: MeshLine1, z: ndarray, **kwargs):
    """Plot a function defined at the nodes of the 1D mesh."""
    if "ax" not in kwargs:
        # create new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = kwargs["ax"]

    color = kwargs["color"] if "color" in kwargs else 'ko-'
    plot_kwargs = kwargs["plot_kwargs"] if "plot_kwargs" in kwargs else {}
    ix = np.argsort(m.p[0])
    ax.plot(m.p[0][ix], z[ix], color, **plot_kwargs)
    ax.show = lambda: plt.show()
    return ax


@plot.register(MeshTri1)
def plot_meshtri(m: MeshTri1, z: ndarray, **kwargs) -> Axes:
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
        If True, show colorbar. If a string, use it as a label for the
        colorbar. Not shown by default.
    figsize (optional)
        Passed on to matplotlib.
    shading (optional)
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

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = plt.cm.jet

    plot_kwargs = kwargs["plot_kwargs"] if "plot_kwargs" in kwargs else {}

    if len(z) == 2 * len(m.p[0]):
        im = ax.quiver(m.p[0], m.p[1], *z.reshape(2, -1),
                       **{k: v for k, v in kwargs.items()
                          if k in ['angles',
                                   'scale',
                                   'width',
                                   'headwidth',
                                   'headlength',
                                   'minshaft',
                                   'pivot',
                                   'color']},  # for backwards compatibility
                       **plot_kwargs)
    else:
        im = ax.tripcolor(m.p[0], m.p[1], m.t.T, z, cmap=cmap,
                          **{k: v for k, v in kwargs.items()
                             if k in ['shading',
                                      'edgecolors',
                                      'vmin',
                                      'vmax']},  # for backwards compatibility
                          **plot_kwargs)

    if "levels" in kwargs:
        ax.tricontour(m.p[0], m.p[1], m.t.T, z,
                      levels=kwargs["levels"],
                      **{**{'colors': 'k'}, **plot_kwargs})

    if "colorbar" in kwargs and kwargs["colorbar"] is not False:
        if isinstance(kwargs["colorbar"], str):
            plt.colorbar(im, ax=ax, label=kwargs["colorbar"])
        elif isinstance(kwargs["colorbar"], dict):
            plt.colorbar(im, ax=ax, **kwargs["colorbar"])
        else:
            plt.colorbar(im, ax=ax)

    ax.show = lambda: plt.show()
    return ax


@plot.register(MeshQuad1)
def plot_meshquad(m: MeshQuad1, z, **kwargs):
    """Visualise nodal functions on quadrilateral meshes.

    The quadrilaterals are split into two triangles
    (:class:`skfem.mesh.MeshTri`) and the respective plotting function for the
    triangular mesh is used.

    """
    if len(z) == m.t.shape[-1]:
        m, z = m.to_meshtri(z)
    else:
        m = m.to_meshtri()
    return plot(m, z, **kwargs)


@plot.register(CellBasis)
def plot_basis(basis: CellBasis, z: ndarray, **kwargs) -> Axes:
    """Plot on a refined mesh via :meth:`CellBasis.refinterp`."""
    if "nrefs" in kwargs:
        nrefs = kwargs["nrefs"]
    elif "Nrefs" in kwargs:
        nrefs = kwargs["Nrefs"]
    else:
        nrefs = 1
    return plot(*basis.refinterp(z, nrefs=nrefs), **kwargs)


@singledispatch
def plot3(m, z: ndarray, **kwargs) -> Axes:
    """Plot functions defined on nodes of the mesh (3D)."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot3.register(MeshTri1)
def plot3_meshtri(m: MeshTri1, z: ndarray, **kwargs) -> Axes:
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
    ax = kwargs.get("ax", plt.figure().add_subplot(1, 1, 1, projection='3d'))
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = plt.cm.jet

    ax.plot_trisurf(m.p[0], m.p[1], z,
                    triangles=m.t.T,
                    cmap=cmap,
                    antialiased=False)

    ax.show = lambda: plt.show()
    return ax


@plot3.register(CellBasis)
def plot3_basis(basis: CellBasis, z: ndarray, **kwargs) -> Axes:
    """Plot on a refined mesh via :meth:`CellBasis.refinterp`."""
    if "nrefs" in kwargs:
        nrefs = kwargs["nrefs"]
    elif "Nrefs" in kwargs:
        nrefs = kwargs["Nrefs"]
    else:
        nrefs = 1
    return plot3(*basis.refinterp(z, nrefs=nrefs), **kwargs)


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)


def show(*args, **kwargs):
    plt.show(*args, **kwargs)
