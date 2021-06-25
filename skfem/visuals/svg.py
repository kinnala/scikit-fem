"""Drawing meshes using svg."""

from functools import singledispatch

import numpy as np
from numpy import ndarray

from ..assembly import InteriorBasis
from ..mesh import Mesh2D, MeshTri


@singledispatch
def draw(m, **kwargs) -> str:
    """Visualise meshes by drawing the edges.

    Parameters
    ----------
    m
        A mesh object.

    Returns
    -------
    string
        The svg xml source as a string.

    """
    raise NotImplementedError("Type {} not supported.".format(type(m)))


def draw_mesh2d(m: Mesh2D, **kwargs) -> str:
    """Support for two-dimensional meshes."""
    if "boundaries_only" in kwargs:
        facets = m.facets[:, m.boundary_facets()]
    else:
        facets = m.facets
    p = m.p.copy()
    maxx = np.max(p[0])
    minx = np.min(p[0])
    maxy = np.max(p[1])
    miny = np.min(p[1])
    width = kwargs["width"] if "width" in kwargs else 300
    if "height" in kwargs:
        height = kwargs["height"]
    else:
        height = width * (maxy - miny) / (maxx - minx)
    stroke = kwargs["stroke"] if "stroke" in kwargs else 1
    color = kwargs["color"] if "color" in kwargs else "#7856FA"
    sx = (width - 2 * stroke) / (maxx - minx)
    sy = (height - 2 * stroke) / (maxy - miny)
    p[0] = sx * (p[0] - minx) + stroke
    p[1] = sy * (maxy - p[1]) + stroke
    template = ("""<line x1="{}" y1="{}" x2="{}" y2="{}" """
                """style="stroke:{};stroke-width:{}"/>""")
    lines = ""
    for s, t, u, v in zip(p[0, facets[0]],
                          p[1, facets[0]],
                          p[0, facets[1]],
                          p[1, facets[1]]):
        lines += template.format(s, t, u, v, color, stroke)
    return ("""<svg xmlns="http://www.w3.org/2000/svg" version="1.1" """
            """width="{}" height="{}">{}</svg>""").format(width, height, lines)


@draw.register(Mesh2D)
def draw_geometry2d(m: Mesh2D, **kwargs) -> str:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 1
    m = m._splitref(nrefs)
    return draw_mesh2d(m, **kwargs)


@draw.register(InteriorBasis)
def draw_basis(ib: InteriorBasis, **kwargs) -> str:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 2
    m, _ = ib.refinterp(ib.mesh.p[0], nrefs=nrefs)
    return draw(m, boundaries_only=True, **kwargs)


@singledispatch
def plot(m, x: ndarray, **kwargs) -> None:
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot.register(MeshTri)
def plot_mesh_tri(m: MeshTri, x: ndarray, **kwargs) -> None:
    t = m.t
    p = m.p.copy()
    maxx = np.max(p[0])
    minx = np.min(p[0])
    maxy = np.max(p[1])
    miny = np.min(p[1])
    maxval = np.max(x)
    minval = np.min(x)
    width = kwargs["width"] if "width" in kwargs else 300
    if "height" in kwargs:
        height = kwargs["height"]
    else:
        height = width * (maxy - miny) / (maxx - minx)
    stroke = kwargs["stroke"] if "stroke" in kwargs else 1
    sx = (width - 2 * stroke) / (maxx - minx)
    sy = (height - 2 * stroke) / (maxy - miny)
    p[0] = sx * (p[0] - minx) + stroke
    p[1] = sy * (maxy - p[1]) + stroke
    template = ("""<polygon points="{},{} {},{} {},{}" """
                """style="fill:rgb(100, {}, {});" />""")
    elems = ""
    for ix, tri in enumerate(zip(p[0, t[0]],
                                 p[1, t[0]],
                                 p[0, t[1]],
                                 p[1, t[1]],
                                 p[0, t[2]],
                                 p[1, t[2]])):
        color = int((x[t[:, ix]].mean() - minval) / (maxval - minval) * 255)
        elems += template.format(*tri, color, 255 - color)
    elems += draw_mesh2d(m, boundaries_only=True, color='black')
    return ("""<svg xmlns="http://www.w3.org/2000/svg" version="1.1" """
            """width="{}" height="{}" shape-rendering="crispEdges">"""
            """<defs>"""
            """<linearGradient id="cbar" x1="0%" y1="0%" x2="0%" y2="100%">"""
            """<stop offset="0%" style="stop-color:rgb(100,255,0);" />"""
            """<stop offset="100%" style="stop-color:rgb(100,0,255);" />"""
            """</linearGradient>"""
            """</defs>"""
            """{}"""
            """<polygon points="{},{} {},{} {},{} {},{}" """
            """fill="url(#cbar)" style="stroke:black;stroke-width:{}" />"""
            """</svg>""").format(width + 30,
                                 height,
                                 elems,
                                 width + 29,
                                 height - 1,
                                 width + 29,
                                 1,
                                 width + 10,
                                 1,
                                 width + 10,
                                 height - 1,
                                 stroke)


@plot.register(InteriorBasis)
def plot_basis(ib: InteriorBasis, x: ndarray, **kwargs) -> str:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 0
    m, X = ib.refinterp(x, nrefs=nrefs)
    return plot(m, X, **kwargs)
