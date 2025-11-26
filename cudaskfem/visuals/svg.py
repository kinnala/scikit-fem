"""Drawing meshes using svg."""

import webbrowser
from functools import singledispatch
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
from numpy import ndarray

from ..assembly import CellBasis
from ..mesh import Mesh2D


def points_to_figure(p, kwargs):
    """Map points to figure coordinates and find figure size."""
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
    sx = (width - 2 * stroke) / (maxx - minx)
    sy = (height - 2 * stroke) / (maxy - miny)
    p[0] = sx * (p[0] - minx) + stroke
    p[1] = sy * (maxy - p[1]) + stroke
    return p, width, height, stroke


class Server(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(self.svg.encode("utf8"))


@dataclass
class SvgPlot:

    svg: str = ""

    def _repr_svg_(self) -> str:
        return self.svg

    def show(self, port=8000):
        server = Server
        server.svg = self.svg
        url = "http://localhost:{}".format(port)
        print("Serving the plot at " + url)
        webbrowser.open_new_tab(url)
        HTTPServer(("localhost", port), Server).handle_request()


@singledispatch
def draw(m, **kwargs) -> SvgPlot:
    """Visualize meshes by drawing the edges.

    Parameters
    ----------
    m
        A mesh object.

    Returns
    -------
    SvgPlot
        A dataclass with the attribute ``svg``.  Gets automatically visualized
        in Jupyter via ``_repr_svg_``.

    """
    raise NotImplementedError("Type {} not supported.".format(type(m)))


def draw_mesh2d(m: Mesh2D, **kwargs) -> SvgPlot:
    """Support for two-dimensional meshes."""
    if "boundaries_only" in kwargs:
        facets = m.facets[:, m.boundary_facets()]
    else:
        facets = m.facets
    p, width, height, stroke = points_to_figure(m.p.copy(), kwargs)
    color = kwargs["color"] if "color" in kwargs else "#7856FA"
    template = ("""<line x1="{}" y1="{}" x2="{}" y2="{}" """
                """style="stroke:{};stroke-width:{}"/>""")
    lines = ""
    for s, t, u, v in zip(p[0, facets[0]],
                          p[1, facets[0]],
                          p[0, facets[1]],
                          p[1, facets[1]]):
        lines += template.format(s, t, u, v, color, stroke)
    return SvgPlot((
        """<svg xmlns="http://www.w3.org/2000/svg" version="1.1" """
        """width="{}" height="{}">{}</svg>"""
    ).format(width, height, lines))


@draw.register(Mesh2D)
def draw_geometry2d(m: Mesh2D, **kwargs) -> SvgPlot:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 0
    m = m._splitref(nrefs)
    return draw_mesh2d(m, **kwargs)


@draw.register(CellBasis)
def draw_basis(ib: CellBasis, **kwargs) -> SvgPlot:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 2
    m, _ = ib.refinterp(ib.mesh.p[0], nrefs=nrefs)
    return draw(m, boundaries_only=True, **kwargs)


@singledispatch
def plot(m, x: ndarray, **kwargs) -> SvgPlot:
    """Visualize discrete solutions: one value per node.

    Parameters
    ----------
    m
        A mesh object.
    x
        One value per node of the mesh.

    Returns
    -------
    SvgPlot
        A dataclass with the attribute ``svg``.  Gets automatically visualized
        in Jupyter via ``_repr_svg_``.

    """
    raise NotImplementedError("Type {} not supported.".format(type(m)))


COLORS = np.array([
    [49, 54, 149],
    [69, 117, 180],
    [116, 173, 209],
    [171, 217, 233],
    [224, 243, 248],
    [255, 255, 191],
    [254, 224, 144],
    [253, 174, 97],
    [244, 109, 67],
    [215, 48, 39],
    [165, 0, 38],
])


@plot.register(Mesh2D)
def plot_mesh2d(m: Mesh2D, x: ndarray, **kwargs) -> SvgPlot:
    t = m.t
    minval = np.min(x)
    maxval = np.max(x)
    p, width, height, stroke = points_to_figure(m.p.copy(), kwargs)
    template = ("""<polygon points=""" +
                '"' + ("{},{} " * t.shape[0]) + '"' +
                """style="fill:rgb({}, {}, {});" />""")
    elems = ""
    for ix, e in enumerate(t.T):
        scalar = (x[e].mean() - minval) / (maxval - minval) * (len(COLORS) - 1)
        ix1 = int(np.floor(scalar))
        ix2 = int(np.ceil(scalar))
        color = (COLORS[ix2] - COLORS[ix1]) * (scalar - ix1) + COLORS[ix1]
        elems += template.format(*p[:, e].flatten(order='F'), *color)
    elems += draw_mesh2d(m, boundaries_only=True, color="black").svg
    return SvgPlot((
        """<svg xmlns="http://www.w3.org/2000/svg" version="1.1" """
        """width="{}" height="{}" shape-rendering="crispEdges">"""
        """{}"""
        """</svg>"""
    ).format(width + 30,
             height,
             elems))


@plot.register(CellBasis)
def plot_basis(ib: CellBasis, x: ndarray, **kwargs) -> SvgPlot:
    nrefs = kwargs["nrefs"] if "nrefs" in kwargs else 0
    m, X = ib.refinterp(x, nrefs=nrefs)
    return plot(m, X, **kwargs)
