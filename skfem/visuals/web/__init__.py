"""Interactive visualization in the browser.

The plots are rendered by a small dependency-free JavaScript viewer
(``viewer.js``, shipped with scikit-fem) using WebGL2, or a software
rasterizer on a 2D canvas if WebGL2 is not available.  The same viewer works
in Jupyter, in standalone HTML files, in the browser from an interactive
Python session and in single-page web apps (e.g. running scikit-fem in
Pyodide).

Examples
--------
>>> from skfem import MeshTet, Basis, ElementTetP1
>>> m = MeshTet().refined(2)
>>> basis = Basis(m, ElementTetP1())
>>> p = basis.plot(m.p[0], 'web')  # displays in Jupyter
>>> p.save('plot.html')            # doctest: +SKIP
>>> p.show()                       # doctest: +SKIP

"""

import html as _html
import json
import logging
import os
import tempfile
import uuid
import webbrowser
from importlib import resources
from typing import Any, Dict, Optional

import numpy as np

from ..data import VisData, sample


logger = logging.getLogger(__name__)

# options passed through to the JavaScript viewer
VIEWER_OPTIONS = (
    'field', 'component', 'cmap', 'bands', 'edges', 'warp', 'warp_scale',
    'deform', 'deform_scale', 'clip', 'clip_position', 'clip_flip', 'view',
    'vmin', 'vmax', 'title', 'colorbar', 'toolbar', 'info', 'hint', 'theme',
    'renderer',
)

# above this size a warning is issued
WARN_BYTES = 20 * 1024 * 1024

# above this number of sub-cells, three-dimensional plots by default include
# only elements touching the boundary
MAX_CELLS_3D = 400_000


def viewer_js() -> str:
    """Return the source code of the JavaScript viewer.

    The script defines ``window.SkfemViewer`` with the function
    ``mount(element, data, options)``, where ``data`` is the output of
    :meth:`~skfem.visuals.data.VisData.to_dict`.  It returns a handle with
    the methods ``update(data, options)``, ``png()`` and ``destroy()``.

    """
    return (resources.files(__package__)
            .joinpath('viewer.js')
            .read_text(encoding='utf-8'))


def _script_safe(s: str) -> str:
    # make JSON safe for inclusion inside a <script> element
    return (s.replace('<', '\\u003c')
            .replace('>', '\\u003e')
            .replace('&', '\\u0026'))


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError("Object of type {} is not JSON serializable"
                    .format(type(x)))


class WebPlot:
    """An interactive plot rendered by the web viewer.

    Displays itself in Jupyter (``_repr_html_``); use :meth:`show` to open
    it in a browser, :meth:`save` to write a standalone HTML file, or
    :meth:`fragment`, :meth:`to_json` and :func:`viewer_js` to embed it into
    a web page.

    """

    def __init__(self,
                 data: VisData,
                 options: Optional[Dict[str, Any]] = None,
                 height: int = 480):
        self.data = data
        self.options = dict(options or {})
        self.height = int(height)
        self._payload: Optional[Dict[str, Any]] = None

    @property
    def payload(self) -> Dict[str, Any]:
        """The data passed to the viewer (see :meth:`VisData.to_dict`)."""
        if self._payload is None:
            self._payload = self.data.to_dict()
        return self._payload

    def to_json(self) -> str:
        """JSON with keys ``data`` and ``options``.

        ``SkfemViewer.mount(element, JSON.parse(s))`` accepts it as is.

        """
        return json.dumps({'data': self.payload, 'options': self.options},
                          default=_jsonable)

    def _check_size(self, s: str) -> None:
        if len(s) > WARN_BYTES:
            logger.warning("The plot is {:.0f} MB; consider a smaller nrefs "
                           "or plotting a subset of the elements."
                           .format(len(s) / 2 ** 20))

    def fragment(self, element_id: Optional[str] = None,
                 include_viewer: bool = True) -> str:
        """HTML snippet for embedding into an existing page.

        Parameters
        ----------
        element_id
            The id of the created ``div``.
        include_viewer
            If ``False``, the page must load :func:`viewer_js` itself.

        """
        element_id = element_id or 'skfem-' + uuid.uuid4().hex[:12]
        data = _script_safe(self.to_json())
        self._check_size(data)
        parts = ['<div id="{}" style="width:100%;height:{}px"></div>'
                 .format(element_id, self.height)]
        if include_viewer:
            parts.append('<script>\n{}\n</script>'.format(viewer_js()))
        parts.append(
            '<script type="application/json" id="{0}-data">{1}</script>\n'
            '<script>(function(){{var s=JSON.parse(document.getElementById('
            '"{0}-data").textContent);SkfemViewer.mount(document.'
            'getElementById("{0}"),s.data,s.options);}})();</script>'
            .format(element_id, data))
        return '\n'.join(parts)

    def html(self, title: Optional[str] = None) -> str:
        """Standalone HTML document filling the browser window."""
        title = title or self.options.get('title') or 'scikit-fem'
        data = _script_safe(self.to_json())
        self._check_size(data)
        return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; --page-bg: #ffffff; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --page-bg: #14171c; }}
}}
:root[data-theme="dark"] {{ --page-bg: #14171c; }}
html, body {{ margin: 0; height: 100%; background: var(--page-bg); }}
#skfem {{ position: fixed; inset: 0; }}
#skfem > .skv {{ border: 0; border-radius: 0; }}
</style>
</head>
<body>
<div id="skfem"></div>
<script>
{viewer}
</script>
<script type="application/json" id="skfem-data">{data}</script>
<script>
(function () {{
  var s = JSON.parse(document.getElementById('skfem-data').textContent);
  window.skfemPlot = SkfemViewer.mount(document.getElementById('skfem'),
                                       s.data, s.options);
}})();
</script>
</body>
</html>
""".format(title=_html.escape(title), viewer=viewer_js(), data=data)

    def _repr_html_(self) -> str:
        # an iframe isolates the plot from the notebook and from other plots
        return ('<iframe srcdoc="{}" style="width:100%;height:{}px;border:0" '
                'sandbox="allow-scripts allow-downloads" '
                'loading="lazy"></iframe>'
                .format(_html.escape(self.html(), quote=True),
                        self.height + 2))

    def save(self, filename) -> None:
        """Write a standalone HTML file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.html())

    def show(self, filename: Optional[str] = None) -> str:
        """Open the plot in a web browser and return the file name.

        Parameters
        ----------
        filename
            Optional output file; a temporary file is used by default.

        """
        if filename is None:
            fd, filename = tempfile.mkstemp(prefix='skfem-', suffix='.html')
            os.close(fd)
        self.save(filename)
        webbrowser.open('file://' + os.path.abspath(filename))
        return filename

    def __repr__(self) -> str:
        d = self.data
        return ("<WebPlot: {} {} cells, {} points, fields {}>"
                .format(d.ncells, d.cell_type, d.npoints, d.fields))


def _boundary_elements(mesh) -> np.ndarray:
    return np.unique(mesh.f2t[0, mesh.boundary_facets()])


def plot(obj, x: Any = None, **kwargs) -> WebPlot:
    """Interactive plot of a mesh and fields.

    Parameters
    ----------
    obj
        A :class:`~skfem.mesh.Mesh`, a :class:`~skfem.assembly.CellBasis` or
        :class:`~skfem.visuals.data.VisData`.
    x
        The field(s); see :func:`~skfem.visuals.data.sample`.  Use a dict to
        plot several fields, e.g. ``{'u': u, 'grad u': (u, lambda w:
        w.grad)}``; the viewer has a menu for switching between them.
    nrefs (optional)
        Number of refinements of the reference element; by default based on
        the polynomial degree.
    elements (optional)
        Plot only a subset of elements.
    boundary_only (optional)
        Three-dimensional meshes: include only elements touching the
        boundary.  Reduces the size but disables clipping.  Default is
        ``True`` for very large meshes.
    height (optional)
        Height in pixels for Jupyter and :meth:`WebPlot.fragment`.
    field, component (optional)
        Initially visible field and its component (an integer index or
        ``'magnitude'``).
    cmap (optional)
        One of ``'viridis'`` (default), ``'turbo'``, ``'jet'``,
        ``'coolwarm'``, ``'magma'``, ``'plasma'``, ``'grey'``.
    bands (optional)
        Number of discrete color bands; ``0`` for smooth.
    vmin, vmax (optional)
        Color range.
    edges (optional)
        Draw element edges; ``True`` by default.
    warp (optional)
        Two-dimensional scalar fields: elevate the surface by the field.
        ``True`` or a relative height.
    deform (optional)
        Displace the geometry by a vector field: ``True`` for the first
        vector field or a field name.  ``deform_scale`` sets the factor.
    clip, clip_position, clip_flip (optional)
        Three-dimensional meshes: initial clip plane normal (``'x'``,
        ``'y'`` or ``'z'``) and relative position in ``[0, 1]``.
    view (optional)
        Initial view: ``'xy'``, ``'xz'``, ``'yz'`` or ``'iso'``.
    title (optional)
        Title shown in the corner.
    theme (optional)
        ``'light'`` or ``'dark'``; follows the browser by default.
    renderer (optional)
        ``'auto'`` (WebGL2 if available) or ``'canvas'``.

    Returns
    -------
    WebPlot

    """
    height = kwargs.pop('height', 480)
    options = {k: kwargs.pop(k) for k in list(kwargs)
               if k in VIEWER_OPTIONS}
    if isinstance(options.get('component'), str) \
            and options['component'] in ('mag', 'magnitude'):
        options['component'] = 'mag'
    if isinstance(obj, VisData):
        if x is not None:
            raise ValueError("VisData already contains the fields.")
        data = obj
    else:
        nrefs = kwargs.pop('nrefs', kwargs.pop('Nrefs', None))
        elements = kwargs.pop('elements', None)
        boundary_only = kwargs.pop('boundary_only', None)
        mesh = obj.mesh if hasattr(obj, 'mesh') else (
            obj.bases[0].mesh if hasattr(obj, 'bases') else obj)
        no_clip = False
        if mesh.dim() == 3 and elements is None:
            if boundary_only is None:
                # estimate the number of sub-cells using one element
                probe = sample(obj, x, nrefs=nrefs, elements=[0])
                if mesh.nelements * probe.ncells > MAX_CELLS_3D:
                    logger.warning("Large mesh: plotting only the elements "
                                   "touching the boundary; use "
                                   "boundary_only=False to plot all.")
                    boundary_only = True
            if boundary_only:
                elements = _boundary_elements(mesh)
                no_clip = True
        if kwargs:
            logger.warning("Unused arguments: {}".format(', '.join(kwargs)))
        data = sample(obj, x, nrefs=nrefs, elements=elements)
        if no_clip:
            data.meta['no_clip'] = True
    return WebPlot(data, options, height=height)


def plot3(obj, x: Any = None, **kwargs) -> WebPlot:
    """Plot with the field as elevation (two-dimensional meshes)."""
    kwargs.setdefault('warp', True)
    return plot(obj, x, **kwargs)


def draw(obj, **kwargs) -> WebPlot:
    """Interactive plot of a mesh."""
    return plot(obj, None, **kwargs)
