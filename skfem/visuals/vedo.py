"""Visualization using vedo (VTK)."""

import tempfile

from .data import sample


def draw(m, backend=False, **kwargs):
    """Visualize meshes."""
    from vedo import Plotter, UnstructuredGrid
    vp = Plotter()
    grid = None
    with tempfile.NamedTemporaryFile() as tmp:
        m.save(tmp.name + '.vtk',
               encode_cell_data=False,
               encode_point_data=True,
               **kwargs)
        grid = UnstructuredGrid(tmp.name + '.vtk')
        vp += grid.tomesh()
        # save these for further use
        grid.show = lambda: vp.show()
        grid.plotter = vp
    return grid


def plot(basis, z, **kwargs):
    """Plot fields sampled by :func:`skfem.visuals.data.sample`."""
    data = sample(basis, z, nrefs=kwargs.get("nrefs", 1))
    point_data = {k: v for k, v in data.point_data.items()}
    if set(point_data) == {'u'}:
        point_data = {'z': point_data['u']}  # backwards compatibility
    return draw(data.to_mesh(), point_data=point_data)
