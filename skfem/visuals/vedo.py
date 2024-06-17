import tempfile


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
    nrefs = kwargs["nrefs"] if 'nrefs' in kwargs else 1
    m, Z = basis.refinterp(z, nrefs=nrefs)
    return draw(m, point_data={'z': Z})
