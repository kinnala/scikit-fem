import tempfile


def draw(m, backend=False, **kwargs):
    """Visualize meshes."""
    import vedo
    vedo.embedWindow(backend)
    from vedo import Plotter, UGrid
    vp = Plotter()
    grid = None
    with tempfile.NamedTemporaryFile() as tmp:
        m.save(tmp.name + '.vtk',
               encode_cell_data=False,
               encode_point_data=True,
               **kwargs)
        grid = UGrid(tmp.name + '.vtk')
        # save these for further use
        grid.show = lambda: vp.show([grid.tomesh()]).close()
        grid.plotter = vp
    return grid
