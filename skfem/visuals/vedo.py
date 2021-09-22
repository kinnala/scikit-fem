import tempfile


def draw(m, backend=False, **kwargs):
    """Visualize meshes."""
    import vedo
    vedo.embedWindow(backend)
    from vedo import Plotter
    vp = Plotter()
    tetmesh = None
    with tempfile.NamedTemporaryFile() as tmp:
        m.save(tmp.name + '.vtk',
               encode_cell_data=False,
               encode_point_data=True,
               **kwargs)
        tetmesh = vp.load(tmp.name + '.vtk')
        # save these for further use
        tetmesh.show = lambda: vp.show([tetmesh]).close()
        tetmesh.plotter = vp
    return tetmesh
