import tempfile

from vedo import Plotter


def draw(m, **kwargs):
    """Visualize meshes."""
    vp = Plotter()
    plot = None
    with tempfile.NamedTemporaryFile() as tmp:
        m.save(tmp.name + '.vtk', **kwargs)
        plot = vp.load(tmp.name + '.vtk')
    plot.show = lambda: vp.show([plot])
    return plot
