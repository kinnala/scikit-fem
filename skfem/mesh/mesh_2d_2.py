class Mesh2D2:
    """Mixin for quadratic 2D meshes."""

    def plot(self, *args, **kwargs):
        """Convenience wrapper for :func:`skfem.visuals.matplotlib.plot`."""
        from skfem.visuals.matplotlib import plot, show
        from skfem.assembly import CellBasis
        ax = plot(CellBasis(self, self.elem()), *args, **kwargs)
        ax.show = show
        return ax

    def draw(self, *args, **kwargs):
        """Convenience wrapper for :func:`skfem.visuals.matplotlib.draw`."""
        from skfem.visuals.matplotlib import draw, show
        from skfem.assembly import CellBasis
        ax = draw(CellBasis(self, self.elem()), *args, **kwargs)
        ax.show = show
        return ax

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True).svg

    def element_finder(self, *args, **kwargs):
        raise NotImplementedError
