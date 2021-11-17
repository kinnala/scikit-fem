class Mesh2D2:
    """Mixin for quadratic 2D meshes."""

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True).svg

    def element_finder(self, *args, **kwargs):
        raise NotImplementedError
