class Mesh2D2:
    """Mixin for quadratic 2D meshes."""

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True).svg

    def element_finder(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def init_refdom(cls):
        return cls.__bases__[-1].init_refdom()

    def draw(self, *args, **kwargs):
        from ..assembly import CellBasis
        return CellBasis(self, self.elem()).draw(*args, **kwargs)
