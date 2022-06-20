import logging
import importlib
from warnings import warn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import ndarray
from skfem.assembly.dofs import Dofs, DofsView
from skfem.element import (DiscreteField, Element, ElementComposite,
                           ElementVector)
from skfem.mapping import Mapping
from skfem.mesh import Mesh
from skfem.quadrature import get_quadrature
from skfem.refdom import Refdom
from skfem.generic_utils import deprecated


logger = logging.getLogger(__name__)


class AbstractBasis:
    """Finite element basis at global quadrature points.

    Please see the following implementations:

    - :class:`~skfem.assembly.CellBasis`, basis functions inside elements
    - :class:`~skfem.assembly.FacetBasis`, basis functions on boundary
    - :class:`~skfem.assembly.InteriorFacetBasis`, basis functions on facets
      inside the domain

    """

    mesh: Mesh
    elem: Element
    mapping: Mapping
    tind: Optional[ndarray] = None
    tind_normals: Optional[ndarray] = None
    dx: ndarray
    basis: List[Tuple[DiscreteField, ...]] = []
    X: ndarray
    W: ndarray
    dofs: Dofs

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 refdom: Type[Refdom] = Refdom,
                 dofs: Optional[Dofs] = None):

        if mesh.refdom != elem.refdom:
            raise ValueError("Incompatible Mesh and Element.")

        self.mapping = mesh._mapping() if mapping is None else mapping
        self.dofs = Dofs(mesh, elem) if dofs is None else dofs

        # global degree-of-freedom location
        try:
            doflocs = self.mapping.F(elem.doflocs.T)
            self.doflocs = np.zeros((doflocs.shape[0], self.N))

            # match mapped dofs and global dof numbering
            for itr in range(doflocs.shape[0]):
                for jtr in range(self.dofs.element_dofs.shape[0]):
                    self.doflocs[itr, self.dofs.element_dofs[jtr]] =\
                        doflocs[itr, :, jtr]
        except Exception:
            logger.warning("Unable to calculate global DOF locations.")

        self.mesh = mesh
        self.elem = elem

        self.Nbfun = self.dofs.element_dofs.shape[0]

        self.nelems = 0  # subclasses should overwrite

        if quadrature is not None:
            self.X, self.W = quadrature
        else:
            self.X, self.W = get_quadrature(
                refdom,
                intorder if intorder is not None else 2 * self.elem.maxdeg
            )

    @property
    def nodal_dofs(self):
        return self.dofs.nodal_dofs

    @property
    def facet_dofs(self):
        return self.dofs.facet_dofs

    @property
    def edge_dofs(self):
        return self.dofs.edge_dofs

    @property
    def interior_dofs(self):
        return self.dofs.interior_dofs

    @property
    def N(self):
        return self.dofs.N

    @property
    def element_dofs(self):
        if not hasattr(self, '_element_dofs'):
            if self.tind is None:
                self._element_dofs = self.dofs.element_dofs
            else:
                self._element_dofs = self.dofs.element_dofs[:, self.tind]
        return self._element_dofs

    def complement_dofs(self, *D):
        if type(D[0]) is dict:
            # if a dict of Dofs objects are given, flatten all
            D = tuple(D[0][key].all() for key in D[0])
        return np.setdiff1d(np.arange(self.N), np.concatenate(D))

    @deprecated("Basis.get_dofs")
    def find_dofs(self,
                  facets: Dict[str, ndarray] = None,
                  skip: List[str] = None) -> Dict[str, DofsView]:
        if facets is None:
            if self.mesh.boundaries is None:
                facets = {'all': self.mesh.boundary_facets()}
            else:
                facets = self.mesh.boundaries

        return {k: self.dofs.get_facet_dofs(facets[k], skip_dofnames=skip)
                for k in facets}

    def get_dofs(self,
                 facets: Optional[Any] = None,
                 elements: Optional[Any] = None,
                 skip: List[str] = None) -> Any:
        """Find global DOF numbers.

        Accepts an array of facet/element indices.  However, various argument
        types can be turned into an array of facet/element indices.

        Get all boundary DOFs:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriP1
        >>> m = MeshTri().refined()
        >>> basis = Basis(m, ElementTriP1())
        >>> basis.get_dofs().flatten()
        array([0, 1, 2, 3, 4, 5, 7, 8])

        Get DOFs via a function query:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriP1
        >>> m = MeshTri().refined()
        >>> basis = Basis(m, ElementTriP1())
        >>> basis.get_dofs(lambda x: np.isclose(x[0], 0)).flatten()
        array([0, 2, 5])

        Get DOFs via named boundaries:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriP1
        >>> m = (MeshTri()
        ...      .refined()
        ...      .with_boundaries({'left': lambda x: np.isclose(x[0], 0)}))
        >>> basis = Basis(m, ElementTriP1())
        >>> basis.get_dofs('left').flatten()
        array([0, 2, 5])

        Get DOFs via named subdomains:

        >>> from skfem import MeshTri, Basis, ElementTriP1
        >>> m = (MeshTri()
        ...      .refined()
        ...      .with_subdomains({'left': lambda x: x[0] < .5}))
        >>> basis = Basis(m, ElementTriP1())
        >>> basis.get_dofs(elements='left').flatten()
        array([0, 2, 4, 5, 6, 8])

        Further reduce the set of DOFs:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriArgyris
        >>> m = MeshTri().refined()
        >>> basis = Basis(m, ElementTriArgyris())
        >>> basis.get_dofs(lambda x: np.isclose(x[0], 0)).nodal.keys()
        dict_keys(['u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy'])
        >>> basis.get_dofs(lambda x: np.isclose(x[0], 0)).all(['u', 'u_x'])
        array([ 0,  1, 12, 13, 30, 31])

        Skip some DOF names altogether:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriArgyris
        >>> m = MeshTri().refined()
        >>> basis = Basis(m, ElementTriArgyris())
        >>> basis.get_dofs(lambda x: np.isclose(x[0], 0),
        ...                skip=['u_x', 'u_y']).nodal.keys()
        dict_keys(['u', 'u_xx', 'u_xy', 'u_yy'])

        Combine several boundaries into one call:

        >>> import numpy as np
        >>> from skfem import MeshTri, Basis, ElementTriP1
        >>> m = (MeshTri()
        ...      .with_boundaries({'left': lambda x: np.isclose(x[0], 0),
        ...                        'right': lambda x: np.isclose(x[0], 1)}))
        >>> basis = Basis(m, ElementTriP1())
        >>> basis.get_dofs({'left', 'right'}).flatten()
        array([0, 1, 2, 3])

        Parameters
        ----------
        facets
            An array of facet indices.  If ``None``, find facets by
            ``self.mesh.boundary_facets()``.  If callable, call
            ``self.mesh.facets_satisfying(facets)`` to get the facets.  If
            string, use ``self.mesh.boundaries[facets]`` to get the facets.
            If list, tuple or set, use the combined facet indices.
        elements
            An array of element indices.  See above.
        skip
            List of dofnames to skip.

        """
        if isinstance(facets, dict):
            warn("Passing dict to get_dofs is deprecated.", DeprecationWarning)

            def to_indices(f):
                if callable(f):
                    return self.mesh.facets_satisfying(f)
                return f

            return {k: self.dofs.get_facet_dofs(to_indices(facets[k]),
                                                skip_dofnames=skip)
                    for k in facets}

        if elements is not None and facets is not None:
            raise ValueError

        if elements is not None:
            elements = self.mesh.normalize_elements(elements)
            return self.dofs.get_element_dofs(elements, skip_dofnames=skip)

        facets = self.mesh.normalize_facets(facets)
        return self.dofs.get_facet_dofs(facets, skip_dofnames=skip)

    def __repr__(self):
        size = sum([sum([y.size if hasattr(y, 'size') else 0
                         for y in x.astuple])
                    for x in self.basis[0]]) * 8 * len(self.basis)
        rep = ""
        rep += "<skfem {}({}, {}) object>\n".format(type(self).__name__,
                                                    type(self.mesh).__name__,
                                                    type(self.elem).__name__)
        rep += "  Number of elements: {}\n".format(self.nelems)
        rep += "  Number of DOFs: {}\n".format(self.N)
        rep += "  Size: {} B".format(size)
        return rep

    def __str__(self):
        return self.__repr__()

    def default_parameters(self):
        """This is used by :func:`skfem.assembly.asm` to get the default
        parameters for 'w'."""
        raise NotImplementedError("Default parameters not implemented.")

    def interpolate(self, w: ndarray) -> Union[DiscreteField,
                                               Tuple[DiscreteField, ...]]:
        """Interpolate a solution vector to quadrature points.

        Useful when a solution vector is needed in the forms, e.g., when
        evaluating functionals or when solving nonlinear problems.

        Parameters
        ----------
        w
            A solution vector.

        """
        if w.shape[0] != self.N:
            raise ValueError("Input array has wrong size.")

        if isinstance(self.elem, ElementVector):
            # ElementVector shouldn't get split here: workaround
            refs: List[Tuple[ndarray, 'AbstractBasis']] = [(np.array([]),
                                                            self)]
        else:
            refs = self.split(w)
        dfs: List[DiscreteField] = []

        # loop over solution components
        for c in range(len(refs)):
            ref = refs[c][1].basis[0][0]
            ref = ref.astuple
            fs = []

            def linear_combination(n, refn):
                """Global discrete function at quadrature points."""
                out = 0. * refn.copy()
                for i in range(self.Nbfun):
                    values = w[self.element_dofs[i]]
                    out += np.einsum('...,...j->...j', values,
                                     self.basis[i][c].get(n))
                return out

            # interpolate DiscreteField
            for n in range(len(ref)):
                if ref[n] is not None:
                    fs.append(linear_combination(n, ref[n]))
                else:
                    fs.append(None)

            dfs.append(DiscreteField(*fs))

        if len(dfs) > 1:
            return tuple(dfs)
        return dfs[0]

    def split_indices(self) -> List[ndarray]:
        """Return indices for the solution components."""
        if ((isinstance(self.elem, ElementComposite)
             or isinstance(self.elem, ElementVector))):
            nelems = (len(self.elem.elems)
                      if isinstance(self.elem, ElementComposite)
                      else self.mesh.dim())
            o = np.zeros(4, dtype=np.int64)
            output: List[ndarray] = []
            for k in range(nelems):
                e = (self.elem.elems[k]
                     if isinstance(self.elem, ElementComposite)
                     else self.elem.elem)
                output.append(np.concatenate((
                    self.nodal_dofs[o[0]:(o[0] + e.nodal_dofs)].flatten('F'),
                    self.edge_dofs[o[1]:(o[1] + e.edge_dofs)].flatten('F'),
                    self.facet_dofs[o[2]:(o[2] + e.facet_dofs)].flatten('F'),
                    (self.interior_dofs[o[3]:(o[3] + e.interior_dofs)]
                     .flatten('F'))
                )).astype(np.int64))
                o += np.array([e.nodal_dofs,
                               e.edge_dofs,
                               e.facet_dofs,
                               e.interior_dofs])
            return output
        return [np.unique(self.dofs.element_dofs)]

    def split_bases(self) -> List['AbstractBasis']:
        """Return Basis objects for the solution components."""
        if isinstance(self.elem, ElementComposite):
            return [type(self)(self.mesh, e, self.mapping,
                               quadrature=self.quadrature)
                    for e in self.elem.elems]
        elif isinstance(self.elem, ElementVector):
            return [type(self)(self.mesh, self.elem.elem, self.mapping,
                               quadrature=self.quadrature)
                    for _ in range(self.mesh.dim())]
        return [self]

    @property
    def quadrature(self):
        return self.X, self.W

    def split(self, x: ndarray) -> List[Tuple[ndarray, 'AbstractBasis']]:
        """Split a solution vector into components."""
        xs = [x[ix] for ix in self.split_indices()]
        return list(zip(xs, self.split_bases()))

    def zero_w(self) -> ndarray:
        """Return a zero array with correct dimensions for
        :func:`~skfem.assembly.asm`."""
        return np.zeros((self.nelems, 0 if self.W is None else len(self.W)))

    def zeros(self) -> ndarray:
        """Return a zero array with same dimensions as the solution."""
        return np.zeros(self.N)

    def with_element(self, elem: Element) -> 'AbstractBasis':
        """Create a copy of ``self`` that uses different element."""
        raise NotImplementedError

    def global_coordinates(self) -> ndarray:
        raise NotImplementedError

    def _normalize_interp(self, interp) -> Tuple[ndarray, ...]:

        if isinstance(interp, ndarray):
            pass
        elif callable(interp):
            interp = interp(self.global_coordinates())
        elif isinstance(interp, (float, int)):
            interp = interp + self.zero_w()
        elif isinstance(interp, (tuple, list)):
            interp = tuple(self._normalize_interp(c) for c in interp)

        return interp

    def _projection(self, interp):

        from skfem.assembly import BilinearForm, LinearForm
        from skfem.helpers import inner

        interp = self._normalize_interp(interp)
        if not isinstance(interp, tuple):
            interp = (interp,)
        assert len(interp) == len(self.basis[0])

        return (
            (BilinearForm(lambda *args: inner(args[:(len(args) - 1) // 2],
                                              args[(len(args) - 1) // 2:-1]))
             .assemble(self)),
            (LinearForm(lambda *args: inner(interp,
                                            args[:(len(args) - 1)]))
             .assemble(self)),
        )

    def project(self, interp, **kwargs):
        raise NotImplementedError

    def plot(self, x, visuals='matplotlib', **kwargs):
        """Convenience wrapper for skfem.visuals."""
        mod = importlib.import_module('skfem.visuals.{}'.format(visuals))
        return mod.plot(self, x, **kwargs)

    def plot3(self, x, visuals='matplotlib', **kwargs):
        """Convenience wrapper for skfem.visuals."""
        if not isinstance(visuals, str):
            logger.warning("Second argument, 'visuals', must be a string.")
        mod = importlib.import_module('skfem.visuals.{}'.format(visuals))
        return mod.plot3(self, x, **kwargs)

    def draw(self, visuals='matplotlib', **kwargs):
        """Convenience wrapper for skfem.visuals."""
        if not isinstance(visuals, str):
            logger.warning("First argument, 'visuals', must be a string.")
        mod = importlib.import_module('skfem.visuals.{}'.format(visuals))
        return mod.draw(self, **kwargs)
