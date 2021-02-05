r"""This module performs the finite element assembly. The basic workflow is the
following:

1. Initialize :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`.

>>> import skfem as fem
>>> m = fem.MeshTri()
>>> e = fem.ElementTriP1()
>>> m
Triangular mesh with 4 vertices and 2 elements.

2. Create :class:`~skfem.assembly.InteriorBasis` or
   :class:`~skfem.assembly.FacetBasis` objects.

>>> basis = fem.InteriorBasis(m, e)

3. Define the forms using :class:`~skfem.assembly.BilinearForm`,
   :class:`~skfem.assembly.LinearForm`, or
   :class:`~skfem.assembly.Functional`.

>>> form_a = fem.BilinearForm(lambda u, v, w: u * v)
>>> form_l = fem.LinearForm(lambda v, w: w.x[0] ** 2 * v)

Mathematically the above forms are

.. math::

    a(u,v) = \int_\Omega u v \,\mathrm{d}x
    \quad \mathrm{and} \quad
    l(v) = \int_\Omega x^2v \,\mathrm{d}x.

4. Assemble using :func:`~skfem.assembly.asm`.

>>> A = fem.asm(form_a, basis)
>>> b = fem.asm(form_l, basis)
>>> A.todense()
matrix([[0.08333333, 0.04166667, 0.04166667, 0.        ],
        [0.04166667, 0.16666667, 0.08333333, 0.04166667],
        [0.04166667, 0.08333333, 0.16666667, 0.04166667],
        [0.        , 0.04166667, 0.04166667, 0.08333333]])
>>> b
array([0.0162037 , 0.15046296, 0.06712963, 0.09953704])

"""

from typing import Union

from numpy import ndarray

from scipy.sparse import csr_matrix

from .basis import (Basis, InteriorBasis, FacetBasis, ExteriorFacetBasis,
                    InteriorFacetBasis, MortarFacetBasis)
from .dofs import Dofs, DofsView
from .form import Form, BilinearForm, LinearForm, Functional


def asm(form: Form,
        *args, **kwargs) -> Union[ndarray, csr_matrix]:
    """Perform finite element assembly.

    A shorthand for :meth:`skfem.assembly.Form.assemble`.

    """
    return form.assemble(*args, **kwargs)


__all__ = [
    "asm",
    "Basis",
    "InteriorBasis",
    "FacetBasis",
    "ExteriorFacetBasis",
    "InteriorFacetBasis",
    "MortarFacetBasis",
    "Dofs",
    "DofsView",
    "BilinearForm",
    "LinearForm",
    "Functional"]
