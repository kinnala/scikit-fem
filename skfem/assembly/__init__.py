# -*- coding: utf-8 -*-
"""This module contains rest of the tools for performing the finite element
assembly. The basic workflow of assembly is the following:

1. Initialize :class:`~skfem.mesh.Mesh` and :class:`~skfem.mesh.Element`.

>>> from skfem import *
>>> m = MeshTri()
>>> e = ElementTriP1()

2. Create :class:`~skfem.assembly.InteriorBasis` and/or
   :class:`~skfem.assembly.FacetBasis` objects.

>>> basis = InteriorBasis(m, e)

3. Define the forms using :class:`~skfem.assembly.BilinearForm`,
   :class:`~skfem.assembly.LinearForm`, and/or
   :class:`~skfem.assembly.Functional`.

>>> form_a = BilinearForm(lambda u, v, w: u * v)
>>> form_l = LinearForm(lambda v, w: w.x[0] ** 2 * v)

4. Assemble using :func:`~skfem.assembly.asm`.

>>> A = asm(form_a, basis)
>>> b = asm(form_l, basis)

The above steps assemble the matrix corresponding
to the bilinear form

.. math::

    a(u,v) = \int_0^1 \int_0^1 u(x,y)v(x,y) \,\mathrm{d}x \,\mathrm{d}y

and the vector corresponding to the linear form

.. math::

    l(v) = \int_0^1 \int_0^1 x^2v(x,y) \,\mathrm{d}x \,\mathrm{d}y

using piecewise linear basis functions.

"""

from typing import Union

from numpy import ndarray

from scipy.sparse import csr_matrix

from .basis import Basis, InteriorBasis, FacetBasis
from .dofs import Dofs
from .form import Form, BilinearForm, LinearForm, Functional,\
    bilinear_form, linear_form, functional


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
    "Dofs",
    "BilinearForm",
    "LinearForm",
    "Functional",
    "bilinear_form",
    "linear_form",
    "functional"]
